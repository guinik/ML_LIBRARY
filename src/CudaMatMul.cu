#include "CudaMatMul.hpp"
#include "CudaPool.hpp"
#include "Node.hpp"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

static cublasHandle_t g_handle = nullptr;
static cublasLtHandle_t g_ltHandle = nullptr;

#define CUDA_CHECK(x) \
    do \
    { \
        cudaError_t _e = (x); \
        if (_e != cudaSuccess) \
        { \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
        } \
    } while (0)

#define CUBLAS_CHECK(x) \
    do \
    { \
        cublasStatus_t _s = (x); \
        if (_s != CUBLAS_STATUS_SUCCESS) \
        { \
            throw std::runtime_error("cuBLAS error " + std::to_string(_s)); \
        } \
    } while (0)

#define CUBLASLT_CHECK(x) \
    do \
    { \
        cublasStatus_t _s = (x); \
        if (_s != CUBLAS_STATUS_SUCCESS) \
        { \
            throw std::runtime_error("cuBLASLt error " + std::to_string(_s)); \
        } \
    } while (0)

// Cached per-shape cuBLASLt plan: descriptor creation is cheap, but
// cublasLtMatmulAlgoGetHeuristic does real algorithm-search work, so we run it
// once per distinct (m,n,k,batchCount,transposes,broadcast) shape and reuse the
// chosen algorithm on every subsequent call with that same shape.
struct LtPlan
{
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatrixLayout_t leftDesc = nullptr;
    cublasLtMatrixLayout_t rightDesc = nullptr;
    cublasLtMatrixLayout_t cDesc = nullptr;
    cublasLtMatmulAlgo_t algo{};
};

struct MatmulShapeKey
{
    int64_t m, n, k, batchCount;
    cublasOperation_t opLeft, opRight;
    bool bcastLeft, bcastRight;

    bool operator<(const MatmulShapeKey& o) const
    {
        if (m != o.m) return m < o.m;
        if (n != o.n) return n < o.n;
        if (k != o.k) return k < o.k;
        if (batchCount != o.batchCount) return batchCount < o.batchCount;
        if (opLeft != o.opLeft) return opLeft < o.opLeft;
        if (opRight != o.opRight) return opRight < o.opRight;
        if (bcastLeft != o.bcastLeft) return bcastLeft < o.bcastLeft;
        return bcastRight < o.bcastRight;
    }
};

static std::map<MatmulShapeKey, LtPlan> g_ltPlans;
static const size_t LT_WORKSPACE_FLOATS = (4u * 1024u * 1024u) / sizeof(float);

void cudaMatMulInit()
{
    CUBLAS_CHECK(cublasCreate(&g_handle));
    CUBLASLT_CHECK(cublasLtCreate(&g_ltHandle));
}

void cudaMatMulShutdown()
{
    for (auto& [key, plan] : g_ltPlans)
    {
        if (plan.leftDesc)  { cublasLtMatrixLayoutDestroy(plan.leftDesc); }
        if (plan.rightDesc) { cublasLtMatrixLayoutDestroy(plan.rightDesc); }
        if (plan.cDesc)     { cublasLtMatrixLayoutDestroy(plan.cDesc); }
        if (plan.opDesc)    { cublasLtMatmulDescDestroy(plan.opDesc); }
    }
    g_ltPlans.clear();

    if (g_ltHandle)
    {
        cublasLtDestroy(g_ltHandle);
        g_ltHandle = nullptr;
    }
    if (g_handle)
    {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
}

void Tensor::toGPU() const
{
    if (d_data)
    {
        return;
    }
    size_t n = nelems();
    float* ptr = cudaPoolAlloc(n);
    CUDA_CHECK(cudaMemcpy(ptr, data->data(), n * sizeof(float), cudaMemcpyHostToDevice));
    d_data = std::shared_ptr<float>(ptr, PoolDeleter{n});
}

void Tensor::toCPU() const
{
    if (!d_data)
    {
        return;
    }
    size_t n = nelems();
    if (data->size() != n) { data->resize(n); }
    CUDA_CHECK(cudaMemcpy(data->data(), d_data.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
    d_data.reset();
}

void Tensor::invalidateGPU() const
{
    d_data.reset();
}

static void swapLast2(std::vector<size_t>& v)
{
    std::swap(v[v.size() - 1], v[v.size() - 2]);
}

Tensor cudaMatMul(const Tensor& A, const Tensor& B, uint16_t mask)
{
    bool tA = (mask & MatMulFlags::MATMUL_TRANSPOSE_A) != 0;
    bool tB = (mask & MatMulFlags::MATMUL_TRANSPOSE_B) != 0;

    std::vector<size_t> shapeA = A.shape;
    std::vector<size_t> shapeB = B.shape;

    if (tA) { swapLast2(shapeA); }
    if (tB) { swapLast2(shapeB); }

    if (shapeA.size() < 2 || shapeB.size() < 2)
    {
        throw std::runtime_error("cudaMatMul: operands must have >= 2 dims");
    }

    if (shapeA.size() < shapeB.size())
    {
        size_t need = shapeB.size() - shapeA.size();
        shapeA.insert(shapeA.begin(), need, 1);
    }
    else if (shapeB.size() < shapeA.size())
    {
        size_t need = shapeA.size() - shapeB.size();
        shapeB.insert(shapeB.begin(), need, 1);
    }

    size_t M = shapeA[shapeA.size() - 2];
    size_t K = shapeA.back();
    size_t N = shapeB.back();

    size_t batchDims = shapeA.size() - 2;
    size_t batchCount = 1;
    std::vector<size_t> batchShape;
    for (size_t i = 0; i < batchDims; i++)
    {
        size_t d = std::max(shapeA[i], shapeB[i]);
        batchShape.push_back(d);
        batchCount *= d;
    }

    std::vector<size_t> resultShape = batchShape;
    resultShape.push_back(M);
    resultShape.push_back(N);

    if (!A.onGPU())
    {
        A.toGPU();
    }
    if (!B.onGPU())
    {
        B.toGPU();
    }

    long long strideDevB = (B.nelems() > K * N) ? (long long)(K * N) : 0LL;
    long long strideDevA = (A.nelems() > M * K) ? (long long)(M * K) : 0LL;
    long long strideDevC = (long long)(M * N);

    size_t nC = batchCount * M * N;
    float* d_C = cudaPoolAlloc(nC);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Same row-major-via-column-major trick as before: treat B as the "left"
    // cuBLAS operand and A as the "right" one, with M/N swapped.
    cublasOperation_t opLeft  = tB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opRight = tA ? CUBLAS_OP_T : CUBLAS_OP_N;
    int64_t cublasM = (int64_t)N;   // cublas "m" = our N
    int64_t cublasN = (int64_t)M;   // cublas "n" = our M
    int64_t cublasK = (int64_t)K;
    int64_t ldLeft  = tB ? (int64_t)K : (int64_t)N;
    int64_t ldRight = tA ? (int64_t)M : (int64_t)K;

    // If exactly one operand is broadcast across the batch dimension (the
    // "activations @ shared weight" pattern -- every forward Dense/attention
    // projection, and the input-side gradient in backward), fold the batch
    // dimension directly into cublasM/cublasN instead of driving a strided
    // batch GEMM. Row-major [batch, rows, cols] and [(batch*rows), cols] are
    // the exact same memory layout, so this is a free reshape that turns many
    // small batched GEMMs into one large, better-utilized GEMM -- the same
    // approach frameworks like PyTorch use for Linear layers. The broadcast
    // operand (the weight) is untouched either way since it has no batch
    // dimension to begin with.
    //
    // This is only a valid free reshape when batch and the folded dimension
    // are ADJACENT in physical memory: A is physically [batch,M,K] (batch,M
    // adjacent) only when tA is false -- if tA is true, A is physically
    // [batch,K,M] and batch/M are separated by K, so folding would require
    // real data movement. Symmetric argument for B requires tB true.
    bool bBroadcasts = (strideDevB == 0) && (batchCount > 1);
    bool aBroadcasts = (strideDevA == 0) && (batchCount > 1);
    bool canFoldIntoM = bBroadcasts && !aBroadcasts && !tA;
    bool canFoldIntoN = aBroadcasts && !bBroadcasts && tB;

    int64_t effBatchCount = (int64_t)batchCount;
    long long effStrideLeft  = strideDevB;
    long long effStrideRight = strideDevA;
    long long effStrideC     = strideDevC;

    if (canFoldIntoM)
    {
        cublasN *= (int64_t)batchCount;
        effBatchCount = 1;
        effStrideLeft = 0;
        effStrideRight = 0;
        effStrideC = 0;
    }
    else if (canFoldIntoN)
    {
        cublasM *= (int64_t)batchCount;
        effBatchCount = 1;
        effStrideLeft = 0;
        effStrideRight = 0;
        effStrideC = 0;
    }

    // C is our own freshly-allocated, always-tightly-packed buffer, so its
    // leading dimension must track cublasM directly -- including when
    // canFoldIntoN grew it, since unlike leftDesc/rightDesc there's no
    // transpose to route the folded value into the (unconstrained) cols slot.
    int64_t ldC = cublasM;

    MatmulShapeKey key{ cublasM, cublasN, cublasK, effBatchCount, opLeft, opRight,
                         strideDevB == 0, strideDevA == 0 };

    auto it = g_ltPlans.find(key);
    if (it == g_ltPlans.end())
    {
        LtPlan plan{};
        CUBLASLT_CHECK(cublasLtMatmulDescCreate(&plan.opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opLeft, sizeof(opLeft)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan.opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opRight, sizeof(opRight)));

        int64_t leftRows  = (opLeft  == CUBLAS_OP_N) ? cublasM : cublasK;
        int64_t leftCols  = (opLeft  == CUBLAS_OP_N) ? cublasK : cublasM;
        int64_t rightRows = (opRight == CUBLAS_OP_N) ? cublasK : cublasN;
        int64_t rightCols = (opRight == CUBLAS_OP_N) ? cublasN : cublasK;

        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.leftDesc,  CUDA_R_32F, (uint64_t)leftRows,  (uint64_t)leftCols,  ldLeft));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.rightDesc, CUDA_R_32F, (uint64_t)rightRows, (uint64_t)rightCols, ldRight));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan.cDesc,     CUDA_R_32F, (uint64_t)cublasM, (uint64_t)cublasN, ldC));

        int32_t batchCountI = (int32_t)effBatchCount;
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(plan.leftDesc,  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCountI, sizeof(batchCountI)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(plan.rightDesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCountI, sizeof(batchCountI)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(plan.cDesc,     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCountI, sizeof(batchCountI)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(plan.leftDesc,  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &effStrideLeft,  sizeof(effStrideLeft)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(plan.rightDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &effStrideRight, sizeof(effStrideRight)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(plan.cDesc,     CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &effStrideC,     sizeof(effStrideC)));

        cublasLtMatmulPreference_t pref = nullptr;
        CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        size_t workspaceBytes = LT_WORKSPACE_FLOATS * sizeof(float);
        CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes)));

        cublasLtMatmulHeuristicResult_t heuristic{};
        int returnedCount = 0;
        CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
            g_ltHandle, plan.opDesc,
            plan.leftDesc, plan.rightDesc, plan.cDesc, plan.cDesc,
            pref, 1, &heuristic, &returnedCount));
        cublasLtMatmulPreferenceDestroy(pref);

        if (returnedCount == 0)
        {
            throw std::runtime_error("cuBLASLt: no valid algorithm found for this matmul shape");
        }
        plan.algo = heuristic.algo;

        it = g_ltPlans.emplace(key, plan).first;
    }

    const LtPlan& plan = it->second;
    float* workspace = cudaPoolAlloc(LT_WORKSPACE_FLOATS);

    CUBLASLT_CHECK(cublasLtMatmul(
        g_ltHandle, plan.opDesc,
        &alpha,
        B.d_data.get(), plan.leftDesc,
        A.d_data.get(), plan.rightDesc,
        &beta,
        d_C, plan.cDesc,
        d_C, plan.cDesc,
        &plan.algo,
        workspace, LT_WORKSPACE_FLOATS * sizeof(float),
        0));

    cudaPoolFree(workspace, LT_WORKSPACE_FLOATS);

    Tensor result(resultShape.size(), resultShape);
    result.d_data = std::shared_ptr<float>(d_C, PoolDeleter{nC});
    return result;
}
