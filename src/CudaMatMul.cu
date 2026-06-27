#include "CudaMatMul.hpp"
#include "Node.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

static cublasHandle_t g_handle = nullptr;

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t _e = (x);                                                 \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("CUDA: ") +                  \
                                     cudaGetErrorString(_e));                  \
    } while (0)

#define CUBLAS_CHECK(x)                                                        \
    do {                                                                       \
        cublasStatus_t _s = (x);                                              \
        if (_s != CUBLAS_STATUS_SUCCESS)                                       \
            throw std::runtime_error("cuBLAS error " + std::to_string(_s));   \
    } while (0)

void cudaMatMulInit()
{
    CUBLAS_CHECK(cublasCreate(&g_handle));
}

void cudaMatMulShutdown()
{
    if (g_handle) { cublasDestroy(g_handle); g_handle = nullptr; }
}

static void swapLast2(std::vector<size_t>& v)
{
    std::swap(v[v.size() - 1], v[v.size() - 2]);
}

// Compute row-major C = op_A(A) x op_B(B) via cuBLAS.
//
// cuBLAS is column-major. The identity:
//   row-major C = op_A(A) x op_B(B)
//   col-major C^T = op_B(B)^T x op_A(A)^T
//
// So we pass B as cuBLAS "A" and A as cuBLAS "B", swapping the
// transpose flags accordingly.
Tensor cudaMatMul(const Tensor& A, const Tensor& B, uint16_t mask)
{
    bool tA = (mask & MatMulFlags::MATMUL_TRANSPOSE_A) != 0;
    bool tB = (mask & MatMulFlags::MATMUL_TRANSPOSE_B) != 0;

    // Work on logical shapes (with transpose flags applied).
    std::vector<size_t> shapeA   = A.shape;
    std::vector<size_t> shapeB   = B.shape;
    std::vector<size_t> stridesA = A.strides;
    std::vector<size_t> stridesB = B.strides;

    if (tA) { swapLast2(shapeA); swapLast2(stridesA); }
    if (tB) { swapLast2(shapeB); swapLast2(stridesB); }

    if (shapeA.size() < 2 || shapeB.size() < 2)
        throw std::runtime_error("cudaMatMul: operands must have >= 2 dims");

    // Equalise batch ranks by prepending 1s (broadcast).
    if (shapeA.size() < shapeB.size()) {
        size_t need = shapeB.size() - shapeA.size();
        shapeA.insert(shapeA.begin(), need, 1);
        stridesA.insert(stridesA.begin(), need, 0);
    } else if (shapeB.size() < shapeA.size()) {
        size_t need = shapeA.size() - shapeB.size();
        shapeB.insert(shapeB.begin(), need, 1);
        stridesB.insert(stridesB.begin(), need, 0);
    }

    size_t M = shapeA[shapeA.size() - 2];
    size_t K = shapeA.back();
    size_t N = shapeB.back();

    size_t batchDims  = shapeA.size() - 2;
    size_t batchCount = 1;
    std::vector<size_t> batchShape;
    for (size_t i = 0; i < batchDims; i++) {
        size_t d = std::max(shapeA[i], shapeB[i]);
        batchShape.push_back(d);
        batchCount *= d;
    }

    std::vector<size_t> resultShape = batchShape;
    resultShape.push_back(M);
    resultShape.push_back(N);
    Tensor result(resultShape.size(), resultShape);

    // If an operand has only one matrix's worth of data it is being broadcast:
    // pass stride = 0 so cuBLAS reuses the same matrix for every batch.
    long long strideDevB = (B.data->size() > K * N) ? (long long)(K * N) : 0LL;
    long long strideDevA = (A.data->size() > M * K) ? (long long)(M * K) : 0LL;
    long long strideDevC = (long long)(M * N);

    size_t bytesA = A.data->size() * sizeof(float);
    size_t bytesB = B.data->size() * sizeof(float);
    size_t bytesC = batchCount * M * N * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    CUDA_CHECK(cudaMemcpy(d_A, A.data->data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data->data(), bytesB, cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cuBLAS dimensions: m=N, n=M (row-major trick swaps the output dims).
    // Leading dimensions:
    //   lda (for B in cuBLAS): physical last dim of B in memory
    //     tB=false → B stored [K×N], last dim = N
    //     tB=true  → B stored [N×K], last dim = K
    //   ldb (for A in cuBLAS): physical last dim of A in memory
    //     tA=false → A stored [M×K], last dim = K
    //     tA=true  → A stored [K×M], last dim = M
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        g_handle,
        tB ? CUBLAS_OP_T : CUBLAS_OP_N,   // op on cuBLAS "A" = our B
        tA ? CUBLAS_OP_T : CUBLAS_OP_N,   // op on cuBLAS "B" = our A
        (int)N, (int)M, (int)K,
        &alpha,
        d_B, tB ? (int)K : (int)N,        // lda
        strideDevB,
        d_A, tA ? (int)M : (int)K,        // ldb
        strideDevA,
        &beta,
        d_C, (int)N,                       // ldc
        strideDevC,
        (int)batchCount
    ));

    CUDA_CHECK(cudaMemcpy(result.data->data(), d_C, bytesC, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}
