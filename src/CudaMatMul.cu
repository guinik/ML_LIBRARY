#include "CudaMatMul.hpp"
#include "Node.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

static cublasHandle_t g_handle = nullptr;

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

struct CudaDeleter
{
    void operator()(float* p) const
    {
        if (p)
        {
            cudaFree(p);
        }
    }
};

void cudaMatMulInit()
{
    CUBLAS_CHECK(cublasCreate(&g_handle));
}

void cudaMatMulShutdown()
{
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
    size_t bytes = data->size() * sizeof(float);
    float* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    CUDA_CHECK(cudaMemcpy(ptr, data->data(), bytes, cudaMemcpyHostToDevice));
    d_data = std::shared_ptr<float>(ptr, CudaDeleter{});
}

void Tensor::toCPU() const
{
    if (!d_data)
    {
        return;
    }
    size_t bytes = data->size() * sizeof(float);
    CUDA_CHECK(cudaMemcpy(data->data(), d_data.get(), bytes, cudaMemcpyDeviceToHost));
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

    long long strideDevB = (B.data->size() > K * N) ? (long long)(K * N) : 0LL;
    long long strideDevA = (A.data->size() > M * K) ? (long long)(M * K) : 0LL;
    long long strideDevC = (long long)(M * N);

    size_t bytesC = batchCount * M * N * sizeof(float);
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        g_handle,
        tB ? CUBLAS_OP_T : CUBLAS_OP_N,
        tA ? CUBLAS_OP_T : CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        B.d_data.get(), tB ? (int)K : (int)N,
        strideDevB,
        A.d_data.get(), tA ? (int)M : (int)K,
        strideDevA,
        &beta,
        d_C, (int)N,
        strideDevC,
        (int)batchCount
    ));

    Tensor result(resultShape.size(), resultShape);
    result.d_data = std::shared_ptr<float>(d_C, CudaDeleter{});
    return result;
}
