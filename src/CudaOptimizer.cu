#include "CudaOptimizer.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(x) \
    do \
    { \
        cudaError_t _e = (x); \
        if (_e != cudaSuccess) \
        { \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
        } \
    } while (0)

struct CudaDeleter
{
    void operator()(float* p) const
    {
        if (p) { cudaFree(p); }
    }
};

__global__ void adamKernel(
    float* params, float* grads,
    float* m, float* v,
    float lr, float beta1, float beta2, float eps,
    float mCorr, float vCorr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }

    float g = grads[i];
    float mt = beta1 * m[i] + (1.0f - beta1) * g;
    float vt = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mt;
    v[i] = vt;
    params[i] -= lr * (mt / mCorr) / (sqrtf(vt / vCorr) + eps);
}

std::shared_ptr<float> cudaAllocMoments(size_t n)
{
    float* dPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&dPtr, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(dPtr, 0, n * sizeof(float)));
    return std::shared_ptr<float>(dPtr, CudaDeleter{});
}

void cudaAdamStep(
    float* dParams, float* dGrads,
    float* dM, float* dV,
    float lr, float beta1, float beta2, float eps,
    float mCorr, float vCorr, int n)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    adamKernel<<<blocks, threads>>>(
        dParams, dGrads, dM, dV,
        lr, beta1, beta2, eps,
        mCorr, vCorr, n);
}
