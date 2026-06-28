#include "CudaPool.hpp"
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <mutex>

static std::mutex g_poolMutex;
static std::unordered_map<size_t, std::vector<float*>> g_pool;

float* cudaPoolAlloc(size_t nFloats)
{
    size_t bytes = nFloats * sizeof(float);
    {
        std::lock_guard<std::mutex> lock(g_poolMutex);
        auto it = g_pool.find(bytes);
        if (it != g_pool.end() && !it->second.empty())
        {
            float* ptr = it->second.back();
            it->second.pop_back();
            return ptr;
        }
    }
    float* ptr = nullptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
}

void cudaPoolFree(float* ptr, size_t nFloats)
{
    if (!ptr) { return; }
    size_t bytes = nFloats * sizeof(float);
    std::lock_guard<std::mutex> lock(g_poolMutex);
    g_pool[bytes].push_back(ptr);
}

void cudaPoolFlush()
{
    std::lock_guard<std::mutex> lock(g_poolMutex);
    for (auto& [bytes, ptrs] : g_pool)
    {
        for (float* p : ptrs)
        {
            cudaFree(p);
        }
        ptrs.clear();
    }
    g_pool.clear();
}

void PoolDeleter::operator()(float* p) const
{
    cudaPoolFree(p, nFloats);
}
