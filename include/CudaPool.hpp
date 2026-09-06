#pragma once
#include <memory>
#include <cstddef>

float* cudaPoolAlloc(size_t nFloats);
void cudaPoolFree(float* ptr, size_t nFloats);
void cudaPoolFlush();

struct PoolDeleter
{
    size_t nFloats;
    void operator()(float* p) const;
};

inline std::shared_ptr<float> makeGpuBuffer(size_t nFloats)
{
    return std::shared_ptr<float>(cudaPoolAlloc(nFloats), PoolDeleter{nFloats});
}
