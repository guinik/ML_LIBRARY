#pragma once
#include <memory>
#include <cstddef>

std::shared_ptr<float> cudaAllocMoments(size_t n);

void cudaAdamStep(
    float* dParams, float* dGrads,
    float* dM, float* dV,
    float lr, float beta1, float beta2, float eps,
    float mCorr, float vCorr, int n);
