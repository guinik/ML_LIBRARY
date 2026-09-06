#include "Optimizer.hpp"
#include <algorithm>
#include <cmath>

AdamOptimizer::AdamOptimizer(std::vector<std::shared_ptr<Node>> parameterPtrList)
    : parameterNodePtrs(parameterPtrList), numberOfSteps(0)
{
#ifdef USE_CUDA
    d_firstMoment.reserve(parameterNodePtrs.size());
    d_secondMoment.reserve(parameterNodePtrs.size());
    for (size_t i = 0; i < parameterNodePtrs.size(); i++)
    {
        size_t n = parameterNodePtrs[i]->param.value.data->size();
        d_firstMoment.push_back(cudaAllocMoments(n));
        d_secondMoment.push_back(cudaAllocMoments(n));
    }
#else
    firstMomentParameter.reserve(parameterNodePtrs.size());
    secondMomentParameter.reserve(parameterNodePtrs.size());
    for (size_t i = 0; i < parameterNodePtrs.size(); i++)
    {
        size_t n = parameterNodePtrs[i]->param.value.data->size();
        firstMomentParameter.push_back(std::vector<float>(n, 0.0f));
        secondMomentParameter.push_back(std::vector<float>(n, 0.0f));
    }
#endif
}

void AdamOptimizer::applyGradients(float learningRate)
{
    numberOfSteps++;
    float mCorr = 1.0f - std::pow(beta1, (float)numberOfSteps);
    float vCorr = 1.0f - std::pow(beta2, (float)numberOfSteps);

    for (size_t i = 0; i < parameterNodePtrs.size(); i++)
    {
#ifdef USE_CUDA
        auto& node = parameterNodePtrs[i];
        if (!node->param.grad.onGPU()) { continue; }
        if (!node->param.value.onGPU()) { node->param.value.toGPU(); }
        cudaAdamStep(
            node->param.value.d_data.get(),
            node->param.grad.d_data.get(),
            d_firstMoment[i].get(),
            d_secondMoment[i].get(),
            learningRate, beta1, beta2, eps,
            mCorr, vCorr,
            (int)node->param.value.data->size());
#else
        auto& firstMomentList  = firstMomentParameter[i];
        auto& secondMomentList = secondMomentParameter[i];
        auto& paramValues      = *parameterNodePtrs[i]->param.value.data;
        auto& gradValues       = *parameterNodePtrs[i]->param.grad.data;
        for (size_t j = 0; j < paramValues.size(); j++)
        {
            float g  = gradValues[j];
            float mt = (1.0f - beta1) * g + beta1 * firstMomentList[j];
            float vt = (1.0f - beta2) * g * g + beta2 * secondMomentList[j];
            firstMomentList[j]  = mt;
            secondMomentList[j] = vt;
            paramValues[j] -= learningRate * (mt / mCorr) / (std::sqrt(vt / vCorr) + eps);
        }
#endif
    }
}
