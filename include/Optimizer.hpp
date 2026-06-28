#pragma once
#include "Parameter.hpp"
#include "Tensor.hpp"
#include "ExecutionGraph.hpp"
#include "Node.hpp"
#include <vector>
#include <memory>
#ifdef USE_CUDA
#include "CudaOptimizer.hpp"
#endif

class AdamOptimizer
{
public:
    AdamOptimizer(std::vector<std::shared_ptr<Node>> parameterPtrList);
    ~AdamOptimizer() = default;

    void applyGradients(float learningRate = 0.001f);

private:
    std::vector<std::shared_ptr<Node>> parameterNodePtrs;
#ifdef USE_CUDA
    std::vector<std::shared_ptr<float>> d_firstMoment;
    std::vector<std::shared_ptr<float>> d_secondMoment;
#else
    std::vector<std::vector<float>> firstMomentParameter;
    std::vector<std::vector<float>> secondMomentParameter;
#endif
    size_t numberOfSteps;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
};
