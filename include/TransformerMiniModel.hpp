#pragma once
#include "Parameter.hpp"
#include "ExecutionGraph.hpp"
#include "Optimizer.hpp"
#include "Layers.hpp"
#include <optional>

struct TransformerMiniModel
{
    TransformerMiniModel(size_t vocabSize, size_t embedDim, size_t dK, size_t numAttentionLayers, bool causal = false);

    std::shared_ptr<Node> _inputNode;
    std::shared_ptr<Node> _targetNode;
    std::shared_ptr<Node> _resultNode;
    std::shared_ptr<Node> _lossNode;

    Tensor forward(Tensor input, Tensor target);
    void backward();
    void applyGradient(float lr = 0.001f);
    void cleanGradients();

private:
    void dfsCleanGradients(std::shared_ptr<Node> node);

    std::optional<ExecutionGraph> _executionGraph;
    std::vector<std::shared_ptr<Node>> _parameterList;
    std::optional<AdamOptimizer> _optimizer;
};
