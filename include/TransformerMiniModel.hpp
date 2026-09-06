#pragma once
#include "Parameter.hpp"
#include "ExecutionGraph.hpp"
#include "Optimizer.hpp"
#include "Layers.hpp"
#include <optional>
#include <string>
#include <map>

struct TransformerMiniModel
{
    TransformerMiniModel(size_t vocabSize, size_t embedDim, size_t dK, size_t numAttentionLayers, bool causal = false);

    std::shared_ptr<Node> _inputNode;
    std::shared_ptr<Node> _posNode;
    std::shared_ptr<Node> _targetNode;
    std::shared_ptr<Node> _resultNode;
    std::shared_ptr<Node> _lossNode;

    Tensor forward(Tensor input, Tensor target);
    void backward();
    void applyGradient(float lr = 0.001f);
    void cleanGradients();

    void save(const std::string& path) const;
    void load(const std::string& path);
    std::map<std::string, Tensor> stateDict() const;
    size_t paramCount() const;

private:
    void registerLayer(const std::string& name, Layer& layer);
    void registerLayer(Layer& layer);

    std::optional<ExecutionGraph> _executionGraph;
    std::vector<std::shared_ptr<Node>> _parameterList;
    std::vector<std::pair<std::string, std::shared_ptr<Node>>> _namedParams;
    std::map<std::string, int> _layerTypeCounts;
    std::optional<AdamOptimizer> _optimizer;
};
