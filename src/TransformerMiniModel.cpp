#include "TransformerMiniModel.hpp"

TransformerMiniModel::TransformerMiniModel(size_t vocabSize, size_t embedDim, size_t dK, size_t numAttentionLayers, bool causal)
{
    _inputNode = std::make_shared<Node>();

    auto embLayer = DenseLayer(embedDim, vocabSize, Activation::NONE);
    auto current = embLayer.forward({_inputNode});
    for (auto& p : embLayer.parameters)
    {
        _parameterList.push_back(p);
    }

    for (size_t i = 0; i < numAttentionLayers; i++)
    {
        auto attn = SingleHeadAttention(embedDim, dK, causal);
        auto attended = attn.forward({current});
        for (auto& p : attn.parameters)
        {
            _parameterList.push_back(p);
        }

        auto ln = LayerNormLayer(embedDim);
        current = ln.forward({current + attended});
        for (auto& p : ln.parameters)
        {
            _parameterList.push_back(p);
        }
    }

    // Output projection: [seq, embedDim] → [seq, vocabSize]
    auto outLayer = DenseLayer(vocabSize, embedDim, Activation::NONE);
    _resultNode = outLayer.forward({current});
    for (auto& p : outLayer.parameters)
    {
        _parameterList.push_back(p);
    }

    _targetNode = std::make_shared<Node>();

    auto diffNode = std::make_shared<Node>(std::make_shared<SubtractOperation>());
    diffNode->children = {_resultNode, _targetNode};

    _lossNode = std::make_shared<Node>(std::make_shared<SquareOperation>());
    _lossNode->children = {diffNode, nullptr};

    _executionGraph.emplace(_lossNode);
    _optimizer.emplace(_parameterList);
}

Tensor TransformerMiniModel::forward(Tensor input, Tensor target)
{
    _inputNode->param.value = input;
    _targetNode->param.value = target;
    if (_executionGraph.has_value())
        _executionGraph.value().computeForward();
    return _resultNode->param.value;
}

void TransformerMiniModel::backward()
{
    _lossNode->param.grad = _lossNode->param.value;
    _lossNode->param.grad.fillValues(1.0f);
    if (_executionGraph.has_value())
        _executionGraph.value().computeBackward();
}

void TransformerMiniModel::dfsCleanGradients(std::shared_ptr<Node> node)
{
    if (!node) return;
    dfsCleanGradients(node->children[0]);
    dfsCleanGradients(node->children[1]);
    node->param.clearGradients();
}

void TransformerMiniModel::cleanGradients()
{
    dfsCleanGradients(_lossNode);
}

void TransformerMiniModel::applyGradient(float lr)
{
    if (_optimizer.has_value())
        _optimizer.value().applyGradients(lr);
}
