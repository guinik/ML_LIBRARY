#include "TransformerMiniModel.hpp"
#include "Serialization.hpp"

void TransformerMiniModel::registerLayer(const std::string& name, Layer& layer)
{
    for (auto& [paramName, node] : layer.parameters)
    {
        _namedParams.push_back({name + "." + paramName, node});
        _parameterList.push_back(node);
    }
}

void TransformerMiniModel::registerLayer(Layer& layer)
{
    const std::string type = layer.layerType();
    auto& count = _layerTypeCounts[type];
    std::string name = (count == 0) ? type : type + "_" + std::to_string(count);
    count++;
    registerLayer(name, layer);
}

TransformerMiniModel::TransformerMiniModel(size_t vocabSize, size_t embedDim, size_t dK, size_t numAttentionLayers, bool causal)
{
    _inputNode = std::make_shared<Node>();

    auto embLayer = DenseLayer(embedDim, vocabSize, Activation::NONE);
    auto current = embLayer.forward({_inputNode});
    registerLayer("emb", embLayer);

    for (size_t i = 0; i < numAttentionLayers; i++)
    {
        auto attn = SingleHeadAttention(embedDim, dK, causal);
        auto attended = attn.forward({current});
        registerLayer(attn);

        auto ln = LayerNormLayer(embedDim);
        current = ln.forward({current + attended});
        registerLayer(ln);
    }

    auto outLayer = DenseLayer(vocabSize, embedDim, Activation::NONE);
    _resultNode = outLayer.forward({current});
    registerLayer("out", outLayer);

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
    {
        _executionGraph.value().computeForward();
    }
    return _resultNode->param.value;
}

void TransformerMiniModel::backward()
{
    _lossNode->param.grad = _lossNode->param.value;
    _lossNode->param.grad.fillValues(1.0f);
    if (_executionGraph.has_value())
    {
        _executionGraph.value().computeBackward();
    }
}

void TransformerMiniModel::dfsCleanGradients(std::shared_ptr<Node> node)
{
    if (!node)
    {
        return;
    }
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
    {
        _optimizer.value().applyGradients(lr);
    }
}

std::map<std::string, Tensor> TransformerMiniModel::stateDict() const
{
    std::map<std::string, Tensor> sd;
    for (auto& [name, node] : _namedParams)
    {
        sd[name] = node->param.value;
    }
    return sd;
}

void TransformerMiniModel::save(const std::string& path) const
{
    save_model(path, stateDict());
}

void TransformerMiniModel::load(const std::string& path)
{
    auto sd = load_model(path);
    for (auto& [name, node] : _namedParams)
    {
        node->param.value = sd.at(name);
    }
}
