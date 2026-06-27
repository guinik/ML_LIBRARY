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
    _posNode = std::make_shared<Node>();

    auto embLayer = EmbeddingLayer(vocabSize, embedDim);
    auto posLayer = EmbeddingLayer(512, embedDim);

    auto current = embLayer.forward({_inputNode}) + posLayer.forward({_posNode});
    registerLayer("emb", embLayer);
    registerLayer("pos_emb", posLayer);

    for (size_t i = 0; i < numAttentionLayers; i++)
    {
        auto attn = SingleHeadAttention(embedDim, dK, causal);
        auto attended = attn.forward({current});
        registerLayer(attn);

        auto ln1 = LayerNormLayer(embedDim);
        auto normed = ln1.forward({current + attended});
        registerLayer(ln1);

        auto ffn1 = DenseLayer(embedDim * 4, embedDim, Activation::RELU);
        auto ffn2 = DenseLayer(embedDim, embedDim * 4, Activation::NONE);
        auto ffnOut = ffn2.forward({ffn1.forward({normed})});
        registerLayer(ffn1);
        registerLayer(ffn2);

        auto ln2 = LayerNormLayer(embedDim);
        current = ln2.forward({normed + ffnOut});
        registerLayer(ln2);
    }

    auto outLayer = DenseLayer(vocabSize, embedDim, Activation::NONE);
    _resultNode = outLayer.forward({current});
    registerLayer("out", outLayer);

    _targetNode = std::make_shared<Node>();

    _lossNode = std::make_shared<Node>(std::make_shared<CrossEntropyOperation>());
    _lossNode->children = {_resultNode, _targetNode};

    _executionGraph.emplace(_lossNode);
    _optimizer.emplace(_parameterList);
}

Tensor TransformerMiniModel::forward(Tensor input, Tensor target)
{
    size_t batch = input.shape[0];
    size_t seq   = input.shape[1];

    Tensor pos(2, {batch, seq});
    for (size_t b = 0; b < batch; b++)
    {
        for (size_t s = 0; s < seq; s++)
        {
            (*pos.data)[b * seq + s] = static_cast<float>(s);
        }
    }

    _inputNode->param.value = input;
    _posNode->param.value = pos;
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

size_t TransformerMiniModel::paramCount() const
{
    size_t total = 0;
    for (auto& [name, node] : _namedParams)
    {
        total += node->param.value.data->size();
    }
    return total;
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
