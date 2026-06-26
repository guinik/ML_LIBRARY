#pragma once
#include "Parameter.hpp"
#include "Node.hpp"
#include <memory>
#include <string>

std::shared_ptr<Node> operator+(std::shared_ptr<Node> a, std::shared_ptr<Node> b);


struct Layer
{
	virtual ~Layer() = default;
	virtual std::shared_ptr<Node> forward(const std::vector<std::shared_ptr<Node>>& inputsNodes) = 0;
	virtual std::string layerType() const = 0;
	std::vector<std::pair<std::string, std::shared_ptr<Node>>> parameters;
};

enum class Activation
{
	NONE,
	RELU,
	SIGMOID
};

struct DenseLayer : Layer
{
	DenseLayer(size_t outDim, size_t inDim, Activation inputActivation);
	std::shared_ptr<Node> forward(const std::vector<std::shared_ptr<Node>>& inputsNodes) override;
	std::string layerType() const override { return "dense"; }

	std::shared_ptr<Node> weights;
	std::shared_ptr<Node> bias;
	Activation activation;
};

struct LayerNormLayer : Layer
{
	LayerNormLayer(size_t dim, float eps = 1e-5f);
	std::shared_ptr<Node> forward(const std::vector<std::shared_ptr<Node>>& inputsNodes) override;
	std::string layerType() const override { return "layer_norm"; }

	std::shared_ptr<Node> gamma;
	std::shared_ptr<Node> beta;
	float eps;
};

struct SingleHeadAttention : Layer
{
	SingleHeadAttention(size_t d_model, size_t d_k, bool causal = false);
	std::shared_ptr<Node> forward(const std::vector<std::shared_ptr<Node>>& inputsNodes) override;
	std::string layerType() const override { return "attention"; }

	std::shared_ptr<Node> queryWeights;
	std::shared_ptr<Node> keyWeights;
	std::shared_ptr<Node> valueWeights;
	std::shared_ptr<Node> outputWeights;
	size_t internalDim;
	bool causal;
};