#pragma once
#include "Parameter.hpp"
#include "Node.hpp"
#include <memory>


struct Layer
{
	~Layer() = default;
	virtual std::shared_ptr<Node> forward(const std::vector<std::shared_ptr<Node>>& inputsNodes) = 0;
	std::vector<std::shared_ptr<Node>> parameters;
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

	std::shared_ptr<Node> weights;
	std::shared_ptr<Node> bias;
	Activation activation;

};

struct SingleHeadAttention : Layer
{

	SingleHeadAttention(size_t outDim, size_t inDim, Activation inputActivation);
	std::shared_ptr<Node> forward(const std::vector<std::shared_ptr<Node>>& inputsNodes) override;

	std::shared_ptr<Node> queryWeights;
	std::shared_ptr<Node> keyWeights;
	std::shared_ptr<Node> valueWeights;
	Activation activation;


};