#include "Layers.hpp"
#include "Tensor.hpp"
#include <string>
#include <cmath>
inline std::shared_ptr<Node> makeNode(std::shared_ptr<Operation> op, std::shared_ptr<Node> a, 
	std::shared_ptr<Node> b = nullptr)
{

	
	std::shared_ptr<Node> resultNode = std::make_shared<Node>(std::move(op));
	resultNode->children = { std::move(a), std::move(b) };

	return resultNode;

}

std::shared_ptr<Node> operator+(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
	return makeNode(std::make_shared<AddOperation>(), std::move(a), std::move(b));
}

std::shared_ptr<Node> matMul(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
	return makeNode(std::make_shared<MatMulOperation>(static_cast<uint16_t>(MatMulFlags::MATMUL_HAS_BATCH_B | MatMulFlags::MATMUL_VECTOR_B)), std::move(a), std::move(b));
}

std::shared_ptr<Node> relu(std::shared_ptr<Node> a)
{
	return makeNode(std::make_shared<ReluOperation>(), std::move(a));
}

std::shared_ptr<Node> sigmoid(std::shared_ptr<Node> a)
{
	return makeNode(std::make_shared<SigmoidOperation>(), std::move(a));
}


DenseLayer::DenseLayer(size_t outDim, size_t inDim, Activation inputActivation)
{
	size_t numDimensionsWeight{ 2 };
	size_t numDimensionsBias{ 1 };
	weights = std::make_shared<Node>();
	bias = std::make_shared<Node>();
	weights->param.value = Tensor(numDimensionsWeight, { outDim, inDim});
	bias->param.value = Tensor(numDimensionsBias, { outDim });

	weights->param.value.randomize(1.0f / std::sqrt((float)inDim));
	bias->param.value.randomize(1.0f / std::sqrt((float)inDim));
	parameters = { weights, bias };
	activation = inputActivation;

}

std::shared_ptr<Node> DenseLayer::forward(const std::vector<std::shared_ptr<Node>>& inputsNodes)
{
	
	switch(activation)
	{
		case Activation::RELU:
		{
			return relu(matMul(weights, inputsNodes[0]) + bias);
		}
		case Activation::SIGMOID:
		{
			return sigmoid(matMul(weights, inputsNodes[0]) + bias);
		}
		default:
		{
			return matMul(weights, inputsNodes[0]) + bias;

		}
	}

};