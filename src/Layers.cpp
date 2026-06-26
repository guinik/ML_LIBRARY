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

std::shared_ptr<Node> operator*(std::shared_ptr<Node> a, float b)
{
	return makeNode(std::make_shared<ScaleOperation>(b), std::move(a));
}

std::shared_ptr<Node> matMul(std::shared_ptr<Node> a, std::shared_ptr<Node> b, uint16_t flags = MatMulFlags::MATMUL_TRANSPOSE_B)
{
	return makeNode(std::make_shared<MatMulOperation>(flags), std::move(a), std::move(b));
}

std::shared_ptr<Node> relu(std::shared_ptr<Node> a)
{
	return makeNode(std::make_shared<ReluOperation>(), std::move(a));
}

std::shared_ptr<Node> sigmoid(std::shared_ptr<Node> a)
{
	return makeNode(std::make_shared<SigmoidOperation>(), std::move(a));
}

std::shared_ptr<Node> softmax(std::shared_ptr<Node> a)
{
	return makeNode(std::make_shared<SoftmaxOperation>(), std::move(a));
}

std::shared_ptr<Node> causalMask(std::shared_ptr<Node> a)
{
	return makeNode(std::make_shared<CausalMaskOperation>(), std::move(a));
}

std::shared_ptr<Node> multiply(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
	return makeNode(std::make_shared<MultiplyOperation>(), std::move(a), std::move(b));
}

LayerNormLayer::LayerNormLayer(size_t dim, float inputEps) : eps(inputEps)
{
	gamma = std::make_shared<Node>();
	beta  = std::make_shared<Node>();
	gamma->param.value = Tensor(1, {dim});
	beta->param.value  = Tensor(1, {dim});
	gamma->param.value.fillValues(1.0f);
	beta->param.value.fillValues(0.0f);
	parameters = {{"gamma", gamma}, {"beta", beta}};
}

std::shared_ptr<Node> LayerNormLayer::forward(const std::vector<std::shared_ptr<Node>>& inputsNodes)
{
	auto normed = makeNode(std::make_shared<LayerNormOperation>(eps), inputsNodes[0]);
	auto scaled = multiply(normed, gamma);
	return scaled + beta;
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
	parameters = {{"weight", weights}, {"bias", bias}};
	activation = inputActivation;

}

std::shared_ptr<Node> DenseLayer::forward(const std::vector<std::shared_ptr<Node>>& inputsNodes)
{
	
	switch(activation)
	{
		case Activation::RELU:
		{
			return relu(matMul(inputsNodes[0], weights) + bias);
		}
		case Activation::SIGMOID:
		{
			return sigmoid(matMul(inputsNodes[0], weights) + bias);
		}
		default:
		{
			return matMul(inputsNodes[0], weights) + bias;

		}
	}

};




SingleHeadAttention::SingleHeadAttention(size_t d_model, size_t d_k, bool inputCausal)
{
	size_t numDimensionsWeight{ 2 };
	queryWeights = std::make_shared<Node>();
	keyWeights = std::make_shared<Node>();
	valueWeights = std::make_shared<Node>();
	outputWeights = std::make_shared<Node>();
	queryWeights->param.value = Tensor(numDimensionsWeight, { d_k, d_model });
	keyWeights->param.value = Tensor(numDimensionsWeight, { d_k, d_model });
	valueWeights->param.value = Tensor(numDimensionsWeight, { d_k, d_model });
	outputWeights->param.value = Tensor(numDimensionsWeight, { d_model, d_k });

	queryWeights->param.value.randomize(1.0f / std::sqrt((float)d_model));
	keyWeights->param.value.randomize(1.0f / std::sqrt((float)d_model));
	valueWeights->param.value.randomize(1.0f / std::sqrt((float)d_model));
	outputWeights->param.value.randomize(1.0f / std::sqrt((float)d_k));

	internalDim = d_k;
	causal = inputCausal;
	parameters = {{"wq", queryWeights}, {"wk", keyWeights}, {"wv", valueWeights}, {"wo", outputWeights}};
}

std::shared_ptr<Node> SingleHeadAttention::forward(const std::vector<std::shared_ptr<Node>>& inputsNodes)
{
	auto Q = matMul(inputsNodes[0], queryWeights);
	auto K = matMul(inputsNodes[0], keyWeights);
	auto V = matMul(inputsNodes[0], valueWeights);
	auto QK = matMul(Q, K);
	auto normalized = QK * (1.0f / std::sqrt((float)internalDim));
	if (causal)
	{
		normalized = causalMask(normalized);
	}
	auto softmaxRes = softmax(normalized);
	auto attention = matMul(softmaxRes, V, MatMulFlags::MATMUL_NO_TRANSPOSES);
	auto output = matMul(attention, outputWeights);
	return output;
};