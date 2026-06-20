#include "Matrix.hpp"
#include "MiniModel.hpp"
#include <stdexcept>
#include <cmath>
MiniModel::MiniModel(std::vector<size_t> layerSizes)
{
	_inputNode = new Node<Matrix>(Operation::LEAF);
	_inputNode->requires_grad = false;

	Node<Matrix>* currentNode = _inputNode;

	for (size_t layer = 0; layer + 1 < layerSizes.size(); layer++)
	{
		size_t inDim = layerSizes[layer];
		size_t outDim = layerSizes[layer + 1];

		// weight: (outDim, inDim) so weight * input -> (outDim, 1)
		Node<Matrix>* weightNode = new Node<Matrix>(Operation::LEAF);
		weightNode->param.value = Matrix(2, { outDim, inDim });
		weightNode->param.value.randomize(1.0f / std::sqrt((float)inDim));

		Node<Matrix>* matmulNode = new Node<Matrix>(Operation::MULTIPLY);
		matmulNode->children = { weightNode, currentNode };

		// bias: (outDim, 1)
		Node<Matrix>* biasNode = new Node<Matrix>(Operation::LEAF);
		biasNode->param.value = Matrix(2, { outDim, 1 });
		biasNode->param.value.randomize(1.0f / std::sqrt((float)inDim));

		Node<Matrix>* addNode = new Node<Matrix>(Operation::ADD);
		addNode->children = { matmulNode, biasNode };

		bool isLastLayer = (layer + 2 == layerSizes.size());
		if (!isLastLayer)
		{
			Node<Matrix>* reluNode = new Node<Matrix>(Operation::RELU);
			reluNode->children = { addNode, nullptr };
			currentNode = reluNode;
		}
		else
		{
			currentNode = addNode;
		}
	}

	_resultNode = currentNode;

	Node<Matrix>* substractNode = new Node<Matrix>(Operation::SUBSTRACT);
	substractNode->requires_grad = false;
	Node<Matrix>* targetNode = new Node<Matrix>(Operation::LEAF);
	targetNode->requires_grad = false;
	substractNode->children = { _resultNode, targetNode };

	Node<Matrix>* lossNode = new Node<Matrix>(Operation::SQUARE);
	lossNode->children = { substractNode, nullptr };
	lossNode->requires_grad = false;

	_lossNode = lossNode;
	_targetNode = targetNode;
}

Matrix MiniModel::forward(Matrix input, Matrix output)
{
	_inputNode->param.value = input;
	_targetNode->param.value = output;
	forwardStep(_lossNode);

	return _resultNode->param.value;
};

void MiniModel::forwardStep(Node<Matrix>* node)
{
	if (!node) return;

	forwardStep(node->children[0]);
	forwardStep(node->children[1]);
	auto left = node->children[0];
	auto right = node->children[1];


	switch (node->op)
	{
	case Operation::LEAF: break;

	case Operation::ADD:
		if (left && right) {
			node->param.value = left->param.value + right->param.value;
		}
		break;

	case Operation::MULTIPLY:
		if (left && right)
		{
			node->param.value = left->param.value * right->param.value;
		}
		break;
	case Operation::SUBSTRACT:
		if (left && right)
		{
			node->param.value = left->param.value - right->param.value;
		}
		break;

	case Operation::RELU:
		if (left)
		{
			Matrix result = left->param.value;
			for (size_t i{ 0 }; i < left->param.value.data.size(); i++)
			{
				result.data[i] = (result.data[i] > 0) ? result.data[i] : 0;
			}
			node->param.value = result;
		}
		break;

	case Operation::SQUARE:
		{
			if (left)
			{
				node->param.value = left->param.value.square();
			}
			break;
		}
	}

};

void MiniModel::backwardStep(Node<Matrix>* node)
{
	if (!node) return;

	auto left = node->children[0];
	auto right = node->children[1];


	switch (node->op)
	{
	case Operation::LEAF: break;

	case Operation::ADD:
		if (left && right)
		{
			left->param.grad = left->param.grad + node->param.grad;
			right->param.grad = right->param.grad + node->param.grad;
		}
		break;
	case Operation::MULTIPLY:
		if (left && right)
		{
			left->param.grad = left->param.grad + node->param.grad * right->param.value.transpose();
			right->param.grad = right->param.grad + left->param.value.transpose() * node->param.grad;

		}
		break;
	case Operation::SUBSTRACT:
		if (left && right)
		{
			right->param.grad = right->param.grad - node->param.grad;
			left->param.grad = left->param.grad + node->param.grad;

		}
		break;

	case Operation::RELU:
		if (left)
		{
			Matrix result = left->param.value;
			for (size_t i{ 0 }; i < left->param.value.data.size(); i++)
			{
				result.data[i] = (left->param.grad.data[i] > 0) ? node->param.grad.data[i] : 0;
			}
			left->param.grad = left->param.grad + result;

		}
		break;
	case Operation::SQUARE:
		if (left)
		{
			Matrix localGrad = left->param.value;
			for (size_t i = 0; i < localGrad.data.size(); i++)
				localGrad.data[i] = 2.0f * left->param.value.data[i] * node->param.grad.data[i];
			left->param.grad = left->param.grad + localGrad;
		}
		break;
	}



	backwardStep(node->children[0]);
	backwardStep(node->children[1]);
};

void MiniModel::backward()
{
	_lossNode->param.grad = _lossNode->param.value;
	_lossNode->param.grad.fillValues(1.0f);
	backwardStep(_lossNode);
};

void MiniModel::dfsCleanGradients(Node<Matrix>* node)
{
	if (!node) return;
	dfsCleanGradients(node->children[0]);
	dfsCleanGradients(node->children[1]);

	node->param.clearGradients();

};

void MiniModel::cleanGradients()
{
	dfsCleanGradients(_lossNode);

};


void MiniModel::applyGradient(float learning_rate)
{
	applyGradientStep(_lossNode, learning_rate);
};


void MiniModel::applyGradientStep(Node<Matrix>* node, float learning_rate)
{
	if (!node) return;

	if (node->requires_grad)
	{
		node->param.value = node->param.value - node->param.grad * learning_rate;
	}
	if (node->children[0]) applyGradientStep(node->children[0], learning_rate);
	if (node->children[1]) applyGradientStep(node->children[1], learning_rate);

};
