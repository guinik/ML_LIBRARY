#include "Matrix.hpp"
#include "MiniModel.hpp"
#include <stdexcept>
#include <cmath>
MiniModel::MiniModel(std::vector<size_t> layerSizes)
{
	_inputNode = new Node<Matrix>(Operation::LEAF);

	Node<Matrix>* currentNode = _inputNode;

	for (size_t layer = 0; layer + 1 < layerSizes.size(); layer++)
	{
		size_t inDim = layerSizes[layer];
		size_t outDim = layerSizes[layer + 1];

		// weight: (outDim, inDim) so weight * input -> (outDim, 1)
		Node<Matrix>* weightNode = new Node<Matrix>(Operation::LEAF);
		weightNode->param.value = Matrix(2, { outDim, inDim });
		weightNode->param.value.randomize(1.0f / std::sqrt((float)inDim));
		_parameterList.push_back(weightNode);


		Node<Matrix>* matmulNode = new Node<Matrix>(Operation::MULTIPLY);
		matmulNode->children = { weightNode, currentNode };

		// bias: (outDim, 1)
		Node<Matrix>* biasNode = new Node<Matrix>(Operation::LEAF);
		biasNode->param.value = Matrix(2, { outDim, 1 });
		biasNode->param.value.randomize(1.0f / std::sqrt((float)inDim));
		_parameterList.push_back(biasNode);


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
	Node<Matrix>* targetNode = new Node<Matrix>(Operation::LEAF);
	substractNode->children = { _resultNode, targetNode };

	Node<Matrix>* lossNode = new Node<Matrix>(Operation::SQUARE);
	lossNode->children = { substractNode, nullptr };

	_lossNode = lossNode;
	_targetNode = targetNode;

	_executionGraph.emplace(lossNode);
	_optimizer.emplace(_parameterList);


}

Matrix MiniModel::forward(Matrix input, Matrix output)
{
	_inputNode->param.value = input;
	_targetNode->param.value = output;
	if(_executionGraph.has_value())
	{
		_executionGraph.value().computeForward();
	}

	return _resultNode->param.value;
};



void MiniModel::backward()
{
	_lossNode->param.grad = _lossNode->param.value;
	_lossNode->param.grad.fillValues(1.0f);
	
	if (_executionGraph.has_value())
	{
		_executionGraph.value().computeBackward();
	}

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
	if(_optimizer.has_value())
	{
		_optimizer.value().applyGradients(learning_rate);
	}
};