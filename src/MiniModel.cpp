#include "Tensor.hpp"
#include "MiniModel.hpp"
#include <stdexcept>
#include <cmath>
MiniModel::MiniModel(std::vector<size_t> layerSizes)
{
	_inputNode = new Node<Tensor>(Operation::LEAF);

	Node<Tensor>* currentNode = _inputNode;

	for (size_t layer = 0; layer + 1 < layerSizes.size(); layer++)
	{
		size_t inDim = layerSizes[layer];
		size_t outDim = layerSizes[layer + 1];

		// weight: (outDim, inDim) so weight * input -> (outDim, 1)
		Node<Tensor>* weightNode = new Node<Tensor>(Operation::LEAF);
		weightNode->param.value = Tensor(2, { outDim, inDim });
		weightNode->param.value.randomize(1.0f / std::sqrt((float)inDim));
		_parameterList.push_back(weightNode);


		Node<Tensor>* matmulNode = new Node<Tensor>(Operation::MULTIPLY);
		matmulNode->children = { weightNode, currentNode };

		// bias: (outDim, 1)
		Node<Tensor>* biasNode = new Node<Tensor>(Operation::LEAF);
		biasNode->param.value = Tensor(2, { outDim, 1 });
		biasNode->param.value.randomize(1.0f / std::sqrt((float)inDim));
		_parameterList.push_back(biasNode);


		Node<Tensor>* addNode = new Node<Tensor>(Operation::ADD);
		addNode->children = { matmulNode, biasNode };

		bool isLastLayer = (layer + 2 == layerSizes.size());
		if (!isLastLayer)
		{
			Node<Tensor>* reluNode = new Node<Tensor>(Operation::RELU);
			reluNode->children = { addNode, nullptr };
			currentNode = reluNode;
			
		}
		else
		{
			currentNode = addNode;
		}
	}

	_resultNode = currentNode;

	Node<Tensor>* substractNode = new Node<Tensor>(Operation::SUBSTRACT);
	Node<Tensor>* targetNode = new Node<Tensor>(Operation::LEAF);
	substractNode->children = { _resultNode, targetNode };

	Node<Tensor>* lossNode = new Node<Tensor>(Operation::SQUARE);
	lossNode->children = { substractNode, nullptr };

	_lossNode = lossNode;
	_targetNode = targetNode;

	_executionGraph.emplace(lossNode);
	_optimizer.emplace(_parameterList);


}

Tensor MiniModel::forward(Tensor input, Tensor output)
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

void MiniModel::dfsCleanGradients(Node<Tensor>* node)
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