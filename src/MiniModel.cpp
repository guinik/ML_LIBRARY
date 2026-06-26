#include "Tensor.hpp"
#include "MiniModel.hpp"
#include "Layers.hpp"
#include <stdexcept>
#include <cmath>
MiniModel::MiniModel(std::vector<size_t> layerSizes)
{
	_inputNode = std::make_shared<Node>();

	std::shared_ptr<Node> currentNode = _inputNode;

	for (size_t layer = 0; layer + 1 < layerSizes.size(); layer++)
	{
		size_t inDim = layerSizes[layer];
		size_t outDim = layerSizes[layer + 1];

		bool isLastLayer = (layer + 2 == layerSizes.size());
		if (!isLastLayer)
		{
			auto denseLayer = DenseLayer(outDim, inDim, Activation::SIGMOID);
			currentNode = denseLayer.forward({currentNode});
			for (auto& [name, node] : denseLayer.parameters)
			{
				_parameterList.push_back(node);
			}
		}
		else
		{
			auto denseLayer = DenseLayer(outDim, inDim, Activation::NONE);
			currentNode = denseLayer.forward({ currentNode });
			for (auto& [name, node] : denseLayer.parameters)
			{
				_parameterList.push_back(node);
			}
		}
		
	}

	_resultNode = currentNode;

	std::shared_ptr<Node> substractNode = std::make_shared<Node>(std::make_shared<SubtractOperation>());
	std::shared_ptr<Node> targetNode = std::make_shared<Node>();
	substractNode->children = { _resultNode, targetNode };

	std::shared_ptr<Node> lossNode = std::make_shared<Node>(std::make_shared<SquareOperation>());
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

void MiniModel::dfsCleanGradients(std::shared_ptr<Node> node)
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