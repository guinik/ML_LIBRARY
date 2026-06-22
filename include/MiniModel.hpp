
#pragma once
#include "Parameter.hpp"
#include "ExecutionGraph.hpp"
#include <array>
#include <optional>
#include "Optimizer.hpp"
class MiniModel
{
public:
	MiniModel(std::vector<size_t> layerSizes);
	~MiniModel() = default;
	Node<Tensor>* _resultNode;
	Node<Tensor>* _inputNode;
	Node<Tensor>* _lossNode;
	Node<Tensor>* _targetNode;

	Tensor forward(Tensor input, Tensor output);
	void backward();
	void applyGradient(float learning_rate = 0.001);
	void cleanGradients();
private:
	void applyGradientStep(Node<Tensor>* node, float learning_Rate );
	void dfsCleanGradients(Node<Tensor>* node);

	std::optional<ExecutionGraph> _executionGraph; //TODO : decouple from the model class
	std::vector<Node<Tensor>*> _parameterList;
	std::optional<AdamOptimizer> _optimizer;

};