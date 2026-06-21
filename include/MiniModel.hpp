
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
	Node<Matrix>* _resultNode;
	Node<Matrix>* _inputNode;
	Node<Matrix>* _lossNode;
	Node<Matrix>* _targetNode;

	Matrix forward(Matrix input, Matrix output);
	void backward();
	void applyGradient(float learning_rate = 0.001);
	void cleanGradients();
private:
	void applyGradientStep(Node<Matrix>* node, float learning_Rate );
	void dfsCleanGradients(Node<Matrix>* node);

	std::optional<ExecutionGraph> _executionGraph; //TODO : decouple from the model class
	std::vector<Node<Matrix>*> _parameterList;
	std::optional<AdamOptimizer> _optimizer;

};