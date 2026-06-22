
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
	std::shared_ptr<Node> _resultNode;
	std::shared_ptr<Node> _inputNode;
	std::shared_ptr<Node> _lossNode;
	std::shared_ptr<Node> _targetNode;

	Tensor forward(Tensor input, Tensor output);
	void backward();
	void applyGradient(float learning_rate = 0.001);
	void cleanGradients();
private:
	void dfsCleanGradients(std::shared_ptr<Node> node);

	std::optional<ExecutionGraph> _executionGraph; //TODO : decouple from the model class
	std::vector<std::shared_ptr<Node>> _parameterList;
	std::optional<AdamOptimizer> _optimizer;

};