#pragma once
#include <vector>
#include "Parameter.hpp"
#include "Tensor.hpp"
#include "Node.hpp"
#include <optional>


class ExecutionGraph
{
public:
	// TODO, it might make more sense, to define from target node and then build on top the loss
	ExecutionGraph (std::shared_ptr<Node> lossNode);
	~ExecutionGraph() = default;
	
	ExecutionGraph(ExecutionGraph&&) noexcept = default;
	ExecutionGraph(const ExecutionGraph&) noexcept = default;

	ExecutionGraph& operator=(const ExecutionGraph&) = delete;
	ExecutionGraph& operator=(ExecutionGraph&&) = delete;


	void computeForward();
	void computeBackward();
	void cleanGradients();

private:
	const std::vector<std::shared_ptr<Node>> _executionOrder;


};


