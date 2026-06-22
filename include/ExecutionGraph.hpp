#pragma once
#include <vector>
#include "Parameter.hpp"
#include "Tensor.hpp"
#include <optional>
//using ExecutionGraph = std::vector<Node<Tensor>*>;


class ExecutionGraph
{
public:
	// TODO, it might make more sense, to define from target node and then build on top the loss
	ExecutionGraph (Node<Tensor>* lossNode);
	~ExecutionGraph() = default;
	
	ExecutionGraph(ExecutionGraph&&) noexcept = default;
	ExecutionGraph(const ExecutionGraph&) noexcept = default;

	ExecutionGraph& operator=(const ExecutionGraph&) = delete;
	ExecutionGraph& operator=(ExecutionGraph&&) = delete;


	void computeForward();
	void computeBackward();

private:
	const std::vector<Node<Tensor>*> _executionOrder;// defined as forward pass order


};


