#include "ExecutionGraph.hpp"
#include <unordered_set>
#include <stdexcept>
#include <iostream>
namespace
{
	void dfsNode( std::shared_ptr<Node> node, std::unordered_set<std::shared_ptr<Node>>& doneSet,
		std::unordered_set<std::shared_ptr<Node>>& inProgress,
		std::vector<std::shared_ptr<Node>>& result)
	{
		inProgress.insert(node);
		for (auto& child : node->children)
		{
			if (!child) 
			{
				continue;
			}
			if (inProgress.find(child) != inProgress.end())
			{
				throw std::runtime_error(" Cycle found ");
			}
			
			if (doneSet.find(child) == doneSet.end())
			{
				dfsNode(child, doneSet, inProgress, result);
			}
		}
		inProgress.erase(node);
		doneSet.insert(node);
		result.push_back(node);
	};

	std::vector<std::shared_ptr<Node>> topoOrder( std::shared_ptr<Node> lossNode) 
	{

		std::unordered_set<std::shared_ptr<Node>> seenSet;
		std::unordered_set<std::shared_ptr<Node>> inProgress;
		std::vector<std::shared_ptr<Node>> result;

		dfsNode(lossNode, seenSet, inProgress, result);
		return result;
	};

}


ExecutionGraph::ExecutionGraph(std::shared_ptr<Node> lossNode)
	: _executionOrder(topoOrder(lossNode))
{
};

void ExecutionGraph::computeForward()
{
	for(auto& nodePtr : _executionOrder)
	{
		nodePtr->forward();
	}
};


void ExecutionGraph::computeBackward()
{
	for (size_t i{ _executionOrder.size() }; i-- > 0;)
	{
		_executionOrder[i]->backward();
	}
}

void ExecutionGraph::cleanGradients()
{
	for (auto& node : _executionOrder)
	{
		node->param.clearGradients();
	}
}


