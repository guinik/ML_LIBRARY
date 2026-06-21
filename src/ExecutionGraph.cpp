#include "ExecutionGraph.hpp"
#include <unordered_set>
#include <stdexcept>
namespace
{
	template<typename T>
	void dfsNode(Node<T>* node, std::unordered_set<const Node<T>*>& doneSet,
		std::unordered_set<const Node<T>*>& inProgress,
		std::vector<Node<T>*>& result)
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
				dfsNode<T>(child, doneSet, inProgress, result);
			}
		}
		inProgress.erase(node);
		doneSet.insert(node);
		result.push_back(node);
	};
	template <typename T>
	std::vector<Node<T>*> topoOrder( Node<T>* lossNode) 
	{

		std::unordered_set<const Node<T>*> seenSet;
		std::unordered_set<const Node<T>*> inProgress;
		std::vector<Node<T>*> result;

		dfsNode<T>(lossNode, seenSet, inProgress, result);
		return result;
	};


	void computeNodeForward(Node<Matrix>* node)
	{
		auto& left = node->children[0];
		auto& right = node->children[1];
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
					Matrix result(left->param.value.dimensions, left->param.value.shape);
					const float* src = left->param.value.data->data();
					float* dst = result.data->data();
					size_t n = left->param.value.data->size();
					for (size_t i{ 0 }; i < n; i++)
						dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
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



	void computeNodeBackward(Node<Matrix>* node)
	{
	
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
					auto rightTranspose = right->param.value;
					rightTranspose.transpose();
					auto leftTranspose = left->param.value;
					leftTranspose.transpose();

					left->param.grad = left->param.grad + node->param.grad * rightTranspose;
					right->param.grad = right->param.grad + leftTranspose * node->param.grad;

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
					Matrix result(left->param.value.dimensions, left->param.value.shape);
					const float* val = left->param.value.data->data();
					const float* grad = node->param.grad.data->data();
					float* dst = result.data->data();
					size_t n = left->param.value.data->size();
					for (size_t i{ 0 }; i < n; i++)
						dst[i] = val[i] > 0.0f ? grad[i] : 0.0f;
					left->param.grad = left->param.grad + result;
				}
				break;
			case Operation::SQUARE:
				if (left)
				{
					Matrix localGrad(left->param.value.dimensions, left->param.value.shape);
					const float* val = left->param.value.data->data();
					const float* grad = node->param.grad.data->data();
					float* dst = localGrad.data->data();
					size_t n = left->param.value.data->size();
					for (size_t i = 0; i < n; i++)
						dst[i] = 2.0f * val[i] * grad[i];
					left->param.grad = left->param.grad + localGrad;
				}
				break;
		}

	
	}

}


ExecutionGraph::ExecutionGraph(Node<Matrix>* lossNode)
	: _executionOrder(topoOrder<Matrix>(lossNode))
{
};

void ExecutionGraph::computeForward()
{
	for(auto& nodePtr : _executionOrder)
	{
		computeNodeForward(nodePtr);
	}
};


void ExecutionGraph::computeBackward()
{

	for(size_t i{ _executionOrder.size()}; i-- > 0;)
	{
		computeNodeBackward(_executionOrder[i]);
	}
};


