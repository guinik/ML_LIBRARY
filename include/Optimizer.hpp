
#pragma once
#include "Parameter.hpp"
#include "Tensor.hpp"
#include "ExecutionGraph.hpp"
#include <vector>

class AdamOptimizer
{
public:
	AdamOptimizer(std::vector<Node<Tensor>*> parameterPtrList);
	~AdamOptimizer() = default;

	void applyGradients(float learningRate = 0.001f);

private:
	std::vector<Node<Tensor>*> parameterNodePtrs;
	std::vector<std::vector<float>> firstMomentParameter;
	std::vector<std::vector<float>> secondMomentParameter;
	size_t numberOfSteps;

	float beta1 = 0.9f;
	float beta2 = 0.999f;
	float eps = 1e-8f;
};