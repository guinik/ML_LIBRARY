#include "Optimizer.hpp"
#include <algorithm>
#include <cmath>

AdamOptimizer::AdamOptimizer(std::vector<std::shared_ptr<Node>> parameterPtrList) : parameterNodePtrs(parameterPtrList)
{
	firstMomentParameter.reserve(parameterNodePtrs.size());
	secondMomentParameter.reserve(parameterNodePtrs.size());
	for (size_t i{ 0 }; i < parameterNodePtrs.size(); i++)
	{
		size_t numOfParamsNode = parameterNodePtrs[i]->param.value.data->size();
		firstMomentParameter.push_back(std::vector<float>(numOfParamsNode, 0.0f));
		secondMomentParameter.push_back(std::vector<float>(numOfParamsNode, 0.0f));
	}
	numberOfSteps = 0;
};


void AdamOptimizer::applyGradients(float learningRate) 
{
	numberOfSteps++;
	float firstMomentCorrection = (1 - std::pow(beta1, (float)numberOfSteps));
	float secondMomentCorrection = (1 - std::pow(beta2,(float)numberOfSteps));
	for (size_t i{ 0 }; i < parameterNodePtrs.size(); i++)
	{
		auto& firstMomentList = firstMomentParameter[i];
		auto& secondMomentList = secondMomentParameter[i];
		auto& paramValues = *parameterNodePtrs[i]->param.value.data;
		auto& gradValues = *parameterNodePtrs[i]->param.grad.data;
		for (size_t paramIdx{ 0 }; paramIdx < paramValues.size(); paramIdx++)
		{// Think about this access makes sense or no.
			float firstMomentT = (1.0f - beta1) * gradValues[paramIdx] + beta1 * firstMomentList[paramIdx];
			float secondMomentT = (1.0f - beta2) * gradValues[paramIdx] * gradValues[paramIdx] + beta2 * secondMomentList[paramIdx];
			firstMomentList[paramIdx] = firstMomentT;
			secondMomentList[paramIdx] = secondMomentT;
			float firstMomentCorrected = firstMomentT / firstMomentCorrection;
			float secondMomentCorrected = secondMomentT / secondMomentCorrection;

			paramValues[paramIdx] -= learningRate *(firstMomentCorrected) / (std::sqrt(secondMomentCorrected) + eps);
		}
	}

};


