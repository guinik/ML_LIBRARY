#pragma once
#include <array>
#include "Parameter.hpp"

struct Operation
{
	virtual ~Operation() = default;
	virtual Tensor forward(const std::vector<Tensor>& inputs) const = 0;
	virtual std::vector<Tensor> backward(const std::vector<Tensor>& inputs,
		const Tensor& outputs,
		const Tensor& gradOutput) const = 0;
};

// node should be at most binary. 
struct Node
{
	Node(std::shared_ptr<Operation> inputOp) : op(std::move(inputOp)) {};
	Node(){};

	~Node() = default;
	Parameter param{};
	std::array <std::shared_ptr<Node>, 2> children = {nullptr, nullptr};
	std::shared_ptr<Operation> op;

	void forward() 
	{
		if (!op) return;
		std::vector<Tensor> inputTensor;
		for(auto& child : children)
		{
			if (!child) continue;
			inputTensor.push_back(child->param.value);
		}
		param.value = op->forward(inputTensor);
	}

	void backward()
	{
		if (!op) return;
		std::vector<Tensor> inputTensor;
		for (auto& child : children)
		{
			if (!child) continue;
			inputTensor.push_back(child->param.value);
		}
		std::vector<Tensor> gradResult = op->backward(inputTensor, param.value, param.grad);
		for (size_t i{0}; i < children.size(); i++)
		{
			if (!children[i]) continue;
			children[i]->param.grad = children[i]->param.grad + gradResult[i];
		}

	};
};


struct AddOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
};

struct SubtractOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
};

struct MatMulOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
};

struct ReluOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
};

struct SquareOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
};

struct SigmoidOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
};
