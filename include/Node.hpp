#pragma once
#include <array>
#include "Tensor.hpp"
#include "Parameter.hpp"
enum MatMulFlags : uint16_t
{
	MATMUL_NO_TRANSPOSES = 0,
	MATMUL_TRANSPOSE_A = 1 << 0,
	MATMUL_TRANSPOSE_B = 1 << 1,
};


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
			if(children[i]->param.grad.dimensions == 0)
			{
				children[i]->param.grad = gradResult[i];
			}
			else
			{
				children[i]->param.grad = children[i]->param.grad + gradResult[i];

			}
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

struct ScaleOperation : Operation
{
	float scaleFactor;
	ScaleOperation(float inputFactor) : scaleFactor(inputFactor) {};
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
};

struct MatMulOperation : Operation
{
	uint16_t flags;
	MatMulOperation(uint16_t inputFlags = MatMulFlags::MATMUL_NO_TRANSPOSES) : flags(inputFlags) {}
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

struct SoftmaxOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
};

struct CausalMaskOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
};

struct MultiplyOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
};

struct LayerNormOperation : Operation
{
	float eps;
	LayerNormOperation(float inputEps = 1e-5f) : eps(inputEps) {}
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor& output,
		const Tensor& gradOutput) const override;
};

struct CrossEntropyOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor& output,
		const Tensor& gradOutput) const override;
};

struct EmbeddingOperation : Operation
{
	Tensor forward(const std::vector<Tensor>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<Tensor>& inputs,
		const Tensor& output,
		const Tensor& gradOutput) const override;
};