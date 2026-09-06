#pragma once
#include <vector>
#include <utility>
#include "Tensor.hpp"
#include "Parameter.hpp"
#ifdef USE_CUDA
#include "CudaOps.hpp"
#endif
enum MatMulFlags : uint16_t
{
	MATMUL_NO_TRANSPOSES = 0,
	MATMUL_TRANSPOSE_A = 1 << 0,
	MATMUL_TRANSPOSE_B = 1 << 1,
};


struct Operation
{
	virtual ~Operation() = default;
	virtual Tensor forward(const std::vector<const Tensor*>& inputs) const = 0;
	virtual std::vector<Tensor> backward(const std::vector<const Tensor*>& inputs,
		const Tensor& outputs,
		const Tensor& gradOutput) const = 0;
	virtual bool isGpuOp() const { return false; }
};


// children may be any arity, gradResult from op->backward must match children.size()
struct Node
{
	Node(std::shared_ptr<Operation> inputOp) : op(std::move(inputOp)) {};
	Node(){};

	~Node() = default;
	Parameter param{};
	std::vector<std::shared_ptr<Node>> children;
	std::shared_ptr<Operation> op;

	void forward()
	{
		if (!op)
		{
			return;
		}
		std::vector<const Tensor*> inputTensor;
		for (auto& child : children)
		{
			if (!child)
			{
				continue;
			}
#ifdef USE_CUDA
			if (op->isGpuOp())
			{
				child->param.value.toGPU();
			}
			else
			{
				child->param.value.toCPU();
			}
#endif
			inputTensor.push_back(&child->param.value);
		}
		param.value = op->forward(inputTensor);
	}

	void backward()
	{
		if (!op)
		{
			return;
		}
		std::vector<const Tensor*> inputTensor;
		for (auto& child : children)
		{
			if (!child)
			{
				continue;
			}
#ifdef USE_CUDA
			if (op->isGpuOp())
			{
				child->param.value.toGPU();
			}
			else
			{
				child->param.value.toCPU();
			}
#endif
			inputTensor.push_back(&child->param.value);
		}
#ifdef USE_CUDA
		if (!op->isGpuOp())
		{
			param.value.toCPU();
			param.grad.toCPU();
		}
#endif
		std::vector<Tensor> gradResult = op->backward(inputTensor, param.value, param.grad);
		for (size_t i{0}; i < children.size(); i++)
		{
			if (!children[i])
			{
				continue;
			}
			if (children[i]->param.grad.dimensions == 0)
			{
				children[i]->param.grad = std::move(gradResult[i]);
			}
			else
			{
#ifdef USE_CUDA
				children[i]->param.grad = cudaAdd(children[i]->param.grad, gradResult[i]);
#else
				children[i]->param.grad = children[i]->param.grad + gradResult[i];
#endif
			}
		}
	}
};


struct AddOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct SubtractOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct ScaleOperation : Operation
{
	float scaleFactor;
	ScaleOperation(float inputFactor) : scaleFactor(inputFactor) {};
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct MatMulOperation : Operation
{
	uint16_t flags;
	MatMulOperation(uint16_t inputFlags = MatMulFlags::MATMUL_NO_TRANSPOSES) : flags(inputFlags) {}
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct ReluOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct SquareOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct SigmoidOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct SoftmaxOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct CausalMaskOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

// (batch, seq, heads*dHead) -> (batch, heads, seq, dHead), used to fan a fused Q/K/V
// projection out into per-head attention matrices
struct SplitHeadsOperation : Operation
{
	size_t numHeads;
	SplitHeadsOperation(size_t heads) : numHeads(heads) {}
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

// inverse of SplitHeadsOperation: (batch, heads, seq, dHead) -> (batch, seq, heads*dHead)
struct MergeHeadsOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct MultiplyOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor&,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct LayerNormOperation : Operation
{
	float eps;
	LayerNormOperation(float inputEps = 1e-5f) : eps(inputEps) {}
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor& output,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

// fused normalize(x) * gamma + beta, children order is {x, gamma, beta}
struct LayerNormAffineOperation : Operation
{
	float eps;
	LayerNormAffineOperation(float inputEps = 1e-5f) : eps(inputEps) {}
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor& output,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct CrossEntropyOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor& output,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

struct EmbeddingOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor& output,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};

// fused x @ weight^T + bias, children order is {x, weight, bias}
struct DenseOperation : Operation
{
	Tensor forward(const std::vector<const Tensor*>& inputs) const override;
	std::vector<Tensor> backward(
		const std::vector<const Tensor*>& inputs,
		const Tensor& output,
		const Tensor& gradOutput) const override;
	bool isGpuOp() const override { return true; }
};