#include "Node.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <string>
namespace
{
	void swapLastTwoDims(std::vector<size_t>& A)
	{
		std::swap(A[A.size() - 1], A[A.size() - 2]);
	};

	Tensor matMul(const Tensor& A, const Tensor& B, uint16_t mask)
	{


		std::vector<size_t> shapeA = A.shape;
		std::vector<size_t> shapeB = B.shape;

		std::vector<size_t> stridesA = A.strides;
		std::vector<size_t> stridesB = B.strides;

		bool hasBatchA = (mask & MatMulFlags::MATMUL_HAS_BATCH_A);
		bool vectorA = (mask & MatMulFlags::MATMUL_VECTOR_A);
		bool vectorB = (mask & MatMulFlags::MATMUL_VECTOR_B);

		bool promotedA = false, promotedB = false;

		if (vectorA)
		{
			size_t insertPos = hasBatchA ? shapeA.size() - 1 : 0;
			shapeA.insert(shapeA.begin() + insertPos, 1);
			stridesA.insert(stridesA.begin() + insertPos, 0);
			promotedA = true;
		}
		if (vectorB)
		{
			shapeB.push_back(1);
			stridesB.push_back(0);
			promotedB = true;
		}


		bool transposeA = (mask & MatMulFlags::MATMUL_TRANSPOSE_A);
		bool transposeB = (mask & MatMulFlags::MATMUL_TRANSPOSE_B);

		if(transposeA)
		{
			swapLastTwoDims(shapeA);
			swapLastTwoDims(stridesA);
		}
		if (transposeB)
		{
			swapLastTwoDims(shapeB);
			swapLastTwoDims(stridesB);
		}

		if (shapeA.size() < 2 || shapeB.size() < 2)
			throw std::runtime_error("Operands must have at least 1 dimension");

		size_t K = shapeA.back();
		if (K != shapeB[shapeB.size() - 2])
			throw std::runtime_error("Inner dimensions don't match for multiplication");

		size_t batchRankA = shapeA.size() - 2;
		size_t batchRankB = shapeB.size() - 2;

		if (batchRankA < batchRankB)
		{
			size_t need = batchRankB - batchRankA;
			shapeA.insert(shapeA.begin(), need, 1);
			stridesA.insert(stridesA.begin(), need, 0);
		}
		else if (batchRankB < batchRankA)
		{
			size_t need = batchRankA - batchRankB;
			shapeB.insert(shapeB.begin(), need, 1);
			stridesB.insert(stridesB.begin(), need, 0);
		}


		size_t batchDimsA= shapeA.size() - 2;
		std::vector<size_t> batchShape;
		for (size_t i = 0; i < batchDimsA; i++)
		{
			if (shapeA[i] != shapeB[i] && shapeA[i] != 1 && shapeB[i] != 1)
				throw std::runtime_error("Batch dimension sizes don't match");
			batchShape.push_back(std::max(shapeA[i], shapeB[i]));
		}

		size_t M = shapeA[shapeA.size() - 2];
		size_t N = shapeB.back();

		size_t batchCount = 1;
		for (auto d : batchShape) batchCount *= d;

		std::vector<size_t> resultShape = batchShape;
		resultShape.push_back(M);
		resultShape.push_back(N);
		Tensor result(resultShape.size(), resultShape);

		const size_t strideA_row = stridesA[stridesA.size() - 2];
		const size_t strideA_col = stridesA[stridesA.size() - 1];
		const size_t strideB_row = stridesB[stridesB.size() - 2];
		const size_t strideB_col = stridesB[stridesB.size() - 1];
		const size_t strideR_row = result.strides[result.strides.size() - 2];
		const size_t strideR_col = result.strides[result.strides.size() - 1];

		const float* dataA = A.data->data();
		const float* dataB = B.data->data();
		float* dataR = result.data->data();

		for (size_t b{ 0 }; b < batchCount; b++)
		{
			size_t baseA = b * stridesA[0];
			size_t baseB = b * stridesB[0];
			size_t baseR = b * M * N;
			for (size_t m{ 0 }; m < M; m++)
			{
				size_t rowA = baseA + m * strideA_row;
				size_t rowR = baseR + m * strideR_row;
				for (size_t n{ 0 }; n < N; n++)
				{
					float sum{};
					size_t colB = baseB + n * strideB_col;
					for (size_t k{ 0 }; k < K; k++)
					{
						sum += dataA[rowA + k * strideA_col] * dataB[colB + k * strideB_row];
					}
					dataR[rowR + n * strideR_col] = sum;
				}
			}
		}

		if (promotedA)
		{
			result.shape.erase(result.shape.end() - 2);
			result.strides.erase(result.strides.end() - 2);
		};
		if (promotedB)
		{
			result.shape.erase(result.shape.end() - 1);
			result.strides.erase(result.strides.end() - 1);
		};


		return result;
	}

} // namespace




Tensor AddOperation::forward(const std::vector<Tensor>& inputs) const
{
	return inputs[0] + inputs[1];	
};

std::vector<Tensor> AddOperation::backward(const std::vector<Tensor>&,
	const Tensor&,
	const Tensor& gradOutput) const {
	
	std::vector<Tensor> result;
	result.reserve(2);
	result = { gradOutput, gradOutput };
	return result;
};

Tensor SubtractOperation::forward(const std::vector<Tensor>& inputs) const
{
	return inputs[0] - inputs[1];
};

std::vector<Tensor> SubtractOperation::backward(const std::vector<Tensor>&,
	const Tensor&,
	const Tensor& gradOutput) const {

	std::vector<Tensor> result;
	result.reserve(2);

	Tensor leftGrad = gradOutput;
	result.push_back(leftGrad);
	Tensor rightGrad = gradOutput * (-1.0f);
	result.push_back(rightGrad);

	return result;
};




Tensor MatMulOperation::forward(const std::vector<Tensor>& inputs) const
{
	return matMul(inputs[0], inputs[1], flags);
};
std::vector<Tensor> MatMulOperation::backward(const std::vector<Tensor>& inputs,
	const Tensor&,
	const Tensor& gradOutput) const {

	std::vector<Tensor> result;
	result.reserve(2);

	// inputs[0]=weights [outDim,inDim], inputs[1]=activations [batch,inDim], gradOutput [batch,outDim]
	// dL/dW = grad^T @ X  →  [outDim,batch] @ [batch,inDim] = [outDim,inDim]
	Tensor leftGrad = matMul(gradOutput, inputs[1], MatMulFlags::MATMUL_TRANSPOSE_A);
	result.push_back(leftGrad);

	// dL/dX = grad @ W  →  [batch,outDim] @ [outDim,inDim] = [batch,inDim]
	Tensor rightGrad = matMul(gradOutput, inputs[0], MatMulFlags::MATMUL_NO_TRANSPOSES);
	result.push_back(rightGrad);

	return result;
};



Tensor ReluOperation::forward(const std::vector<Tensor>& inputs) const
{
	Tensor leftTensor = inputs[0];
	Tensor result(leftTensor.dimensions, leftTensor.shape);
	const float* src = leftTensor.data->data();
	float* dst = result.data->data();
	size_t n = leftTensor.data->size();
	for (size_t i{ 0 }; i < n; i++)
		dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
	return result;

};
std::vector<Tensor> ReluOperation::backward(const std::vector<Tensor>& inputs,
	const Tensor&,
	const Tensor& gradOutput) const {

	std::vector<Tensor> result;
	result.reserve(1);
	Tensor resultLeft(inputs[0].dimensions, inputs[0].shape);
	const float* val = inputs[0].data->data();
	const float* grad = gradOutput.data->data();
	float* dst = resultLeft.data->data();
	size_t n = inputs[0].data->size();
	for (size_t i{ 0 }; i < n; i++)
		dst[i] = val[i] > 0.0f ? grad[i] : 0.0f;

	result.push_back(resultLeft);
	return result;
};



Tensor SquareOperation::forward(const std::vector<Tensor>& inputs) const
{
	return inputs[0].square();

};
std::vector<Tensor> SquareOperation::backward(const std::vector<Tensor>& inputs,
	const Tensor&,
	const Tensor& gradOutput) const {

	Tensor localGrad(inputs[0].dimensions, inputs[0].shape);
	const float* val = inputs[0].data->data();
	const float* grad = gradOutput.data->data();
	float* dst = localGrad.data->data();
	size_t n = inputs[0].data->size();
	for (size_t i = 0; i < n; i++)
		dst[i] = 2.0f * val[i] * grad[i];
		
		
	std::vector<Tensor> result;
	result.push_back(localGrad);
	return result;
};



Tensor SigmoidOperation::forward(const std::vector<Tensor>& inputs) const
{
	Tensor leftTensor = inputs[0];
	Tensor result(leftTensor.dimensions, leftTensor.shape);
	const float* src = leftTensor.data->data();
	float* dst = result.data->data();
	size_t n = leftTensor.data->size();
	
	for (size_t i{ 0 }; i < n; i++)
		dst[i] = (1) / (1 + std::exp(-src[i]));
	return result;

};
std::vector<Tensor> SigmoidOperation::backward(const std::vector<Tensor>& inputs,
	const Tensor& output,
	const Tensor& gradOutput) const {
	std::vector<Tensor> result;
	result.reserve(1);
	Tensor resultLeft(inputs[0].dimensions, inputs[0].shape);
	const float* outputVal = output.data->data();
	const float* grad = gradOutput.data->data();
	float* dst = resultLeft.data->data();
	size_t n = inputs[0].data->size();
	for (size_t i{ 0 }; i < n; i++)
		dst[i] = outputVal[i]* (1- outputVal[i]) * grad[i];

	result.push_back(resultLeft);
	return result;
};

Tensor unbroadcastGrad(const Tensor& grad, const std::vector<size_t>& targetShape)
{
	std::vector<size_t> gradShape = grad.shape;
	size_t extraDims = gradShape.size() - targetShape.size();
	Tensor reduced = grad;
	for (size_t d = 0; d < extraDims; d++)
	{
		reduced = reduced.sumAxis(0);
	}
	return reduced;
}
