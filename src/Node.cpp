#include "Node.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <string>
#ifdef USE_CUDA
#include "CudaMatMul.hpp"
#endif
namespace
{
	void swapLastTwoDims(std::vector<size_t>& A)
	{
		std::swap(A[A.size() - 1], A[A.size() - 2]);
	};

	Tensor matMul(const Tensor& A, const Tensor& B, uint16_t mask)
	{
#ifdef USE_CUDA
		return cudaMatMul(A, B, mask);
#endif
			std::vector<size_t> shapeA = A.shape;
		std::vector<size_t> shapeB = B.shape;

		std::vector<size_t> stridesA = A.strides;
		std::vector<size_t> stridesB = B.strides;

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

		return result;
	}

	Tensor broadcastMultiply(const Tensor& A, const Tensor& B)
	{
		std::vector<size_t> shapeA = A.shape;
		std::vector<size_t> shapeB = B.shape;
		std::vector<size_t> stridesA = A.strides;
		std::vector<size_t> stridesB = B.strides;

		if (shapeA.size() < shapeB.size())
		{
			size_t need = shapeB.size() - shapeA.size();
			shapeA.insert(shapeA.begin(), need, 1);
			stridesA.insert(stridesA.begin(), need, 0);
		}
		else if (shapeB.size() < shapeA.size())
		{
			size_t need = shapeA.size() - shapeB.size();
			shapeB.insert(shapeB.begin(), need, 1);
			stridesB.insert(stridesB.begin(), need, 0);
		}

		std::vector<size_t> resultShape(shapeA.size());
		for (size_t i = 0; i < shapeA.size(); i++)
		{
			resultShape[i] = std::max(shapeA[i], shapeB[i]);
			if (shapeA[i] == 1 && resultShape[i] != 1) { stridesA[i] = 0; }
			if (shapeB[i] == 1 && resultShape[i] != 1) { stridesB[i] = 0; }
		}

		Tensor result(resultShape.size(), resultShape);
		size_t totalElements = result.data->size();
		std::vector<size_t> idx(resultShape.size(), 0);
		const float* dataA = A.data->data();
		const float* dataB = B.data->data();
		float* dataR = result.data->data();

		for (size_t i = 0; i < totalElements; i++)
		{
			size_t offsetA = 0, offsetB = 0;
			for (size_t d = 0; d < idx.size(); d++)
			{
				offsetA += idx[d] * stridesA[d];
				offsetB += idx[d] * stridesB[d];
			}
			dataR[i] = dataA[offsetA] * dataB[offsetB];

			for (size_t d = idx.size(); d-- > 0; )
			{
				if (++idx[d] < resultShape[d]) { break; }
				idx[d] = 0;
			}
		}
		return result;
	}

} // namespace




Tensor AddOperation::forward(const std::vector<Tensor>& inputs) const
{
	return inputs[0] + inputs[1];	
};

std::vector<Tensor> AddOperation::backward(const std::vector<Tensor>& inputs,
	const Tensor&,
	const Tensor& gradOutput) const {

	std::vector<Tensor> result;
	result.reserve(2);
	result.push_back(unbroadcastGrad(gradOutput, inputs[0].shape));
	result.push_back(unbroadcastGrad(gradOutput, inputs[1].shape));
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


Tensor ScaleOperation::forward(const std::vector<Tensor>& inputs) const
{
	return inputs[0] * scaleFactor;
};

std::vector<Tensor> ScaleOperation::backward(const std::vector<Tensor>&,
	const Tensor&,
	const Tensor& gradOutput) const {

	std::vector<Tensor> result;
	result.reserve(1);

	Tensor leftGrad = gradOutput * scaleFactor;
	result.push_back(leftGrad);

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

	bool tA = flags & MatMulFlags::MATMUL_TRANSPOSE_A;
	bool tB = flags & MatMulFlags::MATMUL_TRANSPOSE_B;

	Tensor leftGrad, rightGrad;
	if (!tA && !tB) {
		leftGrad  = matMul(gradOutput,  inputs[1], MatMulFlags::MATMUL_TRANSPOSE_B);
		rightGrad = matMul(inputs[0],   gradOutput, MatMulFlags::MATMUL_TRANSPOSE_A);
	} else if (!tA && tB) {
		leftGrad  = matMul(gradOutput, inputs[1], MatMulFlags::MATMUL_NO_TRANSPOSES);
		rightGrad = matMul(gradOutput, inputs[0], MatMulFlags::MATMUL_TRANSPOSE_A);
	} else if (tA && !tB) {
		leftGrad  = matMul(inputs[1],  gradOutput, MatMulFlags::MATMUL_TRANSPOSE_B);
		rightGrad = matMul(inputs[0],  gradOutput, MatMulFlags::MATMUL_NO_TRANSPOSES);
	} else {
		leftGrad  = matMul(inputs[1], gradOutput, MatMulFlags::MATMUL_TRANSPOSE_A | MatMulFlags::MATMUL_TRANSPOSE_B);
		rightGrad = matMul(gradOutput, inputs[0], MatMulFlags::MATMUL_TRANSPOSE_A | MatMulFlags::MATMUL_TRANSPOSE_B);
	}

	result.push_back(unbroadcastGrad(leftGrad, inputs[0].shape));
	result.push_back(unbroadcastGrad(rightGrad, inputs[1].shape));
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


Tensor SoftmaxOperation::forward(const std::vector<Tensor>& inputs) const
{
	Tensor leftTensor = inputs[0];
	Tensor result(leftTensor.dimensions, leftTensor.shape);
	const float* src = leftTensor.data->data();
	float* dst = result.data->data();

	size_t n = leftTensor.shape.back();
	size_t totalRows = leftTensor.data->size() / n;

	for (size_t b{ 0 }; b < totalRows; b++)
	{
		size_t offset = b * n;

		float maxVal = src[offset];
		for (size_t i{ 1 }; i < n; i++)
			maxVal = std::max(maxVal, src[offset + i]);

		float sum = 0.0f;
		for (size_t i{ 0 }; i < n; i++)
		{
			float e = std::exp(src[offset + i] - maxVal);
			dst[offset + i] = e;
			sum += e;
		}
		for (size_t i{ 0 }; i < n; i++)
			dst[offset + i] /= sum;
	}
	return result;

};

std::vector<Tensor> SoftmaxOperation::backward(const std::vector<Tensor>&,
	const Tensor& output,
	const Tensor& gradOutput) const {
	std::vector<Tensor> result;
	result.reserve(1);
	Tensor weighted = output * gradOutput;
	Tensor dotProduct = weighted.sumAxisKeepDim(weighted.dimensions - 1);
	Tensor gradInput = output * (gradOutput + dotProduct * (-1.0f));
	result.push_back(gradInput);
	return result;
};


Tensor MultiplyOperation::forward(const std::vector<Tensor>& inputs) const
{
	return broadcastMultiply(inputs[0], inputs[1]);
}

std::vector<Tensor> MultiplyOperation::backward(const std::vector<Tensor>& inputs,
	const Tensor&,
	const Tensor& gradOutput) const
{
	Tensor leftGrad  = broadcastMultiply(gradOutput, inputs[1]);
	Tensor rightGrad = broadcastMultiply(gradOutput, inputs[0]);
	return {
		unbroadcastGrad(leftGrad,  inputs[0].shape),
		unbroadcastGrad(rightGrad, inputs[1].shape)
	};
}

Tensor LayerNormOperation::forward(const std::vector<Tensor>& inputs) const
{
	const Tensor& x = inputs[0];
	size_t n = x.shape.back();
	size_t totalRows = x.data->size() / n;
	Tensor result(x.dimensions, x.shape);
	const float* src = x.data->data();
	float* dst = result.data->data();

	for (size_t b = 0; b < totalRows; b++)
	{
		size_t offset = b * n;

		float mean = 0.0f;
		for (size_t i = 0; i < n; i++) { mean += src[offset + i]; }
		mean /= static_cast<float>(n);

		float var = 0.0f;
		for (size_t i = 0; i < n; i++)
		{
			float d = src[offset + i] - mean;
			var += d * d;
		}
		var /= static_cast<float>(n);

		float invStd = 1.0f / std::sqrt(var + eps);
		for (size_t i = 0; i < n; i++)
		{
			dst[offset + i] = (src[offset + i] - mean) * invStd;
		}
	}
	return result;
}

std::vector<Tensor> LayerNormOperation::backward(const std::vector<Tensor>& inputs,
	const Tensor& output,
	const Tensor& gradOutput) const
{
	const Tensor& x = inputs[0];
	size_t n = x.shape.back();
	size_t totalRows = x.data->size() / n;
	Tensor grad(x.dimensions, x.shape);
	const float* src   = x.data->data();
	const float* xhat  = output.data->data();
	const float* dout  = gradOutput.data->data();
	float*       dx    = grad.data->data();

	for (size_t b = 0; b < totalRows; b++)
	{
		size_t offset = b * n;

		float mean = 0.0f;
		for (size_t i = 0; i < n; i++) { mean += src[offset + i]; }
		mean /= static_cast<float>(n);

		float var = 0.0f;
		for (size_t i = 0; i < n; i++)
		{
			float d = src[offset + i] - mean;
			var += d * d;
		}
		var /= static_cast<float>(n);
		float invStd = 1.0f / std::sqrt(var + eps);

		float sumDout = 0.0f, sumDoutXhat = 0.0f;
		for (size_t i = 0; i < n; i++)
		{
			sumDout     += dout[offset + i];
			sumDoutXhat += dout[offset + i] * xhat[offset + i];
		}

		float fn = static_cast<float>(n);
		for (size_t i = 0; i < n; i++)
		{
			dx[offset + i] = invStd * (dout[offset + i]
				- sumDout / fn
				- xhat[offset + i] * sumDoutXhat / fn);
		}
	}
	return { grad };
}

Tensor CausalMaskOperation::forward(const std::vector<Tensor>& inputs) const
{
	Tensor result = inputs[0];
	size_t seq = result.shape.back();
	size_t numMatrices = result.data->size() / (seq * seq);
	float* data = result.data->data();
	for (size_t b = 0; b < numMatrices; b++)
	{
		for (size_t i = 0; i < seq; i++)
		{
			for (size_t j = i + 1; j < seq; j++)
			{
				data[b * seq * seq + i * seq + j] = -1e9f;
			}
		}
	}
	return result;
}

std::vector<Tensor> CausalMaskOperation::backward(const std::vector<Tensor>&,
	const Tensor&,
	const Tensor& gradOutput) const
{
	return { gradOutput };
}

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

Tensor CrossEntropyOperation::forward(const std::vector<Tensor>& inputs) const
{
	const Tensor& logits    = inputs[0];
	const Tensor& targetIds = inputs[1];

	size_t vocabSize = logits.shape.back();
	size_t totalRows = logits.data->size() / vocabSize;

	std::vector<size_t> outShape(logits.shape.begin(), logits.shape.end() - 1);
	Tensor result(outShape.size(), outShape);

	for (size_t i = 0; i < totalRows; i++)
	{
		const float* logitRow = logits.data->data() + i * vocabSize;
		size_t       targetId = static_cast<size_t>((*targetIds.data)[i]);

		float maxVal = logitRow[0];
		for (size_t j = 1; j < vocabSize; j++)
		{
			if (logitRow[j] > maxVal) maxVal = logitRow[j];
		}

		float sumExp = 0.0f;
		for (size_t j = 0; j < vocabSize; j++)
		{
			sumExp += std::exp(logitRow[j] - maxVal);
		}

		(*result.data)[i] = -(logitRow[targetId] - maxVal - std::log(sumExp));
	}

	return result;
}

std::vector<Tensor> CrossEntropyOperation::backward(
	const std::vector<Tensor>& inputs,
	const Tensor&,
	const Tensor& gradOutput) const
{
	const Tensor& logits    = inputs[0];
	const Tensor& targetIds = inputs[1];

	size_t vocabSize = logits.shape.back();
	size_t totalRows = logits.data->size() / vocabSize;

	Tensor gradLogits(logits.dimensions, logits.shape);
	Tensor gradTargets(targetIds.dimensions, targetIds.shape);
	gradTargets.fillValues(0.0f);

	for (size_t i = 0; i < totalRows; i++)
	{
		const float* logitRow = logits.data->data() + i * vocabSize;
		float* gradRow = gradLogits.data->data() + i * vocabSize;
		size_t targetId = static_cast<size_t>((*targetIds.data)[i]);
		float gi = (*gradOutput.data)[i];

		float maxVal = logitRow[0];
		for (size_t j = 1; j < vocabSize; j++)
		{
			if (logitRow[j] > maxVal) maxVal = logitRow[j];
		}

		float sumExp = 0.0f;
		for (size_t j = 0; j < vocabSize; j++)
		{
			sumExp += std::exp(logitRow[j] - maxVal);
		}

		for (size_t j = 0; j < vocabSize; j++)
		{
			gradRow[j] = gi * (std::exp(logitRow[j] - maxVal) / sumExp);
		}
		gradRow[targetId] -= gi;
	}

	return {gradLogits, gradTargets};
}

Tensor EmbeddingOperation::forward(const std::vector<Tensor>& inputs) const
{
	const Tensor& tokenIds = inputs[0];
	const Tensor& weights = inputs[1];

	size_t totalTokens = tokenIds.data->size();
	size_t embedDim = weights.shape[1];

	std::vector<size_t> outShape(tokenIds.shape.begin(), tokenIds.shape.end());
	outShape.push_back(embedDim);
	Tensor result(outShape.size(), outShape);

	for (size_t i = 0; i < totalTokens; i++)
	{
		size_t id = static_cast<size_t>((*tokenIds.data)[i]);
		const float* weightRow = weights.data->data() + id * embedDim;
		float* outRow = result.data->data() + i * embedDim;
		for (size_t e = 0; e < embedDim; e++)
		{
			outRow[e] = weightRow[e];
		}
	}

	return result;
}

std::vector<Tensor> EmbeddingOperation::backward(
	const std::vector<Tensor>& inputs,
	const Tensor&,
	const Tensor& gradOutput) const
{
	const Tensor& tokenIds = inputs[0];
	const Tensor& weights = inputs[1];

	size_t totalTokens = tokenIds.data->size();
	size_t embedDim = weights.shape[1];

	Tensor gradTokens(tokenIds.dimensions, tokenIds.shape);
	gradTokens.fillValues(0.0f);

	Tensor gradWeights(weights.dimensions, weights.shape);
	gradWeights.fillValues(0.0f);

	for (size_t i = 0; i < totalTokens; i++)
	{
		size_t id = static_cast<size_t>((*tokenIds.data)[i]);
		const float* gradRow = gradOutput.data->data() + i * embedDim;
		float* weightRow = gradWeights.data->data() + id * embedDim;
		for (size_t e = 0; e < embedDim; e++)
		{
			weightRow[e] += gradRow[e];
		}
	}

	return {gradTokens, gradWeights};
}
