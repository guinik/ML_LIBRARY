#include "Matrix.hpp"
#include <stdexcept>
#include <cstdlib>
#include <format>

float& Matrix::operator[](const std::vector<size_t>& inputDims) const {
	if (inputDims.size() != shape.size())
	{
		throw std::runtime_error("Invalid shape input");
	}

	size_t projectedDim{ 0 };
	for (size_t i{ shape.size() }; i-- > 0;)
	{
		if (inputDims[i] >= shape[i])
		{
			throw std::runtime_error(std::format("Out of bounds index at dim : {}", i));
		}
		projectedDim += inputDims[i] * strides[i];
	}

	return (*data)[projectedDim];
};

Matrix Matrix::operator+(const Matrix& B) const
{
	if (this->shape.size() != B.shape.size())
		throw std::runtime_error("Matrices have different dimensions");

	for (size_t i{ 0 }; i < shape.size(); i++)
	{
		if (shape[i] != B.shape[i])
			throw std::runtime_error("Shapes are not Compatible");
	}

	Matrix result(dimensions, shape);
	std::vector<size_t> odometerIdx(shape.size(), 0);
	size_t totalElements = data->size();
	for (size_t i{ 0 }; i < totalElements; i++)
	{
		result[odometerIdx] = (*this)[odometerIdx] + B[odometerIdx];

		for (size_t d{ odometerIdx.size() }; d-- > 0;)
		{
			if (++odometerIdx[d] < shape[d])
				break;
			odometerIdx[d] = 0;
		}
	}

	return result;
}

Matrix Matrix::operator-(const Matrix& B) const
{
	if (this->shape.size() != B.shape.size())
		throw std::runtime_error("Matrices have different dimensions");

	for (size_t i{ 0 }; i < shape.size(); i++)
	{
		if (shape[i] != B.shape[i])
			throw std::runtime_error("Shapes are not Compatible");
	}

	Matrix result(dimensions, shape);
	std::vector<size_t> odometerIdx(shape.size(), 0);
	size_t totalElements = data->size();
	for (size_t i{ 0 }; i < totalElements; i++)
	{
		result[odometerIdx] = (*this)[odometerIdx] - B[odometerIdx];

		for (size_t d{ odometerIdx.size() }; d-- > 0;)
		{
			if (++odometerIdx[d] < shape[d])
				break;
			odometerIdx[d] = 0;
		}
	}

	return result;
}

Matrix Matrix::operator+(float offset) const
{
	Matrix result = *this;
	float* p = result.data->data();
	size_t n = result.data->size();
	for (size_t i{ 0 }; i < n; i++)
		p[i] += offset;
	return result;
}

Matrix Matrix::operator-(float offset) const
{
	return operator+(-offset);
}

Matrix Matrix::operator/(float offset) const
{
	return operator*(1/offset);
}


Matrix Matrix::operator*(float scaleFactor) const
{
	Matrix result = *this;
	float* p = result.data->data();
	size_t n = result.data->size();
	for (size_t i{ 0 }; i < n; i++)
		p[i] *= scaleFactor;
	return result;
}



Matrix Matrix::operator*(const Matrix& B) const
{
	std::vector<size_t> shapeA = this->shape;
	std::vector<size_t> shapeB = B.shape;

	std::vector<size_t> stridesA = this->strides;
	std::vector<size_t> stridesB = B.strides;

	bool promotedA = false, promotedB = false;

	if (shapeA.size() == 1) 
	{ 
		shapeA.insert(shapeA.begin(), 1); 
		promotedA = true; 
		stridesA.insert(stridesA.begin(), stridesA[0]); 
	}
	if (shapeB.size() == 1) 
	{ 
		shapeB.push_back(1);              
		promotedB = true;
		stridesB.push_back(1);
	}

	if (shapeA.size() < 2 || shapeB.size() < 2)
		throw std::runtime_error("Operands must have at least 1 dimension");

	size_t K = shapeA.back();
	if (K != shapeB[shapeB.size() - 2])
		throw std::runtime_error("Inner dimensions don't match for multiplication");

	size_t batchDimsA = shapeA.size() - 2;
	size_t batchDimsB = shapeB.size() - 2;
	if (batchDimsA != batchDimsB)
		throw std::runtime_error("Batch dimension count mismatch");

	std::vector<size_t> batchShape;
	for (size_t i = 0; i < batchDimsA; i++)
	{
		if (shapeA[i] != shapeB[i])
			throw std::runtime_error("Batch dimension sizes don't match");
		batchShape.push_back(shapeA[i]);
	}

	size_t M = shapeA[shapeA.size() - 2];
	size_t N = shapeB.back();

	size_t batchCount = 1;
	for (auto d : batchShape) batchCount *= d;

	std::vector<size_t> resultShape = batchShape;
	resultShape.push_back(M);
	resultShape.push_back(N);
	Matrix result(resultShape.size(), resultShape);

	const size_t strideA_row = stridesA[stridesA.size() - 2];
	const size_t strideA_col = stridesA[stridesA.size() - 1];
	const size_t strideB_row = stridesB[stridesB.size() - 2];
	const size_t strideB_col = stridesB[stridesB.size() - 1];
	const size_t strideR_row = result.strides[result.strides.size() - 2];
	const size_t strideR_col = result.strides[result.strides.size() - 1];

	const float* dataA = this->data->data();
	const float* dataB = B.data->data();
	float* dataR = result.data->data();

	for (size_t b{ 0 }; b < batchCount; b++)
	{
		size_t baseA = b * M * K;
		size_t baseB = b * K * N;
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

void Matrix::transpose() {

	std::swap(strides[strides.size() - 2], strides[strides.size() - 1]);
	std::swap(shape[strides.size() - 2], shape[strides.size() - 1]);

}


void Matrix::fillValues(float value)
{
	std::fill(data->begin(), data->end(), value);
}

Matrix Matrix::square() const
{
	Matrix result(dimensions, shape);
	const float* src = data->data();
	float* dst = result.data->data();
	size_t n = data->size();
	for (size_t i = 0; i < n; i++)
		dst[i] = src[i] * src[i];
	return result;
}

void Matrix::randomize(float scale)
{
	for (auto& w : *data)
		w = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
}