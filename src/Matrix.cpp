#include "Matrix.hpp"
#include <stdexcept>
#include <cstdlib>

float& Matrix::operator[](std::vector<size_t> inputDims) {
	if (inputDims.size() != shape.size())
	{
		throw std::runtime_error("Invalid shape input");
	}

	size_t projectedDim{ 0 };
	size_t stride{ 1 };
	std::vector<size_t> dims(inputDims);
	for (size_t i{ shape.size() }; i-- > 0;)
	{
		projectedDim += dims[i] * stride;
		stride *= dims[i];
	}

	return data[projectedDim];
};

Matrix Matrix::operator+(const Matrix& B) const
{
	Matrix result = *this;
	if (this->shape.size() != B.shape.size())
	{
		throw std::runtime_error("Matrices have different dimensions");
	}
	for (size_t i{ 0 }; i < shape.size(); i++)
	{
		if (this->shape[i] != B.shape[i])
		{
			throw std::runtime_error("Shapes are not Compatible");
		}
	}
	for (size_t i{ 0 }; i < data.size(); i++)
	{
		result.data[i] += B.data[i];
	}
	return result;
}

Matrix Matrix::operator-(const Matrix& B) const
{
	return *this + (B * -1.0f);
}

Matrix Matrix::operator+(float offset) const
{
	Matrix result = *this;
	for (size_t i{ 0 }; i < data.size(); i++)
	{
		result.data[i] += offset;
	}
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
	for (size_t i{ 0 }; i < data.size(); i++)
	{
		result.data[i] *= scaleFactor;
	}
	return result;
}



Matrix Matrix::operator*(const Matrix& B) const
{
	std::vector<size_t> shapeA = this->shape;
	std::vector<size_t> shapeB = B.shape;
	bool promotedA = false, promotedB = false;

	if (shapeA.size() == 1) { shapeA.insert(shapeA.begin(), 1); promotedA = true; }
	if (shapeB.size() == 1) { shapeB.push_back(1);              promotedB = true; }

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

	for (size_t b{ 0 }; b < batchCount; b++)
	{
		size_t baseA = b * M * K;
		size_t baseB = b * K * N;
		size_t baseResult = b * M * N;
		for (size_t m{ 0 }; m < M; m++)
		{
			for (size_t n{ 0 }; n < N; n++)
			{
				float sum{};
				for (size_t k{ 0 }; k < K; k++)
				{
					sum += this->data[baseA + m * K + k] * B.data[baseB + k * N + n];
				}
				result.data[baseResult + m * N + n] = sum;
			}
		}
	}

	if (promotedA)
	{
		result.shape.erase(result.shape.end() - 2);
	};
	if (promotedB)
	{
		result.shape.erase(result.shape.end() - 1);
	};

	return result;

}


Matrix Matrix::transpose() const
{
	if (shape.size() < 2)
		throw std::runtime_error("Need at least 2 dimensions to transpose");

	std::vector<size_t> newShape = shape;
	std::swap(newShape[newShape.size() - 2], newShape[newShape.size() - 1]);

	Matrix result(newShape.size(), newShape);

	size_t M = shape[shape.size() - 2];
	size_t N = shape[shape.size() - 1];
	size_t batchCount{ 1 }; 
	for (size_t i = 0; i < shape.size() - 2; i++) {
		batchCount *= shape[i];
	}
	for (size_t b = 0; b < batchCount; b++)
	{
		size_t baseSrc = b * M * N;
		size_t baseDst = b * N * M;
		for (size_t m = 0; m < M; m++)
			for (size_t n = 0; n < N; n++)
				result.data[baseDst + n * M + m] = data[baseSrc + m * N + n];
	}
	return result;
}


void Matrix::fillValues(float value)
{
	for (size_t i{ 0 }; i < data.size(); i++)
	{
		this->data[i] = value;
	}
}

Matrix Matrix::square() const
{
	Matrix result = *this;
	for (size_t i = 0; i < result.data.size(); i++)
		result.data[i] = result.data[i] * result.data[i];
	return result;
}

void Matrix::randomize(float scale)
{
	for (auto& w : data)
		w = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
}