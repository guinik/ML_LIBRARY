#include "Tensor.hpp"
#include <stdexcept>
#include <cstdlib>
#include <format>

float& Tensor::operator[](const std::vector<size_t>& inputDims) const {
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

Tensor Tensor::operator+(const Tensor& B) const
{
	if (this->shape.size() != B.shape.size())
		throw std::runtime_error("Matrices have different dimensions");

	for (size_t i{ 0 }; i < shape.size(); i++)
	{
		if (shape[i] != B.shape[i])
			throw std::runtime_error("Shapes are not Compatible");
	}

	Tensor result(dimensions, shape);
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

Tensor Tensor::operator-(const Tensor& B) const
{
	if (this->shape.size() != B.shape.size())
		throw std::runtime_error("Matrices have different dimensions");

	for (size_t i{ 0 }; i < shape.size(); i++)
	{
		if (shape[i] != B.shape[i])
			throw std::runtime_error("Shapes are not Compatible");
	}

	Tensor result(dimensions, shape);
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

Tensor Tensor::operator+(float offset) const
{
	Tensor result = *this;
	float* p = result.data->data();
	size_t n = result.data->size();
	for (size_t i{ 0 }; i < n; i++)
		p[i] += offset;
	return result;
}

Tensor Tensor::operator-(float offset) const
{
	return operator+(-offset);
}

Tensor Tensor::operator/(float offset) const
{
	return operator*(1/offset);
}


Tensor Tensor::operator*(float scaleFactor) const
{
	Tensor result = *this;
	float* p = result.data->data();
	size_t n = result.data->size();
	for (size_t i{ 0 }; i < n; i++)
		p[i] *= scaleFactor;
	return result;
}

void Tensor::transpose() {

	std::swap(strides[strides.size() - 2], strides[strides.size() - 1]);
	std::swap(shape[strides.size() - 2], shape[strides.size() - 1]);

}


void Tensor::fillValues(float value)
{
	std::fill(data->begin(), data->end(), value);
}

Tensor Tensor::square() const
{
	Tensor result(dimensions, shape);
	const float* src = data->data();
	float* dst = result.data->data();
	size_t n = data->size();
	for (size_t i = 0; i < n; i++)
		dst[i] = src[i] * src[i];
	return result;
}

void Tensor::randomize(float scale)
{
	for (auto& w : *data)
		w = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
}