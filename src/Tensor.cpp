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
	std::vector<size_t> shapeA = this->shape;
	std::vector<size_t> shapeB = B.shape;
	std::vector<size_t> stridesA = this->strides;
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
		if (shapeA[i] != shapeB[i] && shapeA[i] != 1 && shapeB[i] != 1)
			throw std::runtime_error("Shapes are not compatible for broadcasting");
		resultShape[i] = std::max(shapeA[i], shapeB[i]);

		if (shapeA[i] == 1 && resultShape[i] != 1) stridesA[i] = 0;
		if (shapeB[i] == 1 && resultShape[i] != 1) stridesB[i] = 0;
	}


	Tensor result(resultShape.size(), resultShape);
	std::vector<size_t> odometerIdx(resultShape.size(), 0);
	size_t totalElements = 1;
	for (auto d : resultShape) totalElements *= d;

	float* dataR = result.data->data();
	const float* dataA = this->data->data();
	const float* dataB = B.data->data();

	for (size_t i{ 0 }; i < totalElements; i++)
	{
		size_t offsetA = 0, offsetB = 0;
		for (size_t d{ 0 }; d < odometerIdx.size(); d++)
		{
			offsetA += odometerIdx[d] * stridesA[d];
			offsetB += odometerIdx[d] * stridesB[d];
		}
		dataR[i] = dataA[offsetA] + dataB[offsetB];

		for (size_t d{ odometerIdx.size() }; d-- > 0; )
		{
			if (++odometerIdx[d] < resultShape[d])
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

Tensor Tensor::operator*(const Tensor& other) const
{
	Tensor result(this->dimensions, this->shape);
	const std::vector<float>& thisData = *(this->data);
	const std::vector<float>& otherData = *(other.data);
	std::vector<float>& dstData = *(result.data);
	size_t n = this->data->size();
	for (size_t i{0}; i < n; i++)
	{
		dstData[i] = thisData[i] * otherData[i];
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


Tensor Tensor::sumAxisKeepDim(size_t axis) const
{
	if (axis >= shape.size())
		throw std::runtime_error("Axis out of bounds");

	std::vector<size_t> resultShape = shape;
	resultShape[axis] = 1;

	Tensor result(resultShape.size(), resultShape);
	float* dataR = result.data->data();
	const float* dataA = this->data->data();

	size_t totalElements = 1;
	for (auto d : shape) totalElements *= d;

	std::vector<size_t> idx(shape.size(), 0);
	for (size_t i{ 0 }; i < totalElements; i++)
	{
		size_t offsetA = 0;
		for (size_t d{ 0 }; d < idx.size(); d++)
			offsetA += idx[d] * strides[d];

		size_t offsetR = 0;
		for (size_t d{ 0 }; d < idx.size(); d++)
		{
			size_t coord = (d == axis) ? 0 : idx[d];
			offsetR += coord * result.strides[d];
		}

		dataR[offsetR] += dataA[offsetA];

		for (size_t d{ idx.size() }; d-- > 0; )
		{
			if (++idx[d] < shape[d]) break;
			idx[d] = 0;
		}
	}
	return result;
}

Tensor Tensor::sumAxis(size_t axis) const
{
	Tensor summed = sumAxisKeepDim(axis);

	std::vector<size_t> squeezedShape = summed.shape;
	squeezedShape.erase(squeezedShape.begin() + axis);

	Tensor result(squeezedShape.size(), squeezedShape);
	*result.data = *summed.data;
	return result;
}