#pragma once
#include <vector>
#include <memory>


struct Matrix
{
	Matrix() : dimensions(0) {};
	Matrix(size_t  dimensionsInput, std::vector<size_t> shapes) : dimensions(dimensionsInput), shape(shapes)
	{
		size_t totalParameters{ 1 };
		strides = shapes;
		for(size_t i{shapes.size()}; i-- > 0;)
		{
			strides[i] = totalParameters;
			totalParameters *= shapes[i];

		}

		data = std::make_shared<std::vector<float>>(totalParameters);
	};

	Matrix(const Matrix& other)
		: dimensions(other.dimensions), shape(other.shape), strides(other.strides),
		  data(other.data ? std::make_shared<std::vector<float>>(*other.data) : nullptr) {}

	Matrix& operator=(const Matrix& other)
	{
		if (this != &other)
		{
			dimensions = other.dimensions;
			shape      = other.shape;
			strides    = other.strides;
			data       = other.data ? std::make_shared<std::vector<float>>(*other.data) : nullptr;
		}
		return *this;
	}

	~Matrix() = default;

	float& operator[](const std::vector<size_t>& inputDims) const;
	Matrix operator+(const Matrix& B) const;
	Matrix operator-(const Matrix& B) const;
	Matrix operator*(const Matrix& B) const;

	Matrix operator+(float scaleFactor) const;
	Matrix operator*(float scaleFactor) const;
	Matrix operator-(float scaleFactor) const;
	Matrix operator/(float scaleFactor) const;

	std::vector<size_t> shape;
	std::vector<size_t> strides;
	std::shared_ptr<std::vector<float>> data;
	size_t dimensions;

	void transpose();
	void fillValues(float value);
	Matrix square() const;
	void randomize(float scale);
};
