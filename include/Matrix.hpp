#pragma once
#include <vector>



struct Matrix
{
	Matrix() : dimensions(0) {};
	Matrix(size_t  dimensionsInput, std::vector<size_t> shapes) : dimensions(dimensionsInput), shape(shapes)
	{
		size_t totalParameters{ 1 };
		for (auto shapeDim : shape)
		{
			totalParameters *= shapeDim;
		}
		data.resize(totalParameters);
	};
	~Matrix() = default;

	float& operator[](std::vector<size_t> inputDims);
	Matrix operator+(const Matrix& B) const;
	Matrix operator-(const Matrix& B) const;
	Matrix operator*(const Matrix& B) const;

	Matrix operator+(float scaleFactor) const;
	Matrix operator*(float scaleFactor) const;
	Matrix operator-(float scaleFactor) const;
	Matrix operator/(float scaleFactor) const;

	std::vector<size_t> shape;
	std::vector<float> data;
	size_t dimensions;

	Matrix transpose() const;
	void fillValues(float value);
	Matrix square() const;
	void randomize(float scale);
};
