#pragma once
#include <vector>
#include <memory>


struct AbstractTensor
{
	virtual ~AbstractTensor() = default;
	std::vector<size_t> shape;
	std::vector<size_t> strides;
	std::shared_ptr<std::vector<float>> data;
	size_t dimensions;
};


struct Tensor : AbstractTensor
{
	Tensor() : dimensions(0) {};
	Tensor(size_t  dimensionsInput, std::vector<size_t> shapes) : dimensions(dimensionsInput), shape(shapes)
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

	Tensor(const Tensor& other)
		: dimensions(other.dimensions), shape(other.shape), strides(other.strides),
		  data(other.data ? std::make_shared<std::vector<float>>(*other.data) : nullptr) {}

	Tensor& operator=(const Tensor& other)
	{
		if (this != &other)
		{
			shape      = other.shape;
			strides    = other.strides;
			data       = other.data ? std::make_shared<std::vector<float>>(*other.data) : nullptr;
			dimensions = other.dimensions;
		}
		return *this;
	}

	~Tensor() = default;
	float& operator[](const std::vector<size_t>& inputDims) const;
	Tensor operator+(const Tensor& B) const;
	Tensor operator-(const Tensor& B) const;
	Tensor operator+(float scaleFactor) const;
	Tensor operator*(float scaleFactor) const;
	Tensor operator-(float scaleFactor) const;
	Tensor operator/(float scaleFactor) const;

	std::vector<size_t> shape;
	std::vector<size_t> strides;
	std::shared_ptr<std::vector<float>> data;
	size_t dimensions;

	void transpose();
	void fillValues(float value);
	Tensor square() const;
	void randomize(float scale);
	Tensor sumAxis(size_t dim) const;
	Tensor sumAxisKeepDim(size_t axis) const;
};


Tensor unbroadcastGrad(const Tensor& grad, const std::vector<size_t>& targetShape);