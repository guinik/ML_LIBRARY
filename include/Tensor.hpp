#pragma once
#include <vector>
#include <memory>
#include <functional>


struct AbstractTensor
{
	virtual ~AbstractTensor() = default;
	std::vector<size_t> shape;
	std::vector<size_t> strides;
	std::shared_ptr<std::vector<float>> data;
	size_t dimensions{};
};


struct Tensor : AbstractTensor
{
	Tensor() : dimensions(0) {};
	Tensor(size_t  dimensionsInput, std::vector<size_t> shapes) : shape(shapes), dimensions(dimensionsInput)
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
		: shape(other.shape), strides(other.strides),
		  data(other.data ? std::make_shared<std::vector<float>>(*other.data) : nullptr),
		  dimensions(other.dimensions)
#ifdef USE_CUDA
		, d_data(other.d_data)
#endif
	{}

	Tensor& operator=(const Tensor& other)
	{
		if (this != &other)
		{
			shape      = other.shape;
			strides    = other.strides;
			data       = other.data ? std::make_shared<std::vector<float>>(*other.data) : nullptr;
			dimensions = other.dimensions;
#ifdef USE_CUDA
			d_data     = other.d_data;
#endif
		}
		return *this;
	}

	~Tensor() = default;
	float& operator[](const std::vector<size_t>& inputDims) const;
	Tensor operator+(const Tensor& B) const;
	Tensor operator-(const Tensor& B) const;
	Tensor operator*(const Tensor& B) const;
	Tensor operator+(float scaleFactor) const;
	Tensor operator*(float scaleFactor) const;
	Tensor operator-(float scaleFactor) const;
	Tensor operator/(float scaleFactor) const;

	std::vector<size_t> shape;
	std::vector<size_t> strides;
	std::shared_ptr<std::vector<float>> data;
	size_t dimensions;

#ifdef USE_CUDA
	mutable std::shared_ptr<float> d_data;
	void toGPU() const;
	void toCPU() const;
	void invalidateGPU() const;
	bool onGPU() const { return d_data != nullptr; }
#endif

	size_t nelems() const
	{
		size_t n = 1;
		for (auto d : shape) { n *= d; }
		return n;
	}

	void transpose();
	void fillValues(float value);
	Tensor square() const;
	void randomize(float scale);
	Tensor sumAxis(size_t dim) const;
	Tensor sumAxisKeepDim(size_t axis) const;
};


Tensor unbroadcastGrad(const Tensor& grad, const std::vector<size_t>& targetShape);