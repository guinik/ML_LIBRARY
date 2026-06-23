
#pragma once
#include <array>
#include "Tensor.hpp"

struct Parameter{
	Parameter() = default;
	~Parameter() = default;

	Tensor value;
	Tensor grad;

	void clearGradients() {
		grad = Tensor();
	};
};
