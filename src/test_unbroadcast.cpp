#include "Tensor.hpp"
#include "CudaMatMul.hpp"
#include "CudaOps.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// checks the parallel-reduction rewrite of reduceLeadingKernel against a plain cpu sum

static Tensor makeRandomTensor(const std::vector<size_t>& shape)
{
	Tensor t(shape.size(), shape);
	for (size_t i = 0; i < t.data->size(); i++)
	{
		(*t.data)[i] = static_cast<float>((rand() % 2000) - 1000) / 100.0f;
	}
	return t;
}

struct Case
{
	const char* name;
	size_t leadTotal;
	size_t lastDim;
};

int main()
{
	srand(42);
	cudaMatMulInit();

	std::vector<Case> cases = {
		{ "model scale, dense bias", 1024, 256 },
		{ "model scale, output layer bias", 1024, 4096 },
		{ "small", 3, 5 },
		{ "lastDim not a multiple of 32", 17, 100 },
		{ "leadTotal smaller than reduction width", 1024, 3 },
		{ "leadTotal 1", 1, 64 },
	};

	int failures = 0;

	for (auto& c : cases)
	{
		Tensor grad = makeRandomTensor({ c.leadTotal, c.lastDim });
		Tensor result = cudaUnbroadcast(grad, { c.lastDim });
		result.toCPU();

		std::vector<float> expected(c.lastDim, 0.0f);
		for (size_t i = 0; i < c.leadTotal; i++)
		{
			for (size_t d = 0; d < c.lastDim; d++)
			{
				expected[d] += (*grad.data)[i * c.lastDim + d];
			}
		}

		bool shapeOk = result.data->size() == expected.size();
		float maxDiff = 0.0f;
		if (shapeOk)
		{
			for (size_t d = 0; d < c.lastDim; d++)
			{
				maxDiff = std::max(maxDiff, std::fabs((*result.data)[d] - expected[d]));
			}
		}

		bool pass = shapeOk && maxDiff < 1e-1f;
		std::cout << (pass ? "[PASS] " : "[FAIL] ") << c.name
			<< " leadTotal=" << c.leadTotal << " lastDim=" << c.lastDim
			<< " -- shapeOk=" << shapeOk << " maxDiff=" << maxDiff << "\n";
		if (!pass)
		{
			failures++;
		}
	}

	if (failures == 0)
	{
		std::cout << "\nAll reduceLeadingKernel correctness checks passed.\n";
	}
	else
	{
		std::cout << "\n" << failures << " reduceLeadingKernel correctness check(s) FAILED.\n";
	}

	cudaMatMulShutdown();
	return failures == 0 ? 0 : 1;
}
