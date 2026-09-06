#include "Tensor.hpp"
#include "Node.hpp"
#include "CudaMatMul.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// checks the fused DenseOperation (matmul + bias epilogue) forward and backward against independent references

static Tensor makeRandomTensor(const std::vector<size_t>& shape)
{
	Tensor t(shape.size(), shape);
	for (size_t i = 0; i < t.data->size(); i++)
	{
		(*t.data)[i] = static_cast<float>((rand() % 2000) - 1000) / 100.0f;
	}
	return t;
}

int main()
{
	srand(2024);
	cudaMatMulInit();

	const size_t BATCH = 3;
	const size_t SEQ = 5;
	const size_t IN_DIM = 4;
	const size_t OUT_DIM = 6;

	Tensor x = makeRandomTensor({ BATCH, SEQ, IN_DIM });
	Tensor weight = makeRandomTensor({ OUT_DIM, IN_DIM });
	Tensor bias = makeRandomTensor({ OUT_DIM });

	DenseOperation op;
	std::vector<const Tensor*> inputs = { &x, &weight, &bias };
	Tensor dummyOutput;

	Tensor forwardResult = op.forward(inputs);
#ifdef USE_CUDA
	forwardResult.toCPU();
#endif

	std::vector<float> expectedForward(BATCH * SEQ * OUT_DIM, 0.0f);
	for (size_t b = 0; b < BATCH; b++)
	{
		for (size_t s = 0; s < SEQ; s++)
		{
			for (size_t n = 0; n < OUT_DIM; n++)
			{
				float sum = (*bias.data)[n];
				for (size_t k = 0; k < IN_DIM; k++)
				{
					sum += (*x.data)[(b * SEQ + s) * IN_DIM + k] * (*weight.data)[n * IN_DIM + k];
				}
				expectedForward[(b * SEQ + s) * OUT_DIM + n] = sum;
			}
		}
	}

	bool fwdShapeOk = forwardResult.data->size() == expectedForward.size();
	float fwdMaxDiff = 0.0f;
	if (fwdShapeOk)
	{
		for (size_t i = 0; i < expectedForward.size(); i++)
		{
			fwdMaxDiff = std::max(fwdMaxDiff, std::fabs((*forwardResult.data)[i] - expectedForward[i]));
		}
	}
	bool fwdPass = fwdShapeOk && fwdMaxDiff < 1e-2f;
	std::cout << (fwdPass ? "[PASS] " : "[FAIL] ") << "forward (matmul + bias epilogue)"
		<< " -- shapeOk=" << fwdShapeOk << " maxDiff=" << fwdMaxDiff << "\n";

	Tensor gradOutput = makeRandomTensor({ BATCH, SEQ, OUT_DIM });
	std::vector<Tensor> grads = op.backward(inputs, dummyOutput, gradOutput);
	Tensor dx = grads[0];
	Tensor dW = grads[1];
	Tensor dBias = grads[2];
#ifdef USE_CUDA
	dx.toCPU();
	dW.toCPU();
	dBias.toCPU();
#endif

	std::vector<float> expectedDx(BATCH * SEQ * IN_DIM, 0.0f);
	std::vector<float> expectedDW(OUT_DIM * IN_DIM, 0.0f);
	std::vector<float> expectedDBias(OUT_DIM, 0.0f);

	for (size_t b = 0; b < BATCH; b++)
	{
		for (size_t s = 0; s < SEQ; s++)
		{
			for (size_t k = 0; k < IN_DIM; k++)
			{
				float sum = 0.0f;
				for (size_t n = 0; n < OUT_DIM; n++)
				{
					sum += (*gradOutput.data)[(b * SEQ + s) * OUT_DIM + n] * (*weight.data)[n * IN_DIM + k];
				}
				expectedDx[(b * SEQ + s) * IN_DIM + k] = sum;
			}
		}
	}

	for (size_t n = 0; n < OUT_DIM; n++)
	{
		float biasSum = 0.0f;
		for (size_t k = 0; k < IN_DIM; k++)
		{
			float sum = 0.0f;
			for (size_t b = 0; b < BATCH; b++)
			{
				for (size_t s = 0; s < SEQ; s++)
				{
					sum += (*gradOutput.data)[(b * SEQ + s) * OUT_DIM + n] * (*x.data)[(b * SEQ + s) * IN_DIM + k];
				}
			}
			expectedDW[n * IN_DIM + k] = sum;
		}
		for (size_t b = 0; b < BATCH; b++)
		{
			for (size_t s = 0; s < SEQ; s++)
			{
				biasSum += (*gradOutput.data)[(b * SEQ + s) * OUT_DIM + n];
			}
		}
		expectedDBias[n] = biasSum;
	}

	auto checkVec = [](const char* name, const std::vector<float>& got, const std::vector<float>& want, bool gotShapeOk)
	{
		float maxDiff = 0.0f;
		if (gotShapeOk)
		{
			for (size_t i = 0; i < want.size(); i++)
			{
				maxDiff = std::max(maxDiff, std::fabs(got[i] - want[i]));
			}
		}
		bool pass = gotShapeOk && maxDiff < 1e-1f;
		std::cout << (pass ? "[PASS] " : "[FAIL] ") << name
			<< " -- shapeOk=" << gotShapeOk << " maxDiff=" << maxDiff << "\n";
		return pass;
	};

	bool dxPass = checkVec("backward dx", *dx.data, expectedDx, dx.data->size() == expectedDx.size());
	bool dWPass = checkVec("backward dW", *dW.data, expectedDW, dW.data->size() == expectedDW.size());
	bool dBiasPass = checkVec("backward dBias", *dBias.data, expectedDBias, dBias.data->size() == expectedDBias.size());

	bool allPass = fwdPass && dxPass && dWPass && dBiasPass;
	if (allPass)
	{
		std::cout << "\nAll DenseOperation correctness checks passed.\n";
	}
	else
	{
		std::cout << "\nDenseOperation correctness check FAILED.\n";
	}

	cudaMatMulShutdown();
	return allPass ? 0 : 1;
}
