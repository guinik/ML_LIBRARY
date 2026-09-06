#include "Tensor.hpp"
#include "Node.hpp"
#include "CudaMatMul.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// checks MatMulOperation::backward's fused weight gradient against an independent reference

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
	srand(777);
	cudaMatMulInit();

	const size_t BATCH = 3;
	const size_t SEQ = 5;
	const size_t IN_DIM = 4;
	const size_t OUT_DIM = 6;

	Tensor x = makeRandomTensor({ BATCH, SEQ, IN_DIM });
	Tensor weight = makeRandomTensor({ OUT_DIM, IN_DIM });
	Tensor gradOutput = makeRandomTensor({ BATCH, SEQ, OUT_DIM });

	MatMulOperation op(MatMulFlags::MATMUL_TRANSPOSE_B);
	std::vector<const Tensor*> inputs = { &x, &weight };
	Tensor dummyOutput;
	std::vector<Tensor> grads = op.backward(inputs, dummyOutput, gradOutput);

	Tensor leftGrad = grads[0];
	Tensor rightGrad = grads[1];
#ifdef USE_CUDA
	leftGrad.toCPU();
	rightGrad.toCPU();
#endif

	// out[b][s][n] = sum_k x[b][s][k] * weight[n][k]
	std::vector<float> expectedLeftGrad(BATCH * SEQ * IN_DIM, 0.0f);
	std::vector<float> expectedRightGrad(OUT_DIM * IN_DIM, 0.0f);

	for (size_t b = 0; b < BATCH; b++)
	{
		for (size_t s = 0; s < SEQ; s++)
		{
			for (size_t k = 0; k < IN_DIM; k++)
			{
				float sum = 0.0f;
				for (size_t n = 0; n < OUT_DIM; n++)
				{
					float g = (*gradOutput.data)[(b * SEQ + s) * OUT_DIM + n];
					float w = (*weight.data)[n * IN_DIM + k];
					sum += g * w;
				}
				expectedLeftGrad[(b * SEQ + s) * IN_DIM + k] = sum;
			}
		}
	}

	for (size_t n = 0; n < OUT_DIM; n++)
	{
		for (size_t k = 0; k < IN_DIM; k++)
		{
			float sum = 0.0f;
			for (size_t b = 0; b < BATCH; b++)
			{
				for (size_t s = 0; s < SEQ; s++)
				{
					float g = (*gradOutput.data)[(b * SEQ + s) * OUT_DIM + n];
					float xv = (*x.data)[(b * SEQ + s) * IN_DIM + k];
					sum += g * xv;
				}
			}
			expectedRightGrad[n * IN_DIM + k] = sum;
		}
	}

	bool leftShapeOk = leftGrad.data->size() == expectedLeftGrad.size();
	bool rightShapeOk = rightGrad.data->size() == expectedRightGrad.size();

	float maxDiffLeft = 0.0f;
	if (leftShapeOk)
	{
		for (size_t i = 0; i < expectedLeftGrad.size(); i++)
		{
			maxDiffLeft = std::max(maxDiffLeft, std::fabs((*leftGrad.data)[i] - expectedLeftGrad[i]));
		}
	}

	float maxDiffRight = 0.0f;
	if (rightShapeOk)
	{
		for (size_t i = 0; i < expectedRightGrad.size(); i++)
		{
			maxDiffRight = std::max(maxDiffRight, std::fabs((*rightGrad.data)[i] - expectedRightGrad[i]));
		}
	}

	bool passLeft = leftShapeOk && maxDiffLeft < 1e-2f;
	bool passRight = rightShapeOk && maxDiffRight < 1e-2f;

	std::cout << (passLeft ? "[PASS] " : "[FAIL] ") << "leftGrad (dL/dx)"
		<< " -- shapeOk=" << leftShapeOk << " maxDiff=" << maxDiffLeft << "\n";
	std::cout << (passRight ? "[PASS] " : "[FAIL] ") << "rightGrad (dL/dW, fused batch-fold path)"
		<< " -- shapeOk=" << rightShapeOk << " maxDiff=" << maxDiffRight << "\n";

	bool allPass = passLeft && passRight;
	if (allPass)
	{
		std::cout << "\nAll weight-gradient correctness checks passed.\n";
	}
	else
	{
		std::cout << "\nWeight-gradient correctness check FAILED.\n";
	}

	cudaMatMulShutdown();
	return allPass ? 0 : 1;
}
