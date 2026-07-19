#include "Tensor.hpp"
#include "Node.hpp"
#include "CudaMatMul.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// checks the fused LayerNormAffineOperation forward and backward against independent references

static Tensor makeRandomTensor(const std::vector<size_t>& shape)
{
	Tensor t(shape.size(), shape);
	for (size_t i = 0; i < t.data->size(); i++)
	{
		(*t.data)[i] = static_cast<float>((rand() % 2000) - 1000) / 100.0f;
	}
	return t;
}

static bool checkVec(const char* name, const std::vector<float>& got, const std::vector<float>& want, bool gotShapeOk)
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
}

int main()
{
	srand(9001);
	cudaMatMulInit();

	const size_t ROWS = 7;
	const size_t DIM = 8;
	const float EPS = 1e-5f;

	Tensor x = makeRandomTensor({ ROWS, DIM });
	Tensor gamma = makeRandomTensor({ DIM });
	Tensor beta = makeRandomTensor({ DIM });

	LayerNormAffineOperation op(EPS);
	std::vector<const Tensor*> inputs = { &x, &gamma, &beta };
	Tensor dummyOutput;

	Tensor forwardResult = op.forward(inputs);
#ifdef USE_CUDA
	forwardResult.toCPU();
#endif

	std::vector<float> expectedForward(ROWS * DIM, 0.0f);
	std::vector<float> means(ROWS), invStds(ROWS);
	for (size_t r = 0; r < ROWS; r++)
	{
		float mean = 0.0f;
		for (size_t i = 0; i < DIM; i++)
		{
			mean += (*x.data)[r * DIM + i];
		}
		mean /= (float)DIM;
		float var = 0.0f;
		for (size_t i = 0; i < DIM; i++)
		{
			float d = (*x.data)[r * DIM + i] - mean;
			var += d * d;
		}
		var /= (float)DIM;
		float invStd = 1.0f / std::sqrt(var + EPS);
		means[r] = mean;
		invStds[r] = invStd;
		for (size_t i = 0; i < DIM; i++)
		{
			float xhat = ((*x.data)[r * DIM + i] - mean) * invStd;
			expectedForward[r * DIM + i] = xhat * (*gamma.data)[i] + (*beta.data)[i];
		}
	}

	bool fwdShapeOk = forwardResult.data->size() == expectedForward.size();
	bool fwdPass = checkVec("forward (normalize * gamma + beta)", *forwardResult.data, expectedForward, fwdShapeOk);

	Tensor gradOutput = makeRandomTensor({ ROWS, DIM });
	std::vector<Tensor> grads = op.backward(inputs, dummyOutput, gradOutput);
	Tensor dx = grads[0];
	Tensor dGamma = grads[1];
	Tensor dBeta = grads[2];
#ifdef USE_CUDA
	dx.toCPU();
	dGamma.toCPU();
	dBeta.toCPU();
#endif

	std::vector<float> expectedDx(ROWS * DIM, 0.0f);
	std::vector<float> expectedDGamma(DIM, 0.0f);
	std::vector<float> expectedDBeta(DIM, 0.0f);

	for (size_t r = 0; r < ROWS; r++)
	{
		float mean = means[r];
		float invStd = invStds[r];
		float sumDxhat = 0.0f, sumDxhatXhat = 0.0f;
		std::vector<float> xhatRow(DIM), dxhatRow(DIM);
		for (size_t i = 0; i < DIM; i++)
		{
			float xhat = ((*x.data)[r * DIM + i] - mean) * invStd;
			float dxhat = (*gradOutput.data)[r * DIM + i] * (*gamma.data)[i];
			xhatRow[i] = xhat;
			dxhatRow[i] = dxhat;
			sumDxhat += dxhat;
			sumDxhatXhat += dxhat * xhat;
			expectedDGamma[i] += (*gradOutput.data)[r * DIM + i] * xhat;
			expectedDBeta[i] += (*gradOutput.data)[r * DIM + i];
		}
		for (size_t i = 0; i < DIM; i++)
		{
			expectedDx[r * DIM + i] = invStd * (dxhatRow[i] - sumDxhat / (float)DIM - xhatRow[i] * sumDxhatXhat / (float)DIM);
		}
	}

	bool dxPass = checkVec("backward dx", *dx.data, expectedDx, dx.data->size() == expectedDx.size());
	bool dGammaPass = checkVec("backward dGamma", *dGamma.data, expectedDGamma, dGamma.data->size() == expectedDGamma.size());
	bool dBetaPass = checkVec("backward dBeta", *dBeta.data, expectedDBeta, dBeta.data->size() == expectedDBeta.size());

	bool allPass = fwdPass && dxPass && dGammaPass && dBetaPass;
	if (allPass)
	{
		std::cout << "\nAll LayerNormAffineOperation correctness checks passed.\n";
	}
	else
	{
		std::cout << "\nLayerNormAffineOperation correctness check FAILED.\n";
	}

	cudaMatMulShutdown();
	return allPass ? 0 : 1;
}
