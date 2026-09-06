#include "Tensor.hpp"
#include "Node.hpp"
#include "Layers.hpp"
#include "ExecutionGraph.hpp"
#include "CudaMatMul.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// checks SplitHeadsOperation/MergeHeadsOperation and the MultiHeadAttention layer built on
// top of them against independent brute-force references

static Tensor makeRandomTensor(const std::vector<size_t>& shape)
{
	Tensor t(shape.size(), shape);
	for (size_t i = 0; i < t.data->size(); i++)
	{
		(*t.data)[i] = static_cast<float>((rand() % 2000) - 1000) / 100.0f;
	}
	return t;
}

static std::vector<float> toCpuVec(Tensor t)
{
#ifdef USE_CUDA
	t.toCPU();
#endif
	return *t.data;
}

static bool checkVec(const char* name, const std::vector<float>& got, const std::vector<float>& want, float tol)
{
	bool shapeOk = got.size() == want.size();
	float maxDiff = 0.0f;
	if (shapeOk)
	{
		for (size_t i = 0; i < want.size(); i++)
		{
			maxDiff = std::max(maxDiff, std::fabs(got[i] - want[i]));
		}
	}
	bool pass = shapeOk && maxDiff < tol;
	std::cout << (pass ? "[PASS] " : "[FAIL] ") << name
		<< " -- shapeOk=" << shapeOk << " maxDiff=" << maxDiff << "\n";
	return pass;
}

// independent reference for (batch,seq,heads*dHead) -> (batch,heads,seq,dHead)
static std::vector<float> refSplit(const std::vector<float>& x, size_t batch, size_t seq, size_t heads, size_t dHead)
{
	size_t dK = heads * dHead;
	std::vector<float> out(batch * heads * seq * dHead);
	for (size_t b = 0; b < batch; b++)
		for (size_t h = 0; h < heads; h++)
			for (size_t s = 0; s < seq; s++)
				for (size_t d = 0; d < dHead; d++)
					out[((b * heads + h) * seq + s) * dHead + d] = x[(b * seq + s) * dK + h * dHead + d];
	return out;
}

// independent reference for (batch,heads,seq,dHead) -> (batch,seq,heads*dHead), the inverse of refSplit
static std::vector<float> refMerge(const std::vector<float>& x, size_t batch, size_t heads, size_t seq, size_t dHead)
{
	size_t dK = heads * dHead;
	std::vector<float> out(batch * seq * dK);
	for (size_t b = 0; b < batch; b++)
		for (size_t h = 0; h < heads; h++)
			for (size_t s = 0; s < seq; s++)
				for (size_t d = 0; d < dHead; d++)
					out[(b * seq + s) * dK + h * dHead + d] = x[((b * heads + h) * seq + s) * dHead + d];
	return out;
}

// brute-force scaled dot-product multi-head attention, independent of the library's own
// op-graph implementation -- used as ground truth for the MultiHeadAttention layer.
// x: (batch,seq,dModel), Wq/Wk/Wv: (dK,dModel), Wo: (dModel,dK), dK = heads*headDim
static std::vector<float> refMultiHeadAttention(
	const std::vector<float>& x, size_t batch, size_t seq, size_t dModel,
	const std::vector<float>& Wq, const std::vector<float>& Wk, const std::vector<float>& Wv,
	const std::vector<float>& Wo, size_t heads, size_t headDim, bool causal)
{
	size_t dK = heads * headDim;

	auto project = [&](const std::vector<float>& W) {
		std::vector<float> out(batch * seq * dK, 0.0f);
		for (size_t b = 0; b < batch; b++)
			for (size_t s = 0; s < seq; s++)
				for (size_t n = 0; n < dK; n++)
				{
					float sum = 0.0f;
					for (size_t k = 0; k < dModel; k++)
					{
						sum += x[(b * seq + s) * dModel + k] * W[n * dModel + k];
					}
					out[(b * seq + s) * dK + n] = sum;
				}
		return out;
	};

	std::vector<float> Q = project(Wq);
	std::vector<float> K = project(Wk);
	std::vector<float> V = project(Wv);

	std::vector<float> merged(batch * seq * dK, 0.0f);
	float scale = 1.0f / std::sqrt((float)headDim);

	for (size_t b = 0; b < batch; b++)
	{
		for (size_t h = 0; h < heads; h++)
		{
			for (size_t i = 0; i < seq; i++)
			{
				std::vector<float> scores(seq, 0.0f);
				for (size_t j = 0; j < seq; j++)
				{
					if (causal && j > i)
					{
						scores[j] = -1e9f;
						continue;
					}
					float sum = 0.0f;
					for (size_t d = 0; d < headDim; d++)
					{
						sum += Q[(b * seq + i) * dK + h * headDim + d] * K[(b * seq + j) * dK + h * headDim + d];
					}
					scores[j] = sum * scale;
				}
				float maxVal = *std::max_element(scores.begin(), scores.end());
				float denom = 0.0f;
				for (float& s : scores)
				{
					s = std::exp(s - maxVal);
					denom += s;
				}
				for (float& s : scores) { s /= denom; }

				for (size_t d = 0; d < headDim; d++)
				{
					float sum = 0.0f;
					for (size_t j = 0; j < seq; j++)
					{
						sum += scores[j] * V[(b * seq + j) * dK + h * headDim + d];
					}
					merged[(b * seq + i) * dK + h * headDim + d] = sum;
				}
			}
		}
	}

	std::vector<float> out(batch * seq * dModel, 0.0f);
	for (size_t b = 0; b < batch; b++)
		for (size_t s = 0; s < seq; s++)
			for (size_t n = 0; n < dModel; n++)
			{
				float sum = 0.0f;
				for (size_t k = 0; k < dK; k++)
				{
					sum += merged[(b * seq + s) * dK + k] * Wo[n * dK + k];
				}
				out[(b * seq + s) * dModel + n] = sum;
			}
	return out;
}

// runs the actual op sequence a MultiHeadAttention layer builds, calling Operation::forward
// directly (no Node/ExecutionGraph), so this exercises the real implementation code paths
static std::vector<float> runOpSequence(
	const Tensor& x, const Tensor& Wq, const Tensor& Wk, const Tensor& Wv, const Tensor& Wo,
	size_t heads, bool causal)
{
	MatMulOperation projOp(MatMulFlags::MATMUL_TRANSPOSE_B);
	Tensor Q = projOp.forward({ &x, &Wq });
	Tensor K = projOp.forward({ &x, &Wk });
	Tensor V = projOp.forward({ &x, &Wv });

	SplitHeadsOperation splitOp(heads);
	Tensor Qh = splitOp.forward({ &Q });
	Tensor Kh = splitOp.forward({ &K });
	Tensor Vh = splitOp.forward({ &V });

	size_t headDim = Qh.shape[3];

	MatMulOperation qkOp(MatMulFlags::MATMUL_TRANSPOSE_B);
	Tensor QK = qkOp.forward({ &Qh, &Kh });

	ScaleOperation scaleOp(1.0f / std::sqrt((float)headDim));
	Tensor scaled = scaleOp.forward({ &QK });

	Tensor masked = scaled;
	if (causal)
	{
		CausalMaskOperation maskOp;
		masked = maskOp.forward({ &scaled });
	}

	SoftmaxOperation softmaxOp;
	Tensor softmaxRes = softmaxOp.forward({ &masked });

	MatMulOperation attnOp(MatMulFlags::MATMUL_NO_TRANSPOSES);
	Tensor attnHeads = attnOp.forward({ &softmaxRes, &Vh });

	MergeHeadsOperation mergeOp;
	Tensor merged = mergeOp.forward({ &attnHeads });

	MatMulOperation outOp(MatMulFlags::MATMUL_TRANSPOSE_B);
	Tensor output = outOp.forward({ &merged, &Wo });

	return toCpuVec(output);
}

int main()
{
	srand(1234);
	cudaMatMulInit();
	bool allPass = true;

	// --- Case 1: split/merge round-trip identity ---
	{
		const size_t BATCH = 2, SEQ = 5, HEADS = 4, HEAD_DIM = 3;
		const size_t D_K = HEADS * HEAD_DIM;

		Tensor x3 = makeRandomTensor({ BATCH, SEQ, D_K });
		SplitHeadsOperation splitOp(HEADS);
		MergeHeadsOperation mergeOp;

		Tensor split = splitOp.forward({ &x3 });
		Tensor roundTrip3 = mergeOp.forward({ &split });
		allPass &= checkVec("split->merge round-trip (3D identity)", toCpuVec(roundTrip3), *x3.data, 1e-6f);

		Tensor x4 = makeRandomTensor({ BATCH, HEADS, SEQ, HEAD_DIM });
		Tensor merged = mergeOp.forward({ &x4 });
		Tensor roundTrip4 = splitOp.forward({ &merged });
		allPass &= checkVec("merge->split round-trip (4D identity)", toCpuVec(roundTrip4), *x4.data, 1e-6f);
	}

	// --- Case 2: split forward vs. brute-force index math ---
	{
		const size_t BATCH = 2, SEQ = 4, HEADS = 3, HEAD_DIM = 2;
		const size_t D_K = HEADS * HEAD_DIM;

		Tensor x = makeRandomTensor({ BATCH, SEQ, D_K });
		SplitHeadsOperation splitOp(HEADS);
		Tensor got = splitOp.forward({ &x });
		std::vector<float> want = refSplit(*x.data, BATCH, SEQ, HEADS, HEAD_DIM);
		allPass &= checkVec("SplitHeadsOperation::forward vs. brute-force index math", toCpuVec(got), want, 1e-6f);
	}

	// --- Case 3: split/merge backward vs. brute-force reindex of gradOutput ---
	{
		const size_t BATCH = 2, SEQ = 4, HEADS = 3, HEAD_DIM = 2;
		const size_t D_K = HEADS * HEAD_DIM;

		Tensor x = makeRandomTensor({ BATCH, SEQ, D_K });
		SplitHeadsOperation splitOp(HEADS);
		Tensor dummyOut;
		Tensor gradOutSplit = makeRandomTensor({ BATCH, HEADS, SEQ, HEAD_DIM });
		std::vector<Tensor> splitGrads = splitOp.backward({ &x }, dummyOut, gradOutSplit);
		std::vector<float> wantSplitGrad = refMerge(*gradOutSplit.data, BATCH, HEADS, SEQ, HEAD_DIM);
		allPass &= checkVec("SplitHeadsOperation::backward vs. brute-force reindex", toCpuVec(splitGrads[0]), wantSplitGrad, 1e-6f);

		Tensor x4 = makeRandomTensor({ BATCH, HEADS, SEQ, HEAD_DIM });
		MergeHeadsOperation mergeOp;
		Tensor gradOutMerge = makeRandomTensor({ BATCH, SEQ, D_K });
		std::vector<Tensor> mergeGrads = mergeOp.backward({ &x4 }, dummyOut, gradOutMerge);
		std::vector<float> wantMergeGrad = refSplit(*gradOutMerge.data, BATCH, HEADS, SEQ, HEAD_DIM);
		allPass &= checkVec("MergeHeadsOperation::backward vs. brute-force reindex", toCpuVec(mergeGrads[0]), wantMergeGrad, 1e-6f);
	}

	// --- Case 4: numHeads=1 reduces exactly to the old single-head formula ---
	{
		const size_t BATCH = 2, SEQ = 4, D_MODEL = 6, D_K = 6, HEADS = 1;

		Tensor x = makeRandomTensor({ BATCH, SEQ, D_MODEL });
		Tensor Wq = makeRandomTensor({ D_K, D_MODEL });
		Tensor Wk = makeRandomTensor({ D_K, D_MODEL });
		Tensor Wv = makeRandomTensor({ D_K, D_MODEL });
		Tensor Wo = makeRandomTensor({ D_MODEL, D_K });

		std::vector<float> got = runOpSequence(x, Wq, Wk, Wv, Wo, HEADS, /*causal=*/true);
		std::vector<float> want = refMultiHeadAttention(
			*x.data, BATCH, SEQ, D_MODEL, *Wq.data, *Wk.data, *Wv.data, *Wo.data, HEADS, D_K / HEADS, /*causal=*/true);
		allPass &= checkVec("numHeads=1 matches old single-head formula", got, want, 1e-2f);
	}

	// --- Case 5: numHeads=4 genuine multi-head case ---
	{
		const size_t BATCH = 2, SEQ = 5, D_MODEL = 16, D_K = 16, HEADS = 4;

		Tensor x = makeRandomTensor({ BATCH, SEQ, D_MODEL });
		Tensor Wq = makeRandomTensor({ D_K, D_MODEL });
		Tensor Wk = makeRandomTensor({ D_K, D_MODEL });
		Tensor Wv = makeRandomTensor({ D_K, D_MODEL });
		Tensor Wo = makeRandomTensor({ D_MODEL, D_K });

		std::vector<float> got = runOpSequence(x, Wq, Wk, Wv, Wo, HEADS, /*causal=*/true);
		std::vector<float> want = refMultiHeadAttention(
			*x.data, BATCH, SEQ, D_MODEL, *Wq.data, *Wk.data, *Wv.data, *Wo.data, HEADS, D_K / HEADS, /*causal=*/true);
		allPass &= checkVec("numHeads=4 multi-head attention vs. brute-force reference", got, want, 1e-2f);
	}

	// --- Case 6: end-to-end finite-difference gradient check through the real Node/ExecutionGraph ---
	{
		const size_t BATCH = 1, SEQ = 3, D_MODEL = 6, HEADS = 2;

		MultiHeadAttention attn(D_MODEL, D_MODEL, HEADS, /*causal=*/true);
		attn.queryWeights->param.value = makeRandomTensor({ D_MODEL, D_MODEL }) * 0.3f;
		attn.keyWeights->param.value   = makeRandomTensor({ D_MODEL, D_MODEL }) * 0.3f;
		attn.valueWeights->param.value = makeRandomTensor({ D_MODEL, D_MODEL }) * 0.3f;
		attn.outputWeights->param.value = makeRandomTensor({ D_MODEL, D_MODEL }) * 0.3f;

		auto inputNode = std::make_shared<Node>();
		inputNode->param.value = makeRandomTensor({ BATCH, SEQ, D_MODEL });

		auto outputNode = attn.forward({ inputNode });
		ExecutionGraph graph(outputNode);

		auto sumOfOutput = [&]() {
			graph.computeForward();
			std::vector<float> out = toCpuVec(outputNode->param.value);
			float sum = 0.0f;
			for (float v : out) { sum += v; }
			return sum;
		};

		graph.computeForward();
		graph.cleanGradients();
		outputNode->param.grad = Tensor(outputNode->param.value.dimensions, outputNode->param.value.shape);
		outputNode->param.grad.fillValues(1.0f);
		graph.computeBackward();

		Tensor analyticGrad = attn.queryWeights->param.grad;
#ifdef USE_CUDA
		analyticGrad.toCPU();
#endif

		const float eps = 1e-3f;
		float maxDiff = 0.0f;
		// spot-check a handful of Wq elements rather than the full matrix, finite differences
		// are just a correctness sanity check here, not exhaustive
		size_t n = attn.queryWeights->param.value.data->size();
		size_t numChecks = std::min<size_t>(n, 8);
		for (size_t i = 0; i < numChecks; i++)
		{
			size_t idx = i * (n / numChecks);
			float orig = (*attn.queryWeights->param.value.data)[idx];

			(*attn.queryWeights->param.value.data)[idx] = orig + eps;
#ifdef USE_CUDA
			attn.queryWeights->param.value.invalidateGPU();
#endif
			float lossPlus = sumOfOutput();

			(*attn.queryWeights->param.value.data)[idx] = orig - eps;
#ifdef USE_CUDA
			attn.queryWeights->param.value.invalidateGPU();
#endif
			float lossMinus = sumOfOutput();

			(*attn.queryWeights->param.value.data)[idx] = orig;
#ifdef USE_CUDA
			attn.queryWeights->param.value.invalidateGPU();
#endif

			float numericGrad = (lossPlus - lossMinus) / (2.0f * eps);
			float analytic = (*analyticGrad.data)[idx];
			maxDiff = std::max(maxDiff, std::fabs(numericGrad - analytic));
		}

		bool gradPass = maxDiff < 5e-1f;
		std::cout << (gradPass ? "[PASS] " : "[FAIL] ") << "end-to-end finite-difference gradient check (Wq)"
			<< " -- maxDiff=" << maxDiff << "\n";
		allPass &= gradPass;
	}

	if (allPass)
	{
		std::cout << "\nAll multi-head attention correctness checks passed.\n";
	}
	else
	{
		std::cout << "\nMulti-head attention correctness check FAILED.\n";
	}

	cudaMatMulShutdown();
	return allPass ? 0 : 1;
}
