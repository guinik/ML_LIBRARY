#include "Tensor.hpp"
#include "CudaMatMul.hpp"
#include "Node.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

// Standalone numerical correctness check for the cuBLASLt rewrite of
// cudaMatMul. Deliberately does NOT reuse any indexing logic from
// CudaMatMul.cu/Node.cpp -- it's an independent textbook triple-loop
// implementation, so a bug in the library's shape math can't also be
// baked into the "expected" values here. Every (transpose-flag, batch,
// broadcast) combination actually used by TransformerMiniModel's forward
// and backward passes is exercised below.

static std::vector<float> referenceMatMul(
    const std::vector<float>& Adata, const std::vector<size_t>& Ashape, bool tA,
    const std::vector<float>& Bdata, const std::vector<size_t>& Bshape, bool tB,
    size_t& outBatch, size_t& outM, size_t& outN)
{
    size_t Arows = Ashape[Ashape.size() - 2];
    size_t Acols = Ashape.back();
    size_t Brows = Bshape[Bshape.size() - 2];
    size_t Bcols = Bshape.back();

    size_t M = tA ? Acols : Arows;
    size_t K = tA ? Arows : Acols;
    size_t Kb = tB ? Bcols : Brows;
    size_t N = tB ? Brows : Bcols;
    if (K != Kb)
    {
        throw std::runtime_error("referenceMatMul: K mismatch");
    }

    size_t batchA = 1;
    for (size_t i = 0; i + 2 < Ashape.size(); i++) { batchA *= Ashape[i]; }
    size_t batchB = 1;
    for (size_t i = 0; i + 2 < Bshape.size(); i++) { batchB *= Bshape[i]; }
    size_t batch = std::max(batchA, batchB);

    outBatch = batch;
    outM = M;
    outN = N;
    std::vector<float> C(batch * M * N, 0.0f);

    for (size_t b = 0; b < batch; b++)
    {
        size_t ba = (batchA == 1) ? 0 : b;
        size_t bb = (batchB == 1) ? 0 : b;
        const float* Ab = Adata.data() + ba * Arows * Acols;
        const float* Bb = Bdata.data() + bb * Brows * Bcols;
        float* Cb = C.data() + b * M * N;
        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                float sum = 0.0f;
                for (size_t p = 0; p < K; p++)
                {
                    float aVal = tA ? Ab[p * Acols + i] : Ab[i * Acols + p];
                    float bVal = tB ? Bb[j * Bcols + p] : Bb[p * Bcols + j];
                    sum += aVal * bVal;
                }
                Cb[i * N + j] = sum;
            }
        }
    }
    return C;
}

static Tensor makeRandomTensor(const std::vector<size_t>& shape)
{
    Tensor t(shape.size(), shape);
    for (size_t i = 0; i < t.data->size(); i++)
    {
        (*t.data)[i] = static_cast<float>((rand() % 2000) - 1000) / 100.0f;
    }
    return t;
}

struct TestCase
{
    const char* name;
    std::vector<size_t> Ashape;
    bool tA;
    std::vector<size_t> Bshape;
    bool tB;
};

int main()
{
    srand(1234);
    cudaMatMulInit();

    std::vector<TestCase> cases = {
        {"no-transpose, batched, no broadcast",         {2, 5, 3}, false, {2, 3, 4}, false},
        {"transpose-B, weight broadcast (Dense-like)",  {2, 5, 3}, false, {4, 3},    true},
        {"transpose-A, batched, no broadcast",          {2, 3, 5}, true,  {2, 3, 4}, false},
        {"transpose-A + transpose-B, batched",          {2, 3, 5}, true,  {4, 3},    true},
        {"no-transpose, batched, both operands batched, square-ish", {3, 6, 6}, false, {3, 6, 6}, false},
    };

    int failures = 0;
    uint16_t noTrans = MatMulFlags::MATMUL_NO_TRANSPOSES;

    for (auto& tc : cases)
    {
        Tensor A = makeRandomTensor(tc.Ashape);
        Tensor B = makeRandomTensor(tc.Bshape);

        uint16_t flags = noTrans;
        if (tc.tA) { flags |= MatMulFlags::MATMUL_TRANSPOSE_A; }
        if (tc.tB) { flags |= MatMulFlags::MATMUL_TRANSPOSE_B; }

        Tensor gpuResult = cudaMatMul(A, B, flags);
        gpuResult.toCPU();

        size_t refBatch = 0, refM = 0, refN = 0;
        std::vector<float> expected = referenceMatMul(
            *A.data, A.shape, tc.tA, *B.data, B.shape, tc.tB, refBatch, refM, refN);

        bool shapeOk = gpuResult.data->size() == expected.size();
        float maxDiff = 0.0f;
        if (shapeOk)
        {
            for (size_t i = 0; i < expected.size(); i++)
            {
                float diff = std::fabs((*gpuResult.data)[i] - expected[i]);
                maxDiff = std::max(maxDiff, diff);
            }
        }

        bool pass = shapeOk && maxDiff < 1e-2f;
        std::cout << (pass ? "[PASS] " : "[FAIL] ") << tc.name
                  << " -- shapeOk=" << shapeOk
                  << " maxDiff=" << maxDiff << "\n";
        if (!pass) { failures++; }
    }

    if (failures == 0)
    {
        std::cout << "\nAll matmul correctness checks passed.\n";
    }
    else
    {
        std::cout << "\n" << failures << " matmul correctness check(s) FAILED.\n";
    }

    cudaMatMulShutdown();
    return failures == 0 ? 0 : 1;
}
