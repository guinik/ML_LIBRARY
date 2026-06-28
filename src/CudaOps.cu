#include "CudaOps.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstring>

#define CUDA_CHECK(x) \
    do \
    { \
        cudaError_t _e = (x); \
        if (_e != cudaSuccess) \
        { \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e)); \
        } \
    } while (0)

struct CudaDeleter
{
    void operator()(float* p) const
    {
        if (p)
        {
            cudaFree(p);
        }
    }
};

static Tensor makeCudaTensor(const std::vector<size_t>& shape, float* dPtr)
{
    Tensor t(shape.size(), shape);
    t.d_data = std::shared_ptr<float>(dPtr, CudaDeleter{});
    return t;
}

static Tensor makeZeroCudaTensor(const std::vector<size_t>& shape)
{
    size_t n = 1;
    for (auto d : shape) { n *= d; }
    float* dPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&dPtr, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(dPtr, 0, n * sizeof(float)));
    return makeCudaTensor(shape, dPtr);
}

static std::vector<size_t> broadcastShape(
    const std::vector<size_t>& a,
    const std::vector<size_t>& b)
{
    auto A = a, B = b;
    while (A.size() < B.size()) { A.insert(A.begin(), 1); }
    while (B.size() < A.size()) { B.insert(B.begin(), 1); }
    std::vector<size_t> out(A.size());
    for (size_t i = 0; i < A.size(); i++) { out[i] = std::max(A[i], B[i]); }
    return out;
}

// ─── Kernels ─────────────────────────────────────────────────────────────────

__global__ void broadcastBinopKernel(
    const float* A, const float* B, float* C,
    int d0, int d1, int d2, int d3,
    int sA0, int sA1, int sA2, int sA3,
    int sB0, int sB1, int sB2, int sB3,
    int total, int op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    int i3 = idx % d3;
    int tmp = idx / d3;
    int i2 = tmp % d2;
    tmp /= d2;
    int i1 = tmp % d1;
    int i0 = tmp / d1;
    int offA = i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3;
    int offB = i0 * sB0 + i1 * sB1 + i2 * sB2 + i3 * sB3;
    float a = A[offA], b = B[offB];
    if (op == 0) { C[idx] = a + b; }
    else if (op == 1) { C[idx] = a - b; }
    else { C[idx] = a * b; }
}

__global__ void scaleKernel(const float* in, float* out, float factor, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = in[i] * factor; }
}

__global__ void reluKernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = in[i] > 0.0f ? in[i] : 0.0f; }
}

__global__ void reluBackwardKernel(const float* in, const float* grad, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = in[i] > 0.0f ? grad[i] : 0.0f; }
}

__global__ void sigmoidKernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = 1.0f / (1.0f + expf(-in[i])); }
}

__global__ void sigmoidBackwardKernel(const float* out, const float* grad, float* dx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dx[i] = out[i] * (1.0f - out[i]) * grad[i]; }
}

__global__ void squareKernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = in[i] * in[i]; }
}

__global__ void squareBackwardKernel(const float* in, const float* grad, float* dx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dx[i] = 2.0f * in[i] * grad[i]; }
}

__global__ void causalMaskKernel(float* data, int numMatrices, int seq)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = numMatrices * seq * seq;
    if (idx >= total)
    {
        return;
    }
    int pos = idx % (seq * seq);
    int row = pos / seq;
    int col = pos % seq;
    if (col > row) { data[idx] = -1e9f; }
}

__global__ void softmaxKernel(const float* in, float* out, int rows, int cols)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) { return; }
    const float* x = in + row * cols;
    float* y = out + row * cols;

    float localMax = -1e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { localMax = fmaxf(localMax, x[i]); }
    smem[threadIdx.x] = localMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]); }
        __syncthreads();
    }
    float maxVal = smem[0];
    __syncthreads();

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float e = expf(x[i] - maxVal);
        y[i] = e;
        localSum += e;
    }
    smem[threadIdx.x] = localSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    float sumVal = smem[0];
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) { y[i] /= sumVal; }
}

__global__ void softmaxBackwardKernel(const float* out, const float* grad, float* dx, int rows, int cols)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) { return; }
    const float* o = out + row * cols;
    const float* g = grad + row * cols;
    float* d = dx + row * cols;

    float localDot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { localDot += o[i] * g[i]; }
    smem[threadIdx.x] = localDot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    float dot = smem[0];
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) { d[i] = o[i] * (g[i] - dot); }
}

__global__ void layerNormKernel(const float* in, float* out, int rows, int cols, float eps)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) { return; }
    const float* x = in + row * cols;
    float* y = out + row * cols;

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { localSum += x[i]; }
    smem[threadIdx.x] = localSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    float mean = smem[0] / cols;
    __syncthreads();

    float localVar = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float d = x[i] - mean;
        localVar += d * d;
    }
    smem[threadIdx.x] = localVar;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    float invStd = rsqrtf(smem[0] / cols + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) { y[i] = (x[i] - mean) * invStd; }
}

__global__ void layerNormBackwardKernel(
    const float* x, const float* xhat, const float* dout,
    float* dx, int rows, int cols, float eps)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) { return; }
    const float* xr = x + row * cols;
    const float* xhr = xhat + row * cols;
    const float* dr = dout + row * cols;
    float* dxr = dx + row * cols;

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { localSum += xr[i]; }
    smem[threadIdx.x] = localSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    float mean = smem[0] / cols;
    __syncthreads();

    float localVar = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float d = xr[i] - mean;
        localVar += d * d;
    }
    smem[threadIdx.x] = localVar;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    float invStd = rsqrtf(smem[0] / cols + eps);
    __syncthreads();

    float localSumDout = 0.0f, localSumDoutXhat = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        localSumDout += dr[i];
        localSumDoutXhat += dr[i] * xhr[i];
    }
    smem[threadIdx.x] = localSumDout;
    smem[blockDim.x + threadIdx.x] = localSumDoutXhat;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            smem[threadIdx.x] += smem[threadIdx.x + s];
            smem[blockDim.x + threadIdx.x] += smem[blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }
    float sumDout = smem[0];
    float sumDoutXhat = smem[blockDim.x];
    __syncthreads();

    float fn = (float)cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        dxr[i] = invStd * (dr[i] - sumDout / fn - xhr[i] * sumDoutXhat / fn);
    }
}

__global__ void embeddingForwardKernel(
    const float* ids, const float* weights, float* out,
    int numTokens, int embedDim)
{
    int token = blockIdx.x;
    int d = threadIdx.x;
    if (token >= numTokens || d >= embedDim) { return; }
    int id = (int)ids[token];
    out[token * embedDim + d] = weights[id * embedDim + d];
}

__global__ void embeddingBackwardKernel(
    const float* ids, const float* gradOut, float* gradWeights,
    int numTokens, int embedDim)
{
    int token = blockIdx.x;
    int d = threadIdx.x;
    if (token >= numTokens || d >= embedDim) { return; }
    int id = (int)ids[token];
    atomicAdd(&gradWeights[id * embedDim + d], gradOut[token * embedDim + d]);
}

__global__ void crossEntropyForwardKernel(
    const float* logits, const float* targets, float* losses,
    int rows, int vocabSize)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) { return; }
    const float* lr = logits + row * vocabSize;

    float localMax = -1e38f;
    for (int j = threadIdx.x; j < vocabSize; j += blockDim.x) { localMax = fmaxf(localMax, lr[j]); }
    smem[threadIdx.x] = localMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]); }
        __syncthreads();
    }
    float maxVal = smem[0];
    __syncthreads();

    float localSum = 0.0f;
    for (int j = threadIdx.x; j < vocabSize; j += blockDim.x) { localSum += expf(lr[j] - maxVal); }
    smem[threadIdx.x] = localSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    float sumExp = smem[0];

    if (threadIdx.x == 0)
    {
        int tgt = (int)targets[row];
        losses[row] = -(lr[tgt] - maxVal - logf(sumExp));
    }
}

__global__ void crossEntropyBackwardKernel(
    const float* logits, const float* targets, const float* gradOut,
    float* gradLogits, int rows, int vocabSize)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) { return; }
    const float* lr = logits + row * vocabSize;
    float* gr = gradLogits + row * vocabSize;

    float localMax = -1e38f;
    for (int j = threadIdx.x; j < vocabSize; j += blockDim.x) { localMax = fmaxf(localMax, lr[j]); }
    smem[threadIdx.x] = localMax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]); }
        __syncthreads();
    }
    float maxVal = smem[0];
    __syncthreads();

    float localSum = 0.0f;
    for (int j = threadIdx.x; j < vocabSize; j += blockDim.x) { localSum += expf(lr[j] - maxVal); }
    smem[threadIdx.x] = localSum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    float sumExp = smem[0];
    __syncthreads();

    int tgt = (int)targets[row];
    float gi = gradOut[row];
    for (int j = threadIdx.x; j < vocabSize; j += blockDim.x)
    {
        float prob = expf(lr[j] - maxVal) / sumExp;
        gr[j] = gi * prob;
        if (j == tgt) { gr[j] -= gi; }
    }
}

__global__ void reduceLeadingKernel(const float* src, float* dst, int leadTotal, int lastDim)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= lastDim) { return; }
    float sum = 0.0f;
    for (int i = 0; i < leadTotal; i++) { sum += src[i * lastDim + d]; }
    dst[d] = sum;
}

// ─── Host functions ───────────────────────────────────────────────────────────

static Tensor broadcastBinop(const Tensor& A, const Tensor& B, int op)
{
    A.toGPU();
    B.toGPU();

    auto outShape = broadcastShape(A.shape, B.shape);
    size_t total = 1;
    for (auto d : outShape) { total *= d; }

    // Pad A and B shapes/strides to outShape rank, then to 4D
    auto shA = A.shape, shB = B.shape;
    auto stA = A.strides, stB = B.strides;
    while (shA.size() < outShape.size()) { shA.insert(shA.begin(), 1); stA.insert(stA.begin(), 0); }
    while (shB.size() < outShape.size()) { shB.insert(shB.begin(), 1); stB.insert(stB.begin(), 0); }

    // Zero strides for broadcast dims
    for (size_t i = 0; i < outShape.size(); i++)
    {
        if (shA[i] == 1 && outShape[i] != 1) { stA[i] = 0; }
        if (shB[i] == 1 && outShape[i] != 1) { stB[i] = 0; }
    }

    // Pad to 4D
    auto out4 = outShape;
    while (out4.size() < 4) { out4.insert(out4.begin(), 1); stA.insert(stA.begin(), 0); stB.insert(stB.begin(), 0); }

    int shapeArr[4], sA4[4], sB4[4];
    for (int i = 0; i < 4; i++)
    {
        shapeArr[i] = (int)out4[i];
        sA4[i] = (int)stA[i];
        sB4[i] = (int)stB[i];
    }

    float* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dC, total * sizeof(float)));
    int threads = 256;
    int blocks = ((int)total + threads - 1) / threads;
    broadcastBinopKernel<<<blocks, threads>>>(
        A.d_data.get(), B.d_data.get(), dC,
        shapeArr[0], shapeArr[1], shapeArr[2], shapeArr[3],
        sA4[0], sA4[1], sA4[2], sA4[3],
        sB4[0], sB4[1], sB4[2], sB4[3],
        (int)total, op);

    return makeCudaTensor(outShape, dC);
}

Tensor cudaAdd(const Tensor& A, const Tensor& B)
{
    return broadcastBinop(A, B, 0);
}

Tensor cudaSubtract(const Tensor& A, const Tensor& B)
{
    return broadcastBinop(A, B, 1);
}

Tensor cudaElemMul(const Tensor& A, const Tensor& B)
{
    return broadcastBinop(A, B, 2);
}

Tensor cudaScale(const Tensor& A, float factor)
{
    A.toGPU();
    int n = (int)A.data->size();
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scaleKernel<<<blocks, threads>>>(A.d_data.get(), dOut, factor, n);
    return makeCudaTensor(A.shape, dOut);
}

Tensor cudaRelu(const Tensor& A)
{
    A.toGPU();
    int n = (int)A.data->size();
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(A.d_data.get(), dOut, n);
    return makeCudaTensor(A.shape, dOut);
}

Tensor cudaReluBackward(const Tensor& input, const Tensor& gradOutput)
{
    input.toGPU();
    gradOutput.toGPU();
    int n = (int)input.data->size();
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    reluBackwardKernel<<<blocks, threads>>>(input.d_data.get(), gradOutput.d_data.get(), dOut, n);
    return makeCudaTensor(input.shape, dOut);
}

Tensor cudaSigmoid(const Tensor& A)
{
    A.toGPU();
    int n = (int)A.data->size();
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoidKernel<<<blocks, threads>>>(A.d_data.get(), dOut, n);
    return makeCudaTensor(A.shape, dOut);
}

Tensor cudaSigmoidBackward(const Tensor& output, const Tensor& gradOutput)
{
    output.toGPU();
    gradOutput.toGPU();
    int n = (int)output.data->size();
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoidBackwardKernel<<<blocks, threads>>>(output.d_data.get(), gradOutput.d_data.get(), dOut, n);
    return makeCudaTensor(output.shape, dOut);
}

Tensor cudaSquare(const Tensor& A)
{
    A.toGPU();
    int n = (int)A.data->size();
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    squareKernel<<<blocks, threads>>>(A.d_data.get(), dOut, n);
    return makeCudaTensor(A.shape, dOut);
}

Tensor cudaSquareBackward(const Tensor& input, const Tensor& gradOutput)
{
    input.toGPU();
    gradOutput.toGPU();
    int n = (int)input.data->size();
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    squareBackwardKernel<<<blocks, threads>>>(input.d_data.get(), gradOutput.d_data.get(), dOut, n);
    return makeCudaTensor(input.shape, dOut);
}

Tensor cudaSoftmax(const Tensor& A)
{
    A.toGPU();
    size_t cols = A.shape.back();
    size_t rows = A.data->size() / cols;
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, A.data->size() * sizeof(float)));
    int threads = 256;
    size_t smem = threads * sizeof(float);
    softmaxKernel<<<(int)rows, threads, smem>>>(A.d_data.get(), dOut, (int)rows, (int)cols);
    return makeCudaTensor(A.shape, dOut);
}

Tensor cudaSoftmaxBackward(const Tensor& output, const Tensor& gradOutput)
{
    output.toGPU();
    gradOutput.toGPU();
    size_t cols = output.shape.back();
    size_t rows = output.data->size() / cols;
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, output.data->size() * sizeof(float)));
    int threads = 256;
    size_t smem = threads * sizeof(float);
    softmaxBackwardKernel<<<(int)rows, threads, smem>>>(
        output.d_data.get(), gradOutput.d_data.get(), dOut, (int)rows, (int)cols);
    return makeCudaTensor(output.shape, dOut);
}

Tensor cudaLayerNorm(const Tensor& A, float eps)
{
    A.toGPU();
    size_t cols = A.shape.back();
    size_t rows = A.data->size() / cols;
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, A.data->size() * sizeof(float)));
    int threads = 256;
    size_t smem = threads * sizeof(float);
    layerNormKernel<<<(int)rows, threads, smem>>>(A.d_data.get(), dOut, (int)rows, (int)cols, eps);
    return makeCudaTensor(A.shape, dOut);
}

Tensor cudaLayerNormBackward(const Tensor& x, const Tensor& xhat, const Tensor& gradOutput, float eps)
{
    x.toGPU();
    xhat.toGPU();
    gradOutput.toGPU();
    size_t cols = x.shape.back();
    size_t rows = x.data->size() / cols;
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, x.data->size() * sizeof(float)));
    int threads = 256;
    size_t smem = 2 * threads * sizeof(float);
    layerNormBackwardKernel<<<(int)rows, threads, smem>>>(
        x.d_data.get(), xhat.d_data.get(), gradOutput.d_data.get(),
        dOut, (int)rows, (int)cols, eps);
    return makeCudaTensor(x.shape, dOut);
}

Tensor cudaCausalMask(const Tensor& A)
{
    A.toGPU();
    int seq = (int)A.shape.back();
    int total = (int)A.data->size();
    int numMatrices = total / (seq * seq);
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dOut, A.d_data.get(), total * sizeof(float), cudaMemcpyDeviceToDevice));
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    causalMaskKernel<<<blocks, threads>>>(dOut, numMatrices, seq);
    return makeCudaTensor(A.shape, dOut);
}

Tensor cudaEmbeddingForward(const Tensor& ids, const Tensor& weights)
{
    ids.toGPU();
    weights.toGPU();
    int numTokens = (int)ids.data->size();
    int embedDim = (int)weights.shape[1];
    auto outShape = ids.shape;
    outShape.push_back((size_t)embedDim);
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, numTokens * embedDim * sizeof(float)));
    embeddingForwardKernel<<<numTokens, embedDim>>>(
        ids.d_data.get(), weights.d_data.get(), dOut, numTokens, embedDim);
    return makeCudaTensor(outShape, dOut);
}

Tensor cudaEmbeddingBackwardWeights(const Tensor& ids, const Tensor& weightsRef, const Tensor& gradOutput)
{
    ids.toGPU();
    gradOutput.toGPU();
    int numTokens = (int)ids.data->size();
    int embedDim = (int)weightsRef.shape[1];
    int vocabSize = (int)weightsRef.shape[0];
    Tensor gradWeights = makeZeroCudaTensor(weightsRef.shape);
    embeddingBackwardKernel<<<numTokens, embedDim>>>(
        ids.d_data.get(), gradOutput.d_data.get(), gradWeights.d_data.get(),
        numTokens, embedDim);
    (void)vocabSize;
    return gradWeights;
}

Tensor cudaCrossEntropyForward(const Tensor& logits, const Tensor& targets)
{
    logits.toGPU();
    targets.toGPU();
    int vocabSize = (int)logits.shape.back();
    int rows = (int)(logits.data->size() / vocabSize);
    auto outShape = std::vector<size_t>(logits.shape.begin(), logits.shape.end() - 1);
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, rows * sizeof(float)));
    int threads = 256;
    size_t smem = threads * sizeof(float);
    crossEntropyForwardKernel<<<rows, threads, smem>>>(
        logits.d_data.get(), targets.d_data.get(), dOut, rows, vocabSize);
    return makeCudaTensor(outShape, dOut);
}

std::pair<Tensor, Tensor> cudaCrossEntropyBackward(
    const Tensor& logits, const Tensor& targets, const Tensor& gradOutput)
{
    logits.toGPU();
    targets.toGPU();
    gradOutput.toGPU();
    int vocabSize = (int)logits.shape.back();
    int rows = (int)(logits.data->size() / vocabSize);
    float* dGrad = nullptr;
    CUDA_CHECK(cudaMalloc(&dGrad, logits.data->size() * sizeof(float)));
    int threads = 256;
    size_t smem = threads * sizeof(float);
    crossEntropyBackwardKernel<<<rows, threads, smem>>>(
        logits.d_data.get(), targets.d_data.get(), gradOutput.d_data.get(),
        dGrad, rows, vocabSize);
    Tensor gradLogits = makeCudaTensor(logits.shape, dGrad);
    Tensor gradTargets = makeZeroCudaTensor(targets.shape);
    return { gradLogits, gradTargets };
}

Tensor cudaUnbroadcast(const Tensor& grad, const std::vector<size_t>& targetShape)
{
    if (grad.shape == targetShape)
    {
        return grad;
    }
    size_t extraDims = grad.shape.size() - targetShape.size();
    if (extraDims == 0)
    {
        return grad;
    }
    size_t leadTotal = 1;
    for (size_t i = 0; i < extraDims; i++) { leadTotal *= grad.shape[i]; }
    size_t lastDim = 1;
    for (auto d : targetShape) { lastDim *= d; }
    grad.toGPU();
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, lastDim * sizeof(float)));
    int threads = 256;
    int blocks = ((int)lastDim + threads - 1) / threads;
    reduceLeadingKernel<<<blocks, threads>>>(grad.d_data.get(), dOut, (int)leadTotal, (int)lastDim);
    return makeCudaTensor(targetShape, dOut);
}
