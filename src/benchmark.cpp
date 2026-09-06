#include "Tensor.hpp"
#include "TransformerMiniModel.hpp"
#include <iostream>
#include <chrono>
#include <cstdlib>
#ifdef USE_CUDA
#include "CudaMatMul.hpp"
#include "CudaPool.hpp"
#include <cuda_runtime.h>
#endif

static inline void gpuSync()
{
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif
}

// synthetic-data throughput benchmark, compare against scripts/benchmark_pytorch.py

static const size_t VOCAB_SIZE   = 4096;
static const size_t SEQ_LEN      = 64;
static const size_t EMBED_DIM    = 256;
static const size_t DK           = 256;
static const size_t NUM_HEADS    = 8;
static const size_t NUM_LAYERS   = 6;
static const size_t BATCH_SIZE   = 16;
static const float  LR           = 3e-4f;
static const int    WARMUP_STEPS = 10;
static const int    BENCH_STEPS  = 50;

int main()
{
    srand(42);
#ifdef USE_CUDA
    cudaMatMulInit();
#endif

    TransformerMiniModel model(VOCAB_SIZE, EMBED_DIM, DK, NUM_LAYERS, NUM_HEADS, /*causal=*/true);
    std::cout << "Parameters: " << model.paramCount() << "\n";

    Tensor input(2, {BATCH_SIZE, SEQ_LEN});
    Tensor target(2, {BATCH_SIZE, SEQ_LEN});
    for (size_t i = 0; i < input.data->size(); i++)
    {
        (*input.data)[i]  = static_cast<float>(rand() % VOCAB_SIZE);
        (*target.data)[i] = static_cast<float>(rand() % VOCAB_SIZE);
    }

    auto runStep = [&]()
    {
        model.forward(input, target);
        model.cleanGradients();
        model.backward();
        model.applyGradient(LR);
    };

    for (int i = 0; i < WARMUP_STEPS; i++)
    {
        runStep();
    }
    gpuSync();

    double tForward = 0.0, tClean = 0.0, tBackward = 0.0, tOptim = 0.0;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < BENCH_STEPS; i++)
    {
        auto t0 = std::chrono::steady_clock::now();
        model.forward(input, target);
        gpuSync();
        auto t1 = std::chrono::steady_clock::now();

        model.cleanGradients();
        gpuSync();
        auto t2 = std::chrono::steady_clock::now();

        model.backward();
        gpuSync();
        auto t3 = std::chrono::steady_clock::now();

        model.applyGradient(LR);
        gpuSync();
        auto t4 = std::chrono::steady_clock::now();

        tForward  += std::chrono::duration<double>(t1 - t0).count();
        tClean    += std::chrono::duration<double>(t2 - t1).count();
        tBackward += std::chrono::duration<double>(t3 - t2).count();
        tOptim    += std::chrono::duration<double>(t4 - t3).count();
    }
    auto end = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    double msPerStep = elapsed * 1000.0 / BENCH_STEPS;
    double tokensPerSec = static_cast<double>(BATCH_SIZE * SEQ_LEN * BENCH_STEPS) / elapsed;

    std::cout << "Steps: " << BENCH_STEPS << "\n";
    std::cout << "Total time: " << elapsed << " s\n";
    std::cout << "Time/step: " << msPerStep << " ms\n";
    std::cout << "Tokens/sec: " << tokensPerSec << "\n";
    std::cout << "  forward:   " << (tForward  * 1000.0 / BENCH_STEPS) << " ms/step\n";
    std::cout << "  cleanGrad: " << (tClean    * 1000.0 / BENCH_STEPS) << " ms/step\n";
    std::cout << "  backward:  " << (tBackward * 1000.0 / BENCH_STEPS) << " ms/step\n";
    std::cout << "  optimizer: " << (tOptim    * 1000.0 / BENCH_STEPS) << " ms/step\n";

#ifdef USE_CUDA
    cudaPoolFlush();
    cudaMatMulShutdown();
#endif
    return 0;
}
