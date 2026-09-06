#include "Tensor.hpp"
#include "TransformerMiniModel.hpp"
#include "DataLoader.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <cmath>
#include <string>
#ifdef USE_CUDA
#include "CudaMatMul.hpp"
#include "CudaOps.hpp"
#include "CudaPool.hpp"
#endif

static const size_t VOCAB_SIZE     = 4096;
static const size_t SEQ_LEN        = 256;
static const size_t EMBED_DIM      = 768;
static const size_t DK             = 768;
static const size_t NUM_HEADS      = 12;
static const size_t NUM_LAYERS     = 10;
// tune this on the actual training instance via BENCHMARK before committing to a long run --
// every intermediate activation (including the batch*heads*seq*seq attention-score tensor per
// layer) is kept as a full fp32 tensor with no recomputation/checkpointing, so real VRAM use at
// seq_len=256 isn't reliably predictable from param count alone
static const size_t BATCH_SIZE     = 16;
static const int    PRETRAIN_STEPS = 100000;
static const int    FINETUNE_STEPS = 15000;
static const int    LOG_EVERY      = 100;
static const int    SAVE_EVERY     = 1000;
static const float  MAX_LR         = 3e-4f;
static const float  MIN_LR_FRAC    = 0.1f;
static const int    WARMUP_STEPS   = 2000;

static float getLR(int step, int phaseSteps)
{
    if (step < WARMUP_STEPS)
    {
        return MAX_LR * (float)(step + 1) / WARMUP_STEPS;
    }
    float t = (float)(step - WARMUP_STEPS) / (phaseSteps - WARMUP_STEPS);
    float cosine = 0.5f * (1.0f + cosf(3.14159265f * t));
    return MAX_LR * (MIN_LR_FRAC + (1.0f - MIN_LR_FRAC) * cosine);
}

int argmax(const float* row, size_t n)
{
    int best = 0;
    for (size_t i = 1; i < n; i++)
    {
        if (row[i] > row[best])
        {
            best = (int)i;
        }
    }
    return best;
}

static void generateSample(TransformerMiniModel& model, DataLoader& loader)
{
    Tensor genInput(2, {1, SEQ_LEN});
    genInput.fillValues(2.0f);

    Tensor dummy(2, {1, SEQ_LEN});
    dummy.fillValues(0.0f);

    std::vector<uint16_t> generated;
    for (size_t s = 0; s < SEQ_LEN - 1; s++)
    {
        Tensor out = model.forward(genInput, dummy);
#ifdef USE_CUDA
        out.toCPU();
#endif
        const float* row = out.data->data() + s * VOCAB_SIZE;
        int next = argmax(row, VOCAB_SIZE);
        generated.push_back(static_cast<uint16_t>(next));
        (*genInput.data)[s + 1] = static_cast<float>(next);
    }
    std::cout << loader.decode(generated) << "\n";
}

static void runPhase(TransformerMiniModel& model, const std::string& name,
                      const std::string& dataFile, int steps)
{
    std::ifstream exists(dataFile);
    if (!exists.good())
    {
        std::cout << "Skipping phase '" << name << "' (missing " << dataFile << ")\n";
        return;
    }

    DataLoader loader(dataFile, "../data/vocab.json", SEQ_LEN, VOCAB_SIZE, BATCH_SIZE);
    std::cout << "\n=== Phase: " << name << " (" << dataFile << ", " << steps << " steps) ===\n";

    auto trainStart = std::chrono::steady_clock::now();

    for (int step = 0; step < steps; step++)
    {
        auto [inp, tgt] = loader.nextBatch();

        model.forward(inp, tgt);
        model.cleanGradients();
        model.backward();
        model.applyGradient(getLR(step, steps));

        if (step % LOG_EVERY == 0 && step > 0)
        {
            float loss = 0.0f;
#ifdef USE_CUDA
            model._lossNode->param.value.toCPU();
#endif
            for (float v : *model._lossNode->param.value.data)
            {
                loss += v;
            }

            auto now     = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - trainStart).count();
            double secPerStep = elapsed / step;
            double etaSec     = secPerStep * (steps - step);
            int etaMin        = static_cast<int>(etaSec) / 60;
            int etaSec2       = static_cast<int>(etaSec) % 60;

            std::cout << "[" << name << "] Step " << step << "/" << steps
                      << "  Loss: " << loss
                      << "  LR: " << getLR(step, steps)
                      << "  ETA: " << etaMin << "m" << etaSec2 << "s\n";
        }

        if (step % SAVE_EVERY == 0 && step > 0)
        {
            model.save("../tinystories.mlt");
            std::cout << "Checkpoint saved at step " << step << "\n";
            std::cout << "Sample: ";
            generateSample(model, loader);
        }
    }

    model.save("../tinystories.mlt");
    std::cout << "Phase '" << name << "' complete, saved to tinystories.mlt\n";
    std::cout << "Sample: ";
    generateSample(model, loader);
}

int main()
{
    srand(42);
#ifdef USE_CUDA
    cudaMatMulInit();
#endif

    TransformerMiniModel model(VOCAB_SIZE, EMBED_DIM, DK, NUM_LAYERS, NUM_HEADS, /*causal=*/true);
    std::cout << "Parameters: " << model.paramCount() << "\n";

    {
        std::ifstream check("../tinystories.mlt");
        if (check.good())
        {
            std::cout << "Resuming from tinystories.mlt\n";
            model.load("../tinystories.mlt");
        }
    }

    // build_vocab.py --dataset combined produces both of these with a shared
    // vocab, so weights carry over correctly between phases with no manual
    // file swapping. Mutually exclusive with the single-dataset fallback
    // below so a stale data/train.bin from an earlier single-dataset run
    // (built with a different, incompatible vocab) never gets touched.
    std::ifstream tsFile("../data/train_tinystories.bin");
    std::ifstream ddFile("../data/train_dailydialog.bin");
    if (tsFile.good() && ddFile.good())
    {
        runPhase(model, "pretrain (tinystories)", "../data/train_tinystories.bin", PRETRAIN_STEPS);
        runPhase(model, "finetune (dailydialog)", "../data/train_dailydialog.bin", FINETUNE_STEPS);
    }
    else
    {
        runPhase(model, "train", "../data/train.bin", PRETRAIN_STEPS);
    }

#ifdef USE_CUDA
    cudaPoolFlush();
    cudaMatMulShutdown();
#endif
    return 0;
}
