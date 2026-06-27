#include "Tensor.hpp"
#include "TransformerMiniModel.hpp"
#include "DataLoader.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>
#ifdef USE_CUDA
#include "CudaMatMul.hpp"
#endif

static const size_t VOCAB_SIZE  = 4096;
static const size_t SEQ_LEN     = 32;
static const size_t EMBED_DIM   = 128;
static const size_t DK          = 128;
static const size_t NUM_LAYERS  = 4;
static const size_t BATCH_SIZE  = 8;
static const int    STEPS       = 100000;
static const int    LOG_EVERY   = 100;
static const int    SAVE_EVERY  = 1000;

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

int main()
{
    srand(42);
#ifdef USE_CUDA
    cudaMatMulInit();
#endif

    DataLoader loader("../data/train.bin", "../data/vocab.json", SEQ_LEN, VOCAB_SIZE, BATCH_SIZE);

    TransformerMiniModel model(VOCAB_SIZE, EMBED_DIM, DK, NUM_LAYERS, /*causal=*/true);

    std::cout << "Parameters: " << model.paramCount() << "\n";

    {
        std::ifstream check("../tinystories.mlt");
        if (check.good())
        {
            std::cout << "Resuming from tinystories.mlt\n";
            model.load("../tinystories.mlt");
        }
    }

    std::cout << "Training on TinyStories (" << STEPS << " steps, batch=" << BATCH_SIZE << ")...\n";

    auto trainStart = std::chrono::steady_clock::now();

    for (int step = 0; step < STEPS; step++)
    {
        auto [inp, tgt] = loader.nextBatch();

        model.forward(inp, tgt);
        model.cleanGradients();
        model.backward();
        model.applyGradient(0.001f);

        if (step % LOG_EVERY == 0 && step > 0)
        {
            float loss = 0.0f;
            for (float v : *model._lossNode->param.value.data)
            {
                loss += v;
            }

            auto now     = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - trainStart).count();
            double secPerStep = elapsed / step;
            double etaSec     = secPerStep * (STEPS - step);
            int etaMin        = static_cast<int>(etaSec) / 60;
            int etaSec2       = static_cast<int>(etaSec) % 60;

            std::cout << "Step " << step << "/" << STEPS
                      << "  Loss: " << loss
                      << "  ETA: " << etaMin << "m" << etaSec2 << "s\n";
        }

        if (step % SAVE_EVERY == 0 && step > 0)
        {
            model.save("../tinystories.mlt");
            std::cout << "Checkpoint saved at step " << step << "\n";
        }
    }

    model.save("../tinystories.mlt");
    std::cout << "\nSaved to tinystories.mlt\n";

    std::cout << "\n--- Generation ---\n";

    Tensor genInput(2, {1, SEQ_LEN});
    genInput.fillValues(2.0f); // <EOS> as seed

    Tensor dummy(2, {1, SEQ_LEN});
    dummy.fillValues(0.0f);

    std::vector<uint16_t> generated;
    for (size_t step = 0; step < SEQ_LEN - 1; step++)
    {
        Tensor out = model.forward(genInput, dummy);
        const float* row = out.data->data() + step * VOCAB_SIZE;
        int next = argmax(row, VOCAB_SIZE);
        generated.push_back(static_cast<uint16_t>(next));
        (*genInput.data)[step + 1] = static_cast<float>(next);
    }

    std::cout << loader.decode(generated) << "\n";

#ifdef USE_CUDA
    cudaMatMulShutdown();
#endif
    return 0;
}
