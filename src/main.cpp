#include "Tensor.hpp"
#include "TransformerMiniModel.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>

static const size_t VOCAB_SIZE = 6;
static const size_t SEQ_LEN   = 4;

Tensor oneHot(const std::vector<int>& tokens)
{
    Tensor t(2, {SEQ_LEN, VOCAB_SIZE});
    t.fillValues(0.0f);
    for (size_t i = 0; i < tokens.size(); i++)
    {
        (*t.data)[i * VOCAB_SIZE + tokens[i]] = 1.0f;
    }
    return t;
}

int argmax(const float* row, size_t n)
{
    int best = 0;
    for (size_t i = 1; i < n; i++)
    {
        if (row[i] > row[best]) best = (int)i;
    }
    return best;
}

int main()
{
    srand(42);

    const char* tokenName[] = {"A","B","C","D","E","F"};

    // Two independent 3-cycles: {A,B,C} and {D,E,F}.
    std::vector<std::vector<int>> inputs = {
        {0,1,2,0},  {1,2,0,1},  {2,0,1,2},
        {3,4,5,3},  {4,5,3,4},  {5,3,4,5},
    };
    std::vector<std::vector<int>> targets = {
        {1,2,0,1},  {2,0,1,2},  {0,1,2,0},
        {4,5,3,4},  {5,3,4,5},  {3,4,5,3},
    };

    TransformerMiniModel model(VOCAB_SIZE, 16, 16, 2, /*causal=*/true);

    const int epochs = 5000;
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float totalLoss = 0.0f;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            Tensor inp = oneHot(inputs[i]);
            Tensor tgt = oneHot(targets[i]);

            model.forward(inp, tgt);
            model.cleanGradients();
            model.backward();

            float lr = (epoch < 2000) ? 0.005f : 0.001f;
            model.applyGradient(lr);

            for (float v : *model._lossNode->param.value.data)
            {
                totalLoss += v;
            }
        }

        if (epoch % 500 == 0)
        {
            std::cout << "Epoch " << epoch << "  Loss: " << totalLoss << "\n";
        }
    }

    std::cout << "\n--- Autoregressive generation ---\n";
    std::vector<int> seeds = {0, 3};
    for (int seed : seeds)
    {
        std::vector<int> generated = {seed};
        Tensor genInput(2, {SEQ_LEN, VOCAB_SIZE});
        genInput.fillValues(0.0f);
        (*genInput.data)[seed] = 1.0f;

        Tensor dummy(2, {SEQ_LEN, VOCAB_SIZE});
        dummy.fillValues(0.0f);

        for (size_t step = 0; step < SEQ_LEN - 1; step++)
        {
            Tensor out = model.forward(genInput, dummy);
            const float* row = out.data->data() + step * VOCAB_SIZE;
            int next = argmax(row, VOCAB_SIZE);
            generated.push_back(next);
            (*genInput.data)[(step + 1) * VOCAB_SIZE + next] = 1.0f;
        }

        std::cout << "Seed " << tokenName[seed] << " -> ";
        for (int t : generated)
        {
            std::cout << tokenName[t] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
