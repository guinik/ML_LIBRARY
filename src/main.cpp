#include "Matrix.hpp"
#include "MiniModel.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>

int main()
{
    srand(42);
    MiniModel model({ 1, 32, 16, 1 }); // input dim 1, two hidden layers of width 8, output dim 1

    // training data: step function, 0 before x=5, 1 after
    std::vector<float> inputs;
    std::vector<float> outputs;
    for (float x = 0.0f; x <= 10.0f; x += 0.1f)
    {
        inputs.push_back(x);
        outputs.push_back(x < 5.0f ? 0.0f : 1.0f);
    }

    float learningRate = 0.005f;
    int epochs = 10000;

    for (int epoch{ 0 }; epoch < epochs; epoch++)
    {
        float totalLoss = 0.0f;
        for (size_t i{ 0 }; i < inputs.size(); i++)
        {
            Matrix inputMatrix(2, { 1, 1 });
            (*inputMatrix.data)[0] = inputs[i];

            Matrix targetMatrix(2, { 1, 1 });
            (*targetMatrix.data)[0] = outputs[i];

            model.forward(inputMatrix, targetMatrix);
            model.cleanGradients();
            model.backward();
            model.applyGradient(learningRate);

            totalLoss += (*model._lossNode->param.value.data)[0];
        }

        if (epoch % 100 == 0)
            std::cout << "Epoch " << epoch << "  Avg Loss: " << (totalLoss / inputs.size()) << std::endl;
    }

    // see what it learned
    for (float x = 0.0f; x <= 10.0f; x += 1.0f)
    {
        Matrix inputMatrix(2, { 1, 1 });
        (*inputMatrix.data)[0] = x;
        Matrix dummyTarget(2, { 1, 1 }); // unused by the actual prediction, just needed to call forward()

        Matrix predicted = model.forward(inputMatrix, dummyTarget);
        std::cout << "x = " << x << "  predicted = " << (*predicted.data)[0] << std::endl;
    }

    return 0;
}