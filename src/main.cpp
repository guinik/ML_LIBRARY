#include "Tensor.hpp"
#include "MiniModel.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>

float stepFunction(float x)
{
    if (x < 2.0f)  return 0.0f;
    if (x < 4.0f)  return 1.0f;
    if (x < 6.0f)  return 0.3f;
    if (x < 8.0f)  return 0.8f;
    return 0.1f;
}

int main()
{
    srand(4242);
    MiniModel model({ 1, 64, 32, 1 }); // input dim 1, two hidden layers of width 8, output dim 1

    // training data: step function, 0 before x=5, 1 after
    std::vector<float> inputs;
    std::vector<float> outputs;
    for (float x = 0.0f; x <= 10.0f; x += 0.1f)
    {
        inputs.push_back(x);
        outputs.push_back(stepFunction(x));
    }

    float learningRate = 0.005f;
    int epochs = 20000;

    for (int epoch{ 0 }; epoch < epochs; epoch++)
    {
        float totalLoss = 0.0f;
        for (size_t i{ 0 }; i < inputs.size(); i++)
        {
            
            Tensor inputTensor(2, { 1, 1 });
            (*inputTensor.data)[0] = inputs[i];

            Tensor targetTensor(2, { 1, 1 });
            (*targetTensor.data)[0] = outputs[i];

            model.forward(inputTensor, targetTensor);
            model.cleanGradients();
            model.backward();
            if (epoch < 5000)
            {
                model.applyGradient(learningRate);
            }
            else if(epoch < 10000)
            {
                model.applyGradient(learningRate/10);
            }
            else
            {
                model.applyGradient(learningRate/20);
            }

            totalLoss += (*model._lossNode->param.value.data)[0];
        }

        if (epoch % 100 == 0)
            std::cout << "Epoch " << epoch << "  Avg Loss: " << (totalLoss / inputs.size()) << std::endl;
    }

    // see what it learned
    for (float x = 0.0f; x <= 10.0f; x += 0.5f)
    {
        Tensor inputTensor(2, { 1, 1 });
        (*inputTensor.data)[0] = x;
        Tensor dummyTarget(2, { 1, 1 });
        Tensor predicted = model.forward(inputTensor, dummyTarget);
        std::cout << "x = " << x << "  target = " << stepFunction(x)
            << "  predicted = " << (*predicted.data)[0] << std::endl;
    }

    return 0;
}