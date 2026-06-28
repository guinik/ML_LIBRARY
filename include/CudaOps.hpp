#pragma once
#include "Tensor.hpp"
#include <vector>
#include <utility>

Tensor cudaAdd(const Tensor& A, const Tensor& B);
Tensor cudaSubtract(const Tensor& A, const Tensor& B);
Tensor cudaScale(const Tensor& A, float factor);
Tensor cudaElemMul(const Tensor& A, const Tensor& B);
Tensor cudaRelu(const Tensor& A);
Tensor cudaSigmoid(const Tensor& A);
Tensor cudaSquare(const Tensor& A);
Tensor cudaSoftmax(const Tensor& A);
Tensor cudaLayerNorm(const Tensor& A, float eps);
Tensor cudaCausalMask(const Tensor& A);
Tensor cudaEmbeddingForward(const Tensor& ids, const Tensor& weights);
Tensor cudaCrossEntropyForward(const Tensor& logits, const Tensor& targets);

Tensor cudaUnbroadcast(const Tensor& grad, const std::vector<size_t>& targetShape);
Tensor cudaReluBackward(const Tensor& input, const Tensor& gradOutput);
Tensor cudaSigmoidBackward(const Tensor& output, const Tensor& gradOutput);
Tensor cudaSquareBackward(const Tensor& input, const Tensor& gradOutput);
Tensor cudaSoftmaxBackward(const Tensor& output, const Tensor& gradOutput);
Tensor cudaLayerNormBackward(const Tensor& x, const Tensor& xhat, const Tensor& gradOutput, float eps);
Tensor cudaEmbeddingBackwardWeights(const Tensor& ids, const Tensor& weightsRef, const Tensor& gradOutput);
std::pair<Tensor, Tensor> cudaCrossEntropyBackward(const Tensor& logits, const Tensor& targets, const Tensor& gradOutput);
