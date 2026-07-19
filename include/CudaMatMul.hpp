#pragma once
#include "Tensor.hpp"
#include <cstdint>

Tensor cudaMatMul(const Tensor& A, const Tensor& B, uint16_t mask, const Tensor* bias = nullptr);
void   cudaMatMulInit();
void   cudaMatMulShutdown();
