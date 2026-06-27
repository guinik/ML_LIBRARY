#pragma once
#include "Tensor.hpp"
#include <cstdint>

Tensor cudaMatMul(const Tensor& A, const Tensor& B, uint16_t mask);
void   cudaMatMulInit();
void   cudaMatMulShutdown();
