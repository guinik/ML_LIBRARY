#pragma once
#include "Tensor.hpp"
#include <map>
#include <string>
#include <stdexcept>

// .mlt file layout:
//   [8  bytes] magic: 'M','L','T','F',0,0,0,0
//   [4  bytes] format version (uint32_t, little-endian) — currently 1
//   [4  bytes] reserved (zeros)
//   [8  bytes] header length (uint64_t, little-endian)
//   [N  bytes] JSON header (UTF-8)
//   [pad bytes] zeros to reach next 64-byte boundary
//   [data     ] packed float32 tensors, in header order

void save_model(const std::string& path, const std::map<std::string, Tensor>& weights);
std::map<std::string, Tensor> load_model(const std::string& path);
