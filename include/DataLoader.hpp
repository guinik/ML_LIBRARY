#pragma once
#include "Tensor.hpp"
#include <string>
#include <vector>
#include <cstdint>

struct DataLoader
{
    DataLoader(const std::string& binPath, const std::string& vocabPath, size_t seqLen, size_t vocabSize, size_t batchSize);

    std::pair<Tensor, Tensor> nextBatch();
    std::string decode(const std::vector<uint16_t>& ids) const;

    size_t seqLen;
    size_t vocabSize;
    size_t batchSize;

private:
    std::vector<uint16_t> _data;
    std::vector<std::string> _idxToWord;
    size_t _pos = 0;
};
