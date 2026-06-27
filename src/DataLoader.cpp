#include "DataLoader.hpp"
#include <fstream>
#include <stdexcept>
#include <cctype>

static std::vector<std::string> loadVocab(const std::string& path)
{
    std::ifstream f(path);
    if (!f)
    {
        throw std::runtime_error("DataLoader: cannot open vocab: " + path);
    }
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    std::vector<std::string> idxToWord;
    size_t pos = 0;

    while (pos < json.size())
    {
        if (json[pos] != '"')
        {
            pos++;
            continue;
        }
        pos++;
        std::string word;
        while (pos < json.size() && json[pos] != '"')
        {
            word += json[pos++];
        }
        pos++; // skip closing quote

        while (pos < json.size() && json[pos] != ':')
        {
            pos++;
        }
        pos++; // skip colon

        while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos])))
        {
            pos++;
        }

        size_t idx = 0;
        while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos])))
        {
            idx = idx * 10 + static_cast<size_t>(json[pos++] - '0');
        }

        if (idx >= idxToWord.size())
        {
            idxToWord.resize(idx + 1);
        }
        idxToWord[idx] = word;
    }

    return idxToWord;
}

DataLoader::DataLoader(const std::string& binPath, const std::string& vocabPath, size_t inSeqLen, size_t inVocabSize, size_t inBatchSize)
    : seqLen(inSeqLen), vocabSize(inVocabSize), batchSize(inBatchSize)
{
    std::ifstream f(binPath, std::ios::binary | std::ios::ate);
    if (!f)
    {
        throw std::runtime_error("DataLoader: cannot open: " + binPath);
    }
    size_t fileSize = static_cast<size_t>(f.tellg());
    f.seekg(0);
    _data.resize(fileSize / sizeof(uint16_t));
    f.read(reinterpret_cast<char*>(_data.data()), static_cast<std::streamsize>(fileSize));

    _idxToWord = loadVocab(vocabPath);
}

std::pair<Tensor, Tensor> DataLoader::nextBatch()
{
    Tensor input(2, {batchSize, seqLen});
    Tensor target(2, {batchSize, seqLen});

    for (size_t b = 0; b < batchSize; b++)
    {
        if (_pos + seqLen + 1 > _data.size())
        {
            _pos = 0;
        }

        for (size_t i = 0; i < seqLen; i++)
        {
            (*input.data)[b * seqLen + i]  = static_cast<float>(_data[_pos + i]);
            (*target.data)[b * seqLen + i] = static_cast<float>(_data[_pos + i + 1]);
        }

        _pos += seqLen;
    }

    return {input, target};
}

std::string DataLoader::decode(const std::vector<uint16_t>& ids) const
{
    std::string result;
    for (uint16_t id : ids)
    {
        if (id < _idxToWord.size() && !_idxToWord[id].empty())
        {
            if (!result.empty())
            {
                result += ' ';
            }
            result += _idxToWord[id];
        }
    }
    return result;
}
