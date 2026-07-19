#include "Tensor.hpp"
#include "TransformerMiniModel.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cctype>
#include <stdexcept>
#ifdef USE_CUDA
#include "CudaMatMul.hpp"
#include "CudaPool.hpp"
#endif

static const size_t VOCAB_SIZE = 4096;
static const size_t SEQ_LEN    = 64;
static const size_t EMBED_DIM  = 768;
static const size_t DK         = 768;
static const size_t NUM_LAYERS = 6;

struct Vocab
{
	std::unordered_map<std::string, uint16_t> wordToId;
	std::vector<std::string> idToWord;
	uint16_t padId = 0, unkId = 1, eosId = 2;
};

static Vocab loadVocab(const std::string& path)
{
	std::ifstream f(path);
	if (!f)
	{
		throw std::runtime_error("cannot open vocab file: " + path);
	}
	std::stringstream ss;
	ss << f.rdbuf();
	std::string content = ss.str();

	Vocab vocab;
	size_t pos = 0;
	while (pos < content.size())
	{
		size_t keyStart = content.find('"', pos);
		if (keyStart == std::string::npos)
		{
			break;
		}
		size_t keyEnd = keyStart + 1;
		std::string key;
		while (keyEnd < content.size() && content[keyEnd] != '"')
		{
			if (content[keyEnd] == '\\' && keyEnd + 1 < content.size())
			{
				keyEnd++;
			}
			key += content[keyEnd];
			keyEnd++;
		}
		size_t colon = content.find(':', keyEnd);
		if (colon == std::string::npos)
		{
			break;
		}
		size_t numStart = colon + 1;
		while (numStart < content.size() && !std::isdigit((unsigned char)content[numStart]))
		{
			numStart++;
		}
		size_t numEnd = numStart;
		while (numEnd < content.size() && std::isdigit((unsigned char)content[numEnd]))
		{
			numEnd++;
		}
		if (numEnd == numStart)
		{
			break;
		}
		uint16_t id = (uint16_t)std::stoi(content.substr(numStart, numEnd - numStart));
		vocab.wordToId[key] = id;
		if (vocab.idToWord.size() <= id)
		{
			vocab.idToWord.resize(id + 1);
		}
		vocab.idToWord[id] = key;
		pos = numEnd;
	}
	return vocab;
}

static std::vector<std::string> tokenize(const std::string& text)
{
	std::vector<std::string> tokens;
	std::string lower;
	for (char c : text)
	{
		lower += (char)std::tolower((unsigned char)c);
	}
	std::istringstream iss(lower);
	std::string word;
	while (iss >> word)
	{
		size_t start = 0;
		while (start < word.size() && !(std::isalnum((unsigned char)word[start]) || word[start] == '\''))
		{
			start++;
		}
		size_t end = word.size();
		while (end > start && !(std::isalnum((unsigned char)word[end - 1]) || word[end - 1] == '\''))
		{
			end--;
		}
		if (end > start)
		{
			tokens.push_back(word.substr(start, end - start));
		}
	}
	return tokens;
}

static std::vector<uint16_t> encode(const Vocab& vocab, const std::string& text)
{
	std::vector<uint16_t> ids;
	for (auto& tok : tokenize(text))
	{
		auto it = vocab.wordToId.find(tok);
		ids.push_back(it != vocab.wordToId.end() ? it->second : vocab.unkId);
	}
	return ids;
}

static std::string decode(const Vocab& vocab, const std::vector<uint16_t>& ids)
{
	std::string out;
	for (uint16_t id : ids)
	{
		if (id < vocab.idToWord.size() && !vocab.idToWord[id].empty())
		{
			if (!out.empty())
			{
				out += ' ';
			}
			out += vocab.idToWord[id];
		}
	}
	return out;
}

int main()
{
#ifdef USE_CUDA
	cudaMatMulInit();
#endif

	Vocab vocab = loadVocab("../data/vocab.json");

	TransformerMiniModel model(VOCAB_SIZE, EMBED_DIM, DK, NUM_LAYERS, /*causal=*/true);
	model.load("../tinystories.mlt");
	std::cout << "Loaded model (" << model.paramCount() << " params). Type a message, or 'quit' to exit.\n";

	uint16_t userId = vocab.wordToId.count("user") ? vocab.wordToId["user"] : vocab.eosId;

	while (true)
	{
		std::cout << "\nyou: ";
		std::string line;
		if (!std::getline(std::cin, line))
		{
			break;
		}
		if (line == "quit" || line == "exit")
		{
			break;
		}

		std::vector<uint16_t> context = encode(vocab, "user " + line + " bot");
		if (context.size() > SEQ_LEN - 1)
		{
			context.erase(context.begin(), context.begin() + (context.size() - (SEQ_LEN - 1)));
		}

		Tensor genInput(2, { 1, SEQ_LEN });
		genInput.fillValues((float)vocab.eosId);
		for (size_t i = 0; i < context.size(); i++)
		{
			(*genInput.data)[i] = (float)context[i];
		}

		Tensor dummy(2, { 1, SEQ_LEN });
		dummy.fillValues(0.0f);

		std::vector<uint16_t> generated;
		size_t startS = context.empty() ? 0 : context.size() - 1;

		for (size_t s = startS; s < SEQ_LEN - 1; s++)
		{
			Tensor out = model.forward(genInput, dummy);
#ifdef USE_CUDA
			out.toCPU();
#endif
			const float* row = out.data->data() + s * VOCAB_SIZE;
			int best = 0;
			for (size_t v = 1; v < VOCAB_SIZE; v++)
			{
				if (row[v] > row[best])
				{
					best = (int)v;
				}
			}
			if ((uint16_t)best == vocab.eosId || (uint16_t)best == userId)
			{
				break;
			}
			generated.push_back((uint16_t)best);
			(*genInput.data)[s + 1] = (float)best;
		}

		std::cout << "bot: " << decode(vocab, generated) << "\n";
	}

#ifdef USE_CUDA
	cudaPoolFlush();
	cudaMatMulShutdown();
#endif
	return 0;
}
