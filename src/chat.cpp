#include "Tensor.hpp"
#include "TransformerMiniModel.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <map>
#include <utility>
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
static const std::string END_OF_WORD_SUFFIX = "</w>";

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

struct Merges
{
	std::map<std::pair<std::string, std::string>, int> rank;
};

static Merges loadMerges(const std::string& path)
{
	std::ifstream f(path);
	if (!f)
	{
		throw std::runtime_error("cannot open merges file: " + path);
	}

	Merges merges;
	std::string line;
	int rank = 0;
	while (std::getline(f, line))
	{
		if (line.empty() || line[0] == '#')
		{
			continue;
		}
		size_t space = line.find(' ');
		if (space == std::string::npos)
		{
			continue;
		}
		std::string a = line.substr(0, space);
		std::string b = line.substr(space + 1);
		merges.rank[{a, b}] = rank++;
	}
	return merges;
}

// Applies the learned BPE merges (lowest rank = highest priority) to a single
// whitespace-delimited word, matching the </w>-suffixed classic-BPE scheme
// that scripts/build_vocab.py trains with.
static std::vector<std::string> bpeEncodeWord(const std::string& word, const Merges& merges)
{
	std::vector<std::string> symbols;
	for (char c : word)
	{
		symbols.push_back(std::string(1, c));
	}
	if (!symbols.empty())
	{
		symbols.back() += END_OF_WORD_SUFFIX;
	}

	while (symbols.size() > 1)
	{
		int bestRank = -1;
		size_t bestIdx = 0;
		for (size_t i = 0; i + 1 < symbols.size(); i++)
		{
			auto it = merges.rank.find({symbols[i], symbols[i + 1]});
			if (it != merges.rank.end() && (bestRank == -1 || it->second < bestRank))
			{
				bestRank = it->second;
				bestIdx = i;
			}
		}
		if (bestRank == -1)
		{
			break;
		}
		symbols[bestIdx] += symbols[bestIdx + 1];
		symbols.erase(symbols.begin() + bestIdx + 1);
	}

	return symbols;
}

static std::vector<uint16_t> encode(const Vocab& vocab, const Merges& merges, const std::string& text)
{
	std::vector<uint16_t> ids;
	std::string lower;
	for (char c : text)
	{
		lower += (char)std::tolower((unsigned char)c);
	}
	std::istringstream iss(lower);
	std::string word;
	while (iss >> word)
	{
		for (const std::string& symbol : bpeEncodeWord(word, merges))
		{
			auto it = vocab.wordToId.find(symbol);
			ids.push_back(it != vocab.wordToId.end() ? it->second : vocab.unkId);
		}
	}
	return ids;
}

static std::string decode(const Vocab& vocab, const std::vector<uint16_t>& ids)
{
	std::string out;
	bool atWordStart = true;
	for (uint16_t id : ids)
	{
		if (id >= vocab.idToWord.size() || vocab.idToWord[id].empty())
		{
			continue;
		}

		const std::string& token = vocab.idToWord[id];
		bool endsWord = token.size() >= END_OF_WORD_SUFFIX.size() &&
			token.compare(token.size() - END_OF_WORD_SUFFIX.size(), END_OF_WORD_SUFFIX.size(), END_OF_WORD_SUFFIX) == 0;

		if (atWordStart && !out.empty())
		{
			out += ' ';
		}
		out += endsWord ? token.substr(0, token.size() - END_OF_WORD_SUFFIX.size()) : token;
		atWordStart = endsWord;
	}
	return out;
}

int main()
{
#ifdef USE_CUDA
	cudaMatMulInit();
#endif

	Vocab vocab = loadVocab("../data/vocab.json");
	Merges merges = loadMerges("../data/merges.txt");

	TransformerMiniModel model(VOCAB_SIZE, EMBED_DIM, DK, NUM_LAYERS, /*causal=*/true);
	model.load("../tinystories.mlt");
	std::cout << "Loaded model (" << model.paramCount() << " params). Type a message, or 'quit' to exit.\n";

	std::vector<uint16_t> userTokens = encode(vocab, merges, "user");
	uint16_t userId = userTokens.size() == 1 ? userTokens[0] : vocab.eosId;

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

		std::vector<uint16_t> context = encode(vocab, merges, "user " + line + " bot");
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
