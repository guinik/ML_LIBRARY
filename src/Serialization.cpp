#include "Serialization.hpp"
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>

static std::string buildHeader(const std::map<std::string, Tensor>& weights)
{
    std::string json = "{\"version\":1,\"tensors\":{";
    uint64_t offset = 0;
    bool first = true;

    for (const auto& [name, tensor] : weights)
    {
        if (!first)
        {
            json += ',';
        }
        first = false;

        uint64_t byteSize = static_cast<uint64_t>(tensor.data->size()) * sizeof(float);

        json += '"';
        json += name;
        json += "\":{\"shape\":[";
        for (size_t i = 0; i < tensor.shape.size(); i++)
        {
            if (i > 0)
            {
                json += ',';
            }
            json += std::to_string(tensor.shape[i]);
        }
        json += "],\"data_offsets\":[";
        json += std::to_string(offset);
        json += ',';
        json += std::to_string(offset + byteSize);
        json += "]}";

        offset += byteSize;
    }

    json += "}}";
    return json;
}

struct TensorMeta
{
    std::string name;
    std::vector<size_t> shape;
    uint64_t offset_start = 0;
    uint64_t offset_end   = 0;
};

static void skipWs(const std::string& s, size_t& pos)
{
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos])))
    {
        pos++;
    }
}

static std::string parseJsonString(const std::string& s, size_t& pos)
{
    skipWs(s, pos);
    if (pos >= s.size() || s[pos] != '"')
    {
        return {};
    }
    pos++;
    std::string result;
    while (pos < s.size() && s[pos] != '"')
    {
        if (s[pos] == '\\')
        {
            pos++;
            if (pos < s.size())
            {
                result += s[pos++];
            }
        }
        else
        {
            result += s[pos++];
        }
    }
    if (pos < s.size())
    {
        pos++;
    }
    return result;
}

static uint64_t parseJsonUint(const std::string& s, size_t& pos)
{
    skipWs(s, pos);
    uint64_t val = 0;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos])))
    {
        val = val * 10 + static_cast<uint64_t>(s[pos++] - '0');
    }
    return val;
}

static std::vector<uint64_t> parseJsonUintArray(const std::string& s, size_t& pos)
{
    skipWs(s, pos);
    if (pos >= s.size() || s[pos] != '[')
    {
        return {};
    }
    pos++;
    std::vector<uint64_t> result;
    while (pos < s.size())
    {
        skipWs(s, pos);
        if (pos >= s.size())
        {
            break;
        }
        if (s[pos] == ']')
        {
            pos++;
            break;
        }
        if (s[pos] == ',')
        {
            pos++;
            continue;
        }
        result.push_back(parseJsonUint(s, pos));
    }
    return result;
}

static std::vector<TensorMeta> parseHeader(const std::string& json)
{
    std::vector<TensorMeta> result;

    size_t pos = json.find("\"tensors\":");
    if (pos == std::string::npos)
    {
        return result;
    }
    pos += 10;

    skipWs(json, pos);
    if (pos >= json.size() || json[pos] != '{')
    {
        return result;
    }
    pos++;

    while (pos < json.size())
    {
        skipWs(json, pos);
        if (pos >= json.size())
        {
            break;
        }
        if (json[pos] == '}')
        {
            break;
        }
        if (json[pos] == ',')
        {
            pos++;
            continue;
        }

        std::string name = parseJsonString(json, pos);

        skipWs(json, pos);
        if (pos >= json.size() || json[pos] != ':')
        {
            break;
        }
        pos++;

        skipWs(json, pos);
        if (pos >= json.size() || json[pos] != '{')
        {
            break;
        }
        pos++;

        TensorMeta meta;
        meta.name = name;

        while (pos < json.size())
        {
            skipWs(json, pos);
            if (pos >= json.size())
            {
                break;
            }
            if (json[pos] == '}')
            {
                pos++;
                break;
            }
            if (json[pos] == ',')
            {
                pos++;
                continue;
            }

            std::string key = parseJsonString(json, pos);
            skipWs(json, pos);
            if (pos >= json.size() || json[pos] != ':')
            {
                break;
            }
            pos++;

            if (key == "shape")
            {
                auto arr = parseJsonUintArray(json, pos);
                for (auto v : arr)
                {
                    meta.shape.push_back(static_cast<size_t>(v));
                }
            }
            else if (key == "data_offsets")
            {
                auto arr = parseJsonUintArray(json, pos);
                if (arr.size() >= 2)
                {
                    meta.offset_start = arr[0];
                    meta.offset_end = arr[1];
                }
            }
            else
            {
                while (pos < json.size() && json[pos] != ',' && json[pos] != '}')
                {
                    pos++;
                }
            }
        }

        result.push_back(std::move(meta));
    }

    return result;
}

static constexpr uint8_t  MAGIC[8] = {'M','L','T','F',0,0,0,0};
static constexpr uint32_t VERSION  = 1;
static constexpr uint32_t RESERVED = 0;

void save_model(const std::string& path, const std::map<std::string, Tensor>& weights)
{
    std::string header = buildHeader(weights);
    uint64_t headerLen = static_cast<uint64_t>(header.size());

    size_t preData = 24 + header.size();
    size_t padLen  = (64 - (preData % 64)) % 64;

    std::ofstream f(path, std::ios::binary);
    if (!f)
    {
        throw std::runtime_error("save_model: cannot open: " + path);
    }

    f.write(reinterpret_cast<const char*>(MAGIC), 8);
    f.write(reinterpret_cast<const char*>(&VERSION),  4);
    f.write(reinterpret_cast<const char*>(&RESERVED), 4);
    f.write(reinterpret_cast<const char*>(&headerLen), 8);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (size_t i = 0; i < padLen; i++)
    {
        f.put('\0');
    }

    for (const auto& [name, tensor] : weights)
    {
        f.write(reinterpret_cast<const char*>(tensor.data->data()),
                static_cast<std::streamsize>(tensor.data->size() * sizeof(float)));
    }

    if (!f.good())
    {
        throw std::runtime_error("save_model: write error: " + path);
    }
}

std::map<std::string, Tensor> load_model(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        throw std::runtime_error("load_model: cannot open: " + path);
    }

    uint8_t magic[8] = {};
    f.read(reinterpret_cast<char*>(magic), 8);
    if (std::memcmp(magic, MAGIC, 8) != 0)
    {
        throw std::runtime_error("load_model: bad magic (not a .mlt file): " + path);
    }

    uint32_t version = 0, reserved = 0;
    f.read(reinterpret_cast<char*>(&version),  4);
    f.read(reinterpret_cast<char*>(&reserved), 4);
    if (version != VERSION)
    {
        throw std::runtime_error("load_model: unsupported version " + std::to_string(version));
    }

    uint64_t headerLen = 0;
    f.read(reinterpret_cast<char*>(&headerLen), 8);

    std::string header(headerLen, '\0');
    f.read(header.data(), static_cast<std::streamsize>(headerLen));

    size_t preData = 24 + static_cast<size_t>(headerLen);
    size_t padLen  = (64 - (preData % 64)) % 64;
    f.seekg(static_cast<std::streamoff>(padLen), std::ios::cur);

    std::streampos dataStart = f.tellg();

    auto metas = parseHeader(header);

    std::map<std::string, Tensor> result;

    for (const auto& meta : metas)
    {
        size_t totalElements = 1;
        for (size_t s : meta.shape)
        {
            totalElements *= s;
        }

        Tensor t(meta.shape.size(), meta.shape);

        f.seekg(dataStart + static_cast<std::streamoff>(meta.offset_start));
        f.read(reinterpret_cast<char*>(t.data->data()),
               static_cast<std::streamsize>(totalElements * sizeof(float)));

        if (!f.good())
        {
            throw std::runtime_error("load_model: read error for tensor: " + meta.name);
        }

        result[meta.name] = std::move(t);
    }

    return result;
}
