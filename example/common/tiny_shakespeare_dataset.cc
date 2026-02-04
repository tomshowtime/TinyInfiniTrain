#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    CHECK(std::filesystem::exists(path)) << "Dataset file not found: " << path;
    CHECK_GT(sequence_length, 0) << "sequence_length must be positive";

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open dataset file: " << path;

    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_EQ(header.size(), 1024);
    const uint32_t magic = BytesToType<uint32_t>(header, 0);
    const uint32_t version = BytesToType<uint32_t>(header, 4);
    const uint32_t num_tokens = BytesToType<uint32_t>(header, 8);
    (void)version;
    CHECK(kTypeMap.contains(magic)) << "Unsupported dataset magic: " << magic;

    TinyShakespeareFile file;
    file.type = kTypeMap.at(magic);
    const size_t token_size = kTypeToSize.at(file.type);
    CHECK_GT(num_tokens, 0u) << "Dataset is empty";

    const uint64_t total_sequences = static_cast<uint64_t>(num_tokens) / sequence_length;
    CHECK_GE(total_sequences, 2ULL) << "Dataset too small for the requested sequence length";
    const size_t usable_tokens = static_cast<size_t>(total_sequences * sequence_length);

    std::vector<uint8_t> raw_data(static_cast<size_t>(num_tokens) * token_size);
    ifs.read(reinterpret_cast<char *>(raw_data.data()), raw_data.size());
    CHECK_EQ(static_cast<size_t>(ifs.gcount()), raw_data.size()) << "Failed to read dataset payload";

    file.dims = {static_cast<int64_t>(total_sequences), static_cast<int64_t>(sequence_length)};
    file.tensor = infini_train::Tensor(file.dims, DataType::kINT64);
    auto *dst = static_cast<int64_t *>(file.tensor.DataPtr());

    for (size_t idx = 0; idx < usable_tokens; ++idx) {
        const size_t byte_offset = idx * token_size;
        if (file.type == TinyShakespeareType::kUINT16) {
            dst[idx] = static_cast<int64_t>(BytesToType<uint16_t>(raw_data, byte_offset));
        } else {
            dst[idx] = static_cast<int64_t>(BytesToType<uint32_t>(raw_data, byte_offset));
        }
    }

    return file;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      num_samples_(text_file_.dims.empty() ? 0 : static_cast<size_t>(text_file_.dims[0] - 1)) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
    CHECK_GE(text_file_.dims.size(), 2);
    CHECK_EQ(text_file_.dims[1], static_cast<int64_t>(sequence_length_))
        << "Sequence length mismatch between header and request";
    CHECK_GT(num_samples_, 0u) << "No usable samples in dataset";
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
