#include "example/common/tokenizer.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
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

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    CHECK(std::filesystem::exists(filepath)) << "Tokenizer file not found: " << filepath;
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    CHECK_EQ(header.size(), 1024);
    magic_number_ = BytesToType<uint32_t>(header, 0);
    auto version = BytesToType<uint32_t>(header, 4);
    (void)version;
    vocab_size_ = BytesToType<uint32_t>(header, 8);
    CHECK_GT(vocab_size_, 0u);
    CHECK(kEotMap.contains(magic_number_)) << "Unknown tokenizer magic number: " << magic_number_;
    CHECK(kPromptMap.contains(magic_number_)) << "Missing prompt seed for tokenizer magic: " << magic_number_;
    eot_token_ = kEotMap.at(magic_number_);

    token_table_.reserve(vocab_size_);
    for (uint32_t idx = 0; idx < vocab_size_; ++idx) {
        const int length = ifs.get();
        CHECK_NE(length, EOF) << "Unexpected EOF while reading tokenizer vocab length";
        std::string token(std::max(length, 0), '\0');
        if (length > 0) {
            ifs.read(token.data(), length);
            CHECK_EQ(static_cast<int>(ifs.gcount()), length) << "Unexpected EOF while reading tokenizer vocab token";
        }
        token_table_.push_back(token);
    }
    CHECK_EQ(token_table_.size(), vocab_size_) << "Tokenizer vocab truncated";
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    CHECK_LT(token_id, token_table_.size()) << "token_id out of range: " << token_id;
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    uint64_t rng_state = kRngState;
    LOG(INFO) << "start generate text:";
    const uint32_t max_steps = std::min(sequence_length, text_length);
    for (uint32_t t = prompt_len; t < max_steps; ++t) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
        auto logits = model.Forward({x})[0];
        auto logits_cpu = logits->To(Device());
        const auto &logits_dims = logits->Dims();
        CHECK_GE(logits_dims.size(), 3);
        const int64_t batch = logits_dims[0];
        const int64_t seq_len = logits_dims[1];
        const int64_t vocab_size = logits_dims[2];
        CHECK_GT(batch, 0);
        CHECK_GT(seq_len, 0);
        CHECK_GT(vocab_size, 0);

        const float *logits_ptr = static_cast<const float *>(logits_cpu.DataPtr());
        const int64_t target_pos = std::max<int64_t>(
            0, std::min<int64_t>(seq_len - 1, static_cast<int64_t>(t == 0 ? 0 : t - 1)));
        const float *row = logits_ptr + target_pos * vocab_size;

        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t idx = 0; idx < vocab_size; ++idx) {
            max_val = std::max(max_val, row[idx]);
        }

        std::vector<float> probs(static_cast<size_t>(vocab_size));
        float sum = 0.0f;
        for (int64_t idx = 0; idx < vocab_size; ++idx) {
            float val = std::exp(row[idx] - max_val);
            probs[idx] = val;
            sum += val;
        }
        for (float &p : probs) {
            p /= sum;
        }

        const float coin = RandomF32(rng_state);
        const int sampled_token = SampleMult(probs.data(), static_cast<int>(vocab_size), coin);
        x_buff[t] = sampled_token;
        std::cout << Decode(static_cast<uint32_t>(sampled_token));
    }
    std::cout << std::endl;
}
} // namespace infini_train
