//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "Kokoro.h"

#include <cstring>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <regex>
#include <variant>

#include "internal/zh_ps.h"

#include <espeak-ng/speak_lib.h>
#include <nlohmann/json.hpp>

namespace compute_flow::bricks::tts {

constexpr int32_t k_brick_major_version = 1;

using json = nlohmann::json;

static std::unordered_map<std::string, int64_t> load_vocab_phonemes_from_file(const std::string& path) {
    std::ifstream f(path);
    json j;
    f >> j;

    std::unordered_map<std::string, int64_t> vocab;
    vocab.reserve(j.size());

    for (auto& [token, id] : j.items()) {
        vocab.emplace(token, id.get<int64_t>());
    }
    return vocab;
}

static void load_voice_styles_from_file(const std::string& voice_file, TensorData& voice_style, int64_t style_sz, int64_t max_phoneme_length) {
    // In Kokoro, the style input is a vector of 256. How do we fill this vector?
    // This tensor is filled through the voice file. The voice file is a matrix of max_phoneme_length (510) rows and style_sz (256) columns ([max_phoneme_length, style_sz])
    // The voice file is stored as consecutive style vectors of 256 elements (style_dim) laid out back-to-back in the file.
    // Therefore, the number of bytes is equal to: max_phoneme_length * style_sz * sizeof(float)
    // So, which vector style should we pick?
    // Row 0 stores the style embedding for a prompt of one-token. row 1 for a prompt of two-token prompts, and so on.
    // Row 509 stores the style embedding for a prompt of 510 tokens
    // Therefore, the number of tokens in the prompt defines the style embedding to take. So, this can change at runtime.

    std::ifstream in(voice_file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open voice file: " + voice_file);
    }
    in.seekg(0, std::ios::end);
    std::streamoff total_bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    // Validate the expected number of bytes
    COMPUTE_FLOW_ASSERT(total_bytes == (style_sz * max_phoneme_length * sizeof(float)));

    in.read(reinterpret_cast<char*>(voice_style.data), static_cast<std::streamsize>(total_bytes));
    if (!in) {
        throw std::runtime_error("Failed to read voice embedding slice from: " + voice_file);
    }
}

static char32_t next_utf8_cp(const std::string& s, size_t& i) {
    unsigned char c0 = static_cast<unsigned char>(s[i]);

    // 1-byte
    if ((c0 & 0x80) == 0) {
        return static_cast<char32_t>(s[i++]);
    }

    // 2-byte
    if ((c0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
        unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        if ((c1 & 0xC0) == 0x80) {
            char32_t cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
            i += 2;
            return cp;
        }
    }

    // 3-byte
    if ((c0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
        unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
        if (((c1 & 0xC0) == 0x80) && ((c2 & 0xC0) == 0x80)) {
            char32_t cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
            i += 3;
            return cp;
        }
    }

    // 4-byte
    if ((c0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
        unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
        unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
        unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
        if (((c1 & 0xC0) == 0x80) && ((c2 & 0xC0) == 0x80) && ((c3 & 0xC0) == 0x80)) {
            char32_t cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) |
                          ((c2 & 0x3F) << 6) | (c3 & 0x3F);
            i += 4;
            return cp;
        }
    }

    // Malformed: consume one byte
    return static_cast<char32_t>(s[i++]);
}

// Mark considered in the phonemizer library _DEFAULT_MARKS = ';:,.!?¡¿—…"«»“”(){}[]'
// Note: The default marks considered in the phonemizer library are the following:
// _DEFAULT_MARKS = ';:,.!?¡¿—…"«»“”(){}[]'
// https://github.com/bootphon/phonemizer/blob/master/phonemizer/punctuation.py#L25
static bool is_punct(char32_t c) {
    switch (c) {
        case U';': case U':': case U',':
        case U'.': case U'!': case U'?':
        case U'—': case U'…': case U'"':
        case U'(': case U')': case U'“':
        case U'{': case U'}': case U'[':
        case U']': case U'¡': case U'¿':
        case U'«': case U'»':
            return true;
        default:
            return false;
    }
}

static std::string normalize_phoneme_whitespace(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    bool in_space = false;
    for (unsigned char ch : input) {
        if (std::isspace(ch)) {
            in_space = true;
            continue;
        }
        if (in_space && !out.empty()) {
            out.push_back(' ');
        }
        in_space = false;
        out.push_back(static_cast<char>(ch));
    }
    return out;
}

static void init_espeak(const std::string& espeak_data_path, const std::string& language) {
    int32_t init_result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, espeak_data_path.c_str(), espeakINITIALIZE_PHONEME_IPA);
    COMPUTE_FLOW_ASSERT(init_result >= 0);

    espeak_ERROR status = espeak_SetVoiceByName(language.c_str());
    COMPUTE_FLOW_ASSERT(status == EE_OK);
}

ValidationResult Kokoro::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "tts");
    COMPUTE_FLOW_VALIDATE(info.name == "Kokoro");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);

    COMPUTE_FLOW_VALIDATE(manifest.num_modules() == 1);

    COMPUTE_FLOW_VALIDATE(manifest.has_module("kokoro"));

    // Validate presence of the model path
    COMPUTE_FLOW_VALIDATE(manifest["kokoro"].model_path != "");

    // Validate the target framework.
    COMPUTE_FLOW_VALIDATE(manifest["kokoro"].framework_type == RunnerType::ONNXRT);

    // Check whether the models only have 1 accelerator type
    // At the moment, only the CPU is supported
    COMPUTE_FLOW_VALIDATE(manifest["kokoro"].acc_config.priorities.size() == 1);

    // Only CPU is supported as accelerator type
    COMPUTE_FLOW_VALIDATE(manifest["kokoro"].acc_config.has_accelerator(AcceleratorType::Cpu));

    // Validate presence of required metadata
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("max_tokens"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("sample_rate"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("speed"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("language"));

    // Validate presence of required asset
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("voice_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("vocab_phonemes_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("jieba_dict_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("hmm_model_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("user_dict_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("idf_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("stop_words_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("espeak_data_path"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("pinyin_dict_path"));

    // TODO: Instead of max_tokens it should be max_phoneme_length as per python implementation: https://github.com/thewh1teagle/kokoro-onnx/blob/main/src/kokoro_onnx/config.py#L5
    return {};
}

void Kokoro::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)manifest;
    (void)profiler;
    // TODO: Implement configuration updates in a separate MR.
}

void Kokoro::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    RunnerType runner_type = manifest["kokoro"].framework_type;

    _runner["kokoro"] = make_runner(runner_type);
    _runner["kokoro"]->setup(manifest["kokoro"], profiler);
}

void Kokoro::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    _max_tokens                = manifest.get_constant<int64_t>("max_tokens");
    const float     speed      = manifest.get_constant<float>("speed");
    const std::string language = manifest.get_constant<std::string>("language");

    const std::string voice_file          = *manifest.get_asset("voice_file");
    const std::string vocab_phonemes_file = *manifest.get_asset("vocab_phonemes_file");

    _language           = kokoro_language_from_string(language);
    _vocab_phonemes_ids = load_vocab_phonemes_from_file(vocab_phonemes_file);

    switch(_language) {
        case Kokoro::KokoroLanguages::Chinese: {
            const std::string jieba_dict_file = *manifest.get_asset("jieba_dict_file");
            const std::string hmm_model_file  = *manifest.get_asset("hmm_model_file");
            const std::string user_dict_file  = *manifest.get_asset("user_dict_file");
            const std::string idf_file        = *manifest.get_asset("idf_file");
            const std::string stop_words_file = *manifest.get_asset("stop_words_file");
            const std::string pinyin_dict_path = *manifest.get_asset("pinyin_dict_path");

            ::zhps::set_paths(jieba_dict_file,
                              hmm_model_file,
                              user_dict_file,
                              idf_file,
                              stop_words_file,
                              pinyin_dict_path);
            break;
        }
        default: {
            const std::string espeak_data_path = *manifest.get_asset("espeak_data_path");
            init_espeak(espeak_data_path, language);
        }
    }

    TensorFormat style_format = _runner["kokoro"]->get_input_format("style");
    COMPUTE_FLOW_ASSERT(style_format.shape[1] == 256);
    style_format.shape[0] = _max_tokens;

    TensorFormat speed_format = _runner["kokoro"]->get_input_format("speed");
    speed_format.shape[0] = 1;

    _internal_memory.allocate("style", style_format);
    _internal_memory.allocate("speed", speed_format);

    auto t_style_in = _internal_memory.get_tensor("style");
    auto t_speed_in = _internal_memory.get_tensor("speed");

    const int64_t style_sz = style_format.shape[1];

    load_voice_styles_from_file(voice_file, t_style_in, style_sz, _max_tokens);

    *((float*)t_speed_in.data) = speed;
}

void Kokoro::warmup_impl(ProfilerView profiler) {
    (void)profiler;
}

void Kokoro::reset_states_impl(ProfilerView profiler) {
    (void)profiler;
    _is_output_ready = false;
}

bool Kokoro::is_generation_complete() const {
    return _is_output_ready;
}

TensorDict Kokoro::run_impl(const TensorDict& input, ProfilerView profiler) {
    _is_output_ready = true;
    // Validate number of input
    // Validate type of input
    // Validate shape


    // TODO: To think about it... ONNX expects a specific number of dimension. Therefore, if it expects [1, LEN] but we pass,
    // [LEN], the runtime will fail.
    // The safest way to get the correct index to query the the lenght of the string is by querying the number of dimensions
    const int64_t num_prompt_dims = input["text_prompt"].format.shape.size();

    const std::string prompt((const char*)(input["text_prompt"].data), input["text_prompt"].format.shape[num_prompt_dims - 1]);

    // It is not a custom op because it does not work on TensorData but rather on std::string
    std::string phonemes;

    switch(_language) {
        case Kokoro::KokoroLanguages::Chinese: {
            convert_prompt_to_phonemes_zh(prompt, phonemes);
            break;
        }
        default: {
            convert_prompt_to_phonemes_generic(prompt, phonemes);
            break;
        }
    }

    // Since the prompt has a dynamic shape, we can only allocate the phonemes and token_ids intermediate tensors
    // in the run method. However, we must release their memory once used
    // Since phonemes is a std::string, it will be safely released when this method returns.
    // However, the token_ids must be explicitly freed as we allocated with the memory planner
    TensorData t_phonemes;
    t_phonemes.data         = (void*)phonemes.data();
    t_phonemes.format.dt    = DataType::String;
    t_phonemes.format.shape = {1, (int64_t)phonemes.size()};

    TensorFormat ids_format;
    ids_format.dt    = DataType::I64;
    ids_format.shape = {1, custom_op_get_num_token_ids(t_phonemes)};

    _internal_memory.allocate("ids", ids_format);

    auto t_ids_in   = _internal_memory.get_tensor("ids");
    auto t_style_in = _internal_memory.get_tensor("style");
    auto t_speed_in = _internal_memory.get_tensor("speed");

    custom_op_phonemes_to_token_ids(t_phonemes, t_ids_in);

    const int64_t row_style_idx = t_ids_in.format.shape[1];
    const int64_t style_sz      = t_style_in.format.shape[1];

    // Check whether the token_count is greater than max_phoneme_length. It is the maximum allowed in Kokoro
    COMPUTE_FLOW_ASSERT(row_style_idx <= _max_tokens);

    // Point the style to the correct row
    uint8_t* start_style_ptr = (uint8_t*)t_style_in.data;
    t_style_in.data = start_style_ptr + row_style_idx * style_sz * sizeof(float);
    t_style_in.format.shape[0] = 1;

    _runner["kokoro"]->set_input("ids", t_ids_in);
    _runner["kokoro"]->set_input("style", t_style_in);
    _runner["kokoro"]->set_input("speed", t_speed_in);

    _runner["kokoro"]->run();

    auto t_out_waveform = _runner["kokoro"]->get_output("waveform");

    TensorDict output;

    output.declare_tensor("left_channel");
    output.declare_tensor("right_channel");

    const int64_t num_audio_samples = t_out_waveform.format.shape[0];

    output["left_channel"].data         = (void*)t_out_waveform.data;
    output["left_channel"].format.dt    = DataType::F32;
    output["left_channel"].format.shape = {static_cast<int64_t>(num_audio_samples)};

    output["right_channel"].data         = (void*)t_out_waveform.data;
    output["right_channel"].format.dt    = DataType::F32;
    output["right_channel"].format.shape = {static_cast<int64_t>(num_audio_samples)};

    // We must release this intermediate tensor because at the next call
    // we may have a different shape
    _internal_memory.release("ids");

    return output;
}

int64_t Kokoro::custom_op_get_num_token_ids(const TensorData& phonemes) {
    const int64_t num_phonemes = phonemes.format.shape[1];

    const uint8_t* phonemes_ptr = (const unsigned char*)phonemes.data;

    int64_t num_token_ids = 2; // BOS and EOS

    int64_t i = 0;
    while (i < num_phonemes) {
        uint8_t c = phonemes_ptr[i];
        int64_t num_bytes_utf8 = 1;
        if ((c & 0x80) == 0) {
            num_bytes_utf8 = 1;
        } else if ((c & 0xE0) == 0xC0) {
            num_bytes_utf8 = 2;
        } else if ((c & 0xF0) == 0xE0) {
            num_bytes_utf8 = 3;
        } else if ((c & 0xF8) == 0xF0) {
            num_bytes_utf8 = 4;
        } else {
            // Invalid UTF-8 start byte, skip
            ++i;
            continue;
        }
        const std::string phoneme(reinterpret_cast<const char*>(&phonemes_ptr[i]), num_bytes_utf8);

        auto it = _vocab_phonemes_ids.find(phoneme);
        if (it != _vocab_phonemes_ids.end()) {
            num_token_ids++;
        }
        i += num_bytes_utf8;
    }

    return num_token_ids;
}

// Each phonemes is a UTF-8 string, which can represent any
// Unicode character using a variable number of bytes from one to four
// Therefore, when we iterate over the phonemes, we must identify the number of bytes used to represent the UTF-8 string first
// Then, we copy the string and we use the vocabulary to tokenize it
// Since not all UTF-8 strings can be found in the vocabulary, the first step is to filter out those phonemes that cannot be tokenized
// Therefore, filter_to_vocab returns a new string containing only those UTF-8 characters that exist in vocab
void Kokoro::custom_op_phonemes_to_token_ids(const TensorData& phonemes, TensorData& token_ids) {
    const int64_t num_phonemes = phonemes.format.shape[1];

    const uint8_t* phonemes_ptr = (const unsigned char*)phonemes.data;
    int64_t* ids_ptr            = (int64_t*)token_ids.data;

    // Add BOS. TODO: Check if it is correct
    *ids_ptr++ = 0;

    int64_t i = 0;
    while (i < num_phonemes) {
        uint8_t c = phonemes_ptr[i];
        int64_t num_bytes_utf8 = 1;
        if ((c & 0x80) == 0) {
            num_bytes_utf8 = 1;
        } else if ((c & 0xE0) == 0xC0) {
            num_bytes_utf8 = 2;
        } else if ((c & 0xF0) == 0xE0) {
            num_bytes_utf8 = 3;
        } else if ((c & 0xF8) == 0xF0) {
            num_bytes_utf8 = 4;
        } else {
            // Invalid UTF-8 start byte, skip
            ++i;
            continue;
        }
        const std::string phoneme(reinterpret_cast<const char*>(&phonemes_ptr[i]), num_bytes_utf8);

        auto it = _vocab_phonemes_ids.find(phoneme);
        if (it != _vocab_phonemes_ids.end()) {
            *ids_ptr++ = it->second;
        }
        i += num_bytes_utf8;
    }

    // Add EOS
    *ids_ptr++ = 0;
}

void Kokoro::convert_prompt_to_phonemes_generic(const std::string& text_prompt, std::string& phonemes) {

    // Convert a UTF-8 text prompt to IPA phonemes using eSpeak, while preserving punctuation
    // and normalizing whitespace to match the Python reference output.
    // Options mirrored from the Python phonemizer:
    // - with_stress: enabled (IPA output includes stress marks).
    // - preserve_punctuation: enabled (punctuation is kept in the output).
    // - language_switch: remove-flags (implicit; we do not keep language flags).
    // Preserve punctuation

    // With stress

    phonemes.clear();
    std::string buffer;

    auto flush_buffer = [&]() {
        if (buffer.empty()) return;

        const void* ptr = buffer.c_str();
        const char* prev = static_cast<const char*>(ptr);

        while (ptr != nullptr) {
            const char* chunk =
                espeak_TextToPhonemes(&ptr, espeakCHARS_UTF8, espeakPHONEMES_IPA);
            if (!chunk || *chunk == '\0') {
                break;
            }

            phonemes.append(chunk);

            if (ptr == nullptr) {
                break;
            }

            const char* cur = static_cast<const char*>(ptr);
            if (cur == prev) {
                if (cur && *cur != '\0') {
                    ++cur;
                    ptr = static_cast<const void*>(cur);
                } else {
                    break;
                }
            }
            prev = cur;
        }

        buffer.clear();
    };

    // Walk prompt as UTF-8 code points, but keep byte ranges for punctuation/space.
    for (size_t i = 0; i < text_prompt.size();) {
        size_t start = i;
        char32_t cp = next_utf8_cp(text_prompt, i); // advances i
        size_t end = i;                             // bytes [start, end)

        bool is_space_ascii = (cp <= 0x7F) &&
                              std::isspace(static_cast<unsigned char>(text_prompt[start]));

        if (is_punct(cp)) {
            flush_buffer();
            phonemes.append(text_prompt, start, end - start); // append punctuation bytes
            // Preserve any following ASCII whitespace after punctuation.
            size_t j = i;
            bool had_space = false;
            while (j < text_prompt.size()) {
                unsigned char c = static_cast<unsigned char>(text_prompt[j]);
                if (c <= 0x7F && std::isspace(c)) {
                    had_space = true;
                    ++j;
                    continue;
                }
                break;
            }
            if (had_space) {
                phonemes.push_back(' ');
                i = j;
            }
        } else if (is_space_ascii) {
            // Keep whitespace with words so eSpeak sees multi-word phrases (avoids per-word stress).
            buffer.append(text_prompt, start, end - start);
        } else {
            buffer.append(text_prompt, start, end - start);   // accumulate UTF-8 bytes
        }
    }

    flush_buffer();
    phonemes = normalize_phoneme_whitespace(phonemes);
}

void Kokoro::convert_prompt_to_phonemes_zh(const std::string& prompt, std::string& ps) {

    COMPUTE_FLOW_ASSERT(_language == Kokoro::KokoroLanguages::Chinese);
    // separate the text into Chinese and engilish parts
    std::vector<std::string> segments;

    // Captures either English text or non-English text
    std::regex pattern(R"(([A-Za-z \'-]*[A-Za-z][A-Za-z \'-]*)|([^A-Za-z]+))");
    std::sregex_iterator iter(prompt.begin(), prompt.end(), pattern);
    std::sregex_iterator end;

    std::string unk = "❓";

    for (; iter != end; ++iter) {
        std::smatch match = *iter;
        std::string zh = match[2].str();
        // Trim whitespace
        zh.erase(0, zh.find_first_not_of(" \t\n\r\f\v"));
        zh.erase(zh.find_last_not_of(" \t\n\r\f\v") + 1);
        ps = ::zhps::process_zh_tokens_to_phonemes(zh, true, unk);
        segments.push_back(ps);
    }

    ps.clear();
    for (const auto& seg : segments) {
        ps += seg;
    }
}

Kokoro::KokoroLanguages Kokoro::kokoro_language_from_string(std::string language_str) {
    std::transform(language_str.begin(), language_str.end(),
                   language_str.begin(), ::tolower);

    static const std::unordered_map<std::string, KokoroLanguages> language_map = {
        {"cmn", KokoroLanguages::Chinese},
        {"en", KokoroLanguages::English},
        {"en-us", KokoroLanguages::English},
        {"en-gb", KokoroLanguages::English},
        {"es", KokoroLanguages::Spanish},
        {"fr-fr", KokoroLanguages::French},
        {"it", KokoroLanguages::Italian}
    };

    auto it = language_map.find(language_str);

    COMPUTE_FLOW_ASSERT(it != language_map.end());

    return it->second;
}

}  // namespace compute_flow::bricks::tts
