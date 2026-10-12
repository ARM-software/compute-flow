//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "Qwen3TTS.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <npy.hpp>

namespace compute_flow::bricks::tts {

namespace {

using json = nlohmann::json;

constexpr int32_t k_brick_major_version = 6;
constexpr char kIclTalkerPrefixStateName[] = "qwen3_tts_base_icl_talker_prefix";
constexpr char kIclSpeechDecoderStateName[] = "qwen3_tts_base_icl_ref_codes";

static void load_embeddings_from_file(const std::string& path, TensorData& speaker_embedding) {
    // Load embeddings from a .npy file
    npy::npy_data d = npy::read_npy<float>(path);
    COMPUTE_FLOW_ASSERT(d.shape[0] == speaker_embedding.format.shape[0]);
    COMPUTE_FLOW_ASSERT(d.shape[1] == speaker_embedding.format.shape[1]);

    memcpy(speaker_embedding.data, d.data.data(), speaker_embedding.format.get_nbytes());
}

static uint64_t hash_file_fnv1a(const std::string& path) {
    constexpr uint64_t k_fnv_offset = 14695981039346656037ull;
    constexpr uint64_t k_fnv_prime  = 1099511628211ull;

    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    uint64_t hash = k_fnv_offset;
    char buffer[4096];

    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
        std::streamsize num_bytes_read = file.gcount();

        for (std::streamsize i = 0; i < num_bytes_read; ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= k_fnv_prime;
        }
    }
    return hash;
}

static std::string read_text_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if(!file) {
        throw std::runtime_error("Could not open text file: " + path);
    }

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());
}

int64_t ceil_div(int64_t num, int64_t den) {
    COMPUTE_FLOW_ASSERT(den > 0);
    return (num + den - 1) / den;
}

std::string find_file_in_directory(const std::filesystem::path& directory,
                                   const std::string& text) {

    if (!std::filesystem::exists(directory)) {
        throw std::runtime_error("Directory does not exist: " + directory.string());
    }

    if (!std::filesystem::is_directory(directory)) {
        throw std::runtime_error("Path is not a directory: " + directory.string());
    }

    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string filename = entry.path().filename().string();

        if (filename.find(text) != std::string::npos) {
            return entry.path().string();
        }
    }
    // If no matching file is found, return an empty string
    return "";
}

static void read_wav(const std::string& path, std::vector<float>& mono_raw_wav, int32_t expected_sample_rate) {
    constexpr uint16_t wave_format_pcm             = 0x0001;
    constexpr uint16_t wave_format_ieee_float      = 0x0003;
    constexpr uint16_t wave_format_extensible      = 0xFFFE;
    constexpr uint16_t wave_format_num_ch          = 1;
    constexpr uint16_t wave_format_bits_per_sample = 32;

    std::ifstream input_stream(path, std::ios::binary);

    COMPUTE_FLOW_ASSERT(input_stream);

    char riff[4], wave[4], fmt[4];
    uint32_t riff_size, fmt_chunk_sz;
    uint16_t audio_format, audio_num_channels;
    uint32_t audio_sr, byte_rate, data_chunk_sz;
    uint16_t block_align, audio_bits_per_sample;

    std::vector<float> data_chunk;

    std::streampos riff_base = input_stream.tellg();
    input_stream.read(riff, 4);
    input_stream.read(reinterpret_cast<char*>(&riff_size), 4);
    COMPUTE_FLOW_ASSERT(bool(riff_size));
    input_stream.read(wave, 4);
    if(std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        fprintf(stderr,
        "Unsupported WAV format (need %d Hz, mono, 32-bit float), use this ffmpeg command to convert your file:\n"
        "ffmpeg -i input_audio.mp3 -ar %d -ac 1 -c:a pcm_f32le -f wav output.wav\n\n", expected_sample_rate, expected_sample_rate);
        exit(EXIT_FAILURE);
    }

    input_stream.read(fmt, 4);
    COMPUTE_FLOW_ASSERT(std::string(fmt, 4) == "fmt ");
    input_stream.read(reinterpret_cast<char*>(&fmt_chunk_sz), 4);
    COMPUTE_FLOW_ASSERT(fmt_chunk_sz >= 16);

    input_stream.read(reinterpret_cast<char*>(&audio_format), 2);
    input_stream.read(reinterpret_cast<char*>(&audio_num_channels), 2);
    input_stream.read(reinterpret_cast<char*>(&audio_sr), 4);
    input_stream.read(reinterpret_cast<char*>(&byte_rate), 4);
    input_stream.read(reinterpret_cast<char*>(&block_align), 2);
    input_stream.read(reinterpret_cast<char*>(&audio_bits_per_sample), 2);

    if (!(audio_format == wave_format_ieee_float || audio_format == wave_format_pcm || audio_format == wave_format_extensible) ||
        audio_num_channels != wave_format_num_ch ||
        audio_sr != expected_sample_rate ||
        audio_bits_per_sample != wave_format_bits_per_sample) {
        fprintf(stderr,
        "Unsupported WAV format (need %d Hz, mono, 32-bit float), use this ffmpeg command to convert your file:\n"
        "ffmpeg -i input_audio.mp3 -ar %d -ac 1 -c:a pcm_f32le -f wav output.wav\n\n", expected_sample_rate, expected_sample_rate);
        exit(EXIT_FAILURE);
    }

    // Skip any extension bytes in the fmt chunk
    if (fmt_chunk_sz > 16) {
        input_stream.seekg(static_cast<std::streamoff>(fmt_chunk_sz - 16), std::ios::cur);
        COMPUTE_FLOW_ASSERT(bool(input_stream));
    }

    // Compute absolute end of this RIFF chunk: 8 (header) + riff_size bytes
    const std::streampos riff_end = riff_base + static_cast<std::streamoff>(8ull + riff_size);

    // Now we scan for the "data" chunk
    char chunk_id[4];
    uint32_t chunk_size = 0;
    for (;;) {
        std::streampos here = input_stream.tellg();
        COMPUTE_FLOW_ASSERT(here != std::streampos(-1));

        input_stream.read(chunk_id, 4);
        COMPUTE_FLOW_ASSERT(input_stream);
        input_stream.read(reinterpret_cast<char*>(&chunk_size), 4);
        COMPUTE_FLOW_ASSERT(input_stream);

        if (std::string(chunk_id, 4) == "data") {
            data_chunk_sz = chunk_size;
            // Ensure the whole chunk fits in RIFF
            COMPUTE_FLOW_ASSERT(input_stream.tellg() + static_cast<std::streamoff>(data_chunk_sz) <= riff_end);
            break;
        }
        // word-align skip (chunks are padded to even sizes)
        input_stream.seekg(static_cast<std::streamoff>(chunk_size + (chunk_size & 1)), std::ios::cur);
        COMPUTE_FLOW_ASSERT(bool(input_stream));
    }

    const uint32_t num_frames = data_chunk_sz / block_align;
    const uint32_t total_samples = num_frames * audio_num_channels;

    mono_raw_wav.resize(total_samples);
    input_stream.read(reinterpret_cast<char*>(mono_raw_wav.data()),
                      static_cast<std::streamsize>(data_chunk_sz));
}

std::string tensor_to_string(const TensorData& t) {
    if (t.format.dt != DataType::String) {
        throw std::invalid_argument("Qwen3TTS: expected DataType::String for text tensor");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument("Qwen3TTS: text tensor has null data");
    }
    if (t.format.shape.empty()) {
        throw std::invalid_argument("Qwen3TTS: text tensor has empty shape");
    }

    int64_t nbytes = 0;
    if (t.format.shape.size() == 1) {
        nbytes = t.format.shape[0];
    } else if (t.format.shape.size() == 2) {
        nbytes = t.format.shape[1];
    } else {
        throw std::invalid_argument("Qwen3TTS: text tensor has unsupported shape rank");
    }

    if (nbytes < 0) {
        throw std::invalid_argument("Qwen3TTS: text tensor has negative size");
    }
    return std::string(static_cast<const char*>(t.data), static_cast<size_t>(nbytes));
}

char32_t next_utf8_cp(const std::string& s, size_t& i) {
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

bool is_punct(char32_t c) {
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

void append_newline_if_missing(std::string& prompt) {
    if (prompt.empty() || prompt.back() != '\n') {
        prompt.push_back('\n');
    }
}

std::string ensure_prompt_ends_with_punct(const std::string& text) {
    std::string prompt = text;
    size_t end = prompt.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(prompt[end - 1]))) {
        --end;
    }

    if (end == 0) {
        append_newline_if_missing(prompt);
        return prompt;
    }

    char32_t last_cp = 0;
    size_t i = 0;
    while (i < end) {
        last_cp = next_utf8_cp(prompt, i);
    }

    if (!is_punct(last_cp)) {
        prompt.erase(end);
        prompt.push_back('.');
    }

    append_newline_if_missing(prompt);

    return prompt;
}

static int32_t get_token_id_from_json(const ComputeFlowManifest& manifest, const std::string& path, const std::string& key) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    json j;
    f >> j;

    if (!j.is_object()) {
        throw std::runtime_error("Expected JSON object");
    }

    std::unordered_map<std::string, std::string> vocab;
    vocab.reserve(j.size());

    for (const auto& [key, meta_key] : j.items()) {
        if (!meta_key.is_string()) {
            throw std::runtime_error("Expected string value for key: " + key);
        }
        vocab.emplace(key, meta_key.get<std::string>());
    }

    return manifest.get_constant<int32_t>(vocab[key]);
}

bool variant_entry_matches(const json& entry, const std::string& variant) {
    if (entry.is_string()) {
        return entry.get<std::string>() == variant;
    }
    if (entry.is_object()) {
        auto it = entry.find("name");
        if (it != entry.end() && it->is_string() && it->get<std::string>() == variant) {
            return true;
        }
        it = entry.find("variant");
        if (it != entry.end() && it->is_string() && it->get<std::string>() == variant) {
            return true;
        }
    }
    return false;
}

bool variant_in_table(const std::string& variant, const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open variants file: " + path);
    }

    json variants_json;
    try {
        in >> variants_json;
    } catch (const json::parse_error& e) {
        throw std::runtime_error(
            std::string("Failed to parse variants file: ") + path + " (" + e.what() + ")");
    }

    if (variants_json.is_array()) {
        for (const auto& entry : variants_json) {
            if (variant_entry_matches(entry, variant)) {
                return true;
            }
        }
        return false;
    }

    if (variants_json.is_object()) {
        auto it = variants_json.find("variants");
        if (it != variants_json.end() && it->is_array()) {
            for (const auto& entry : *it) {
                if (variant_entry_matches(entry, variant)) {
                    return true;
                }
            }
            return false;
        }
        return variants_json.find(variant) != variants_json.end();
    }

    throw std::runtime_error("Variants file has unsupported JSON format: " + path);
}

} // namespace

ValidationResult Qwen3TTS::validate_custom_voice(const ComputeFlowManifest& manifest) const {
    COMPUTE_FLOW_VALIDATE(manifest.has_module("tokenizer"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("text_embedding"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("codec_embedding"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("talker"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("code_pred"));

    // Validate presence of speech decoder modules
    COMPUTE_FLOW_VALIDATE(manifest.has_module("speech_decoder_pre"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("speech_decoder_transformer"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("speech_decoder_post"));

    // Validate presence of code_pred_post_[1:pos-1]
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_code_pred_groups"));

    const int32_t num_code_pred_groups = manifest.get_constant<int32_t>("num_code_pred_groups");
    for (int64_t pos = 1; pos < num_code_pred_groups; ++pos) {
        COMPUTE_FLOW_VALIDATE(manifest.has_module("code_pred_post_" + std::to_string(pos)));
    }

    COMPUTE_FLOW_VALIDATE(manifest.has_asset("languages_map_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("speakers_map_file"));

    COMPUTE_FLOW_VALIDATE(manifest.has_config("language"));
    COMPUTE_FLOW_VALIDATE(manifest.has_config("speaker"));
    COMPUTE_FLOW_VALIDATE(manifest.has_config("num_active_code_pred_groups"));

    COMPUTE_FLOW_VALIDATE(manifest.has_constant("enable_streaming_mode"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_audio_samples_per_frame"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_max_frames_speech_dec"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_max_frames"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("sample_rate"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("tts_pad_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("tts_bos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("tts_eos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_eos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_bos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_pad_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_think_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_no_think_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_think_bos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_think_eos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("im_start_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("im_end_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("assistant_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("newline_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_chinese_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_english_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_japanese_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_korean_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_german_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_french_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_russian_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_portuguese_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_spanish_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_italian_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_auto_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("spk_serena_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("spk_vivian_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("spk_uncle_fu_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("spk_ryan_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("spk_aiden_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("spk_ono_anna_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("spk_sohee_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("spk_eric_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("spk_dylan_token_id"));

    return {};
}

ValidationResult Qwen3TTS::validate_base(const ComputeFlowManifest& manifest) const {
    COMPUTE_FLOW_VALIDATE(manifest.has_module("tokenizer"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("text_embedding"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("codec_embedding"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("talker"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("code_pred"));

    // Validate presence of speech decoder modules
    COMPUTE_FLOW_VALIDATE(manifest.has_module("speech_decoder_pre"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("speech_decoder_transformer"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("speech_decoder_post"));

    // Validate presence of code_pred_post_[1:pos-1]
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_code_pred_groups"));

    const int32_t num_code_pred_groups = manifest.get_constant<int32_t>("num_code_pred_groups");
    for (int64_t pos = 1; pos < num_code_pred_groups; ++pos) {
        COMPUTE_FLOW_VALIDATE(manifest.has_module("code_pred_post_" + std::to_string(pos)));
    }

    // Validate presence of mel spectrogram and speaker encoder
    COMPUTE_FLOW_VALIDATE(manifest.has_module("mel_spectrogram"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("speaker_encoder"));

    COMPUTE_FLOW_VALIDATE(manifest.has_asset("languages_map_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("speakers_embeddings_path"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("ref_audio_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("ref_text_file"));

    COMPUTE_FLOW_VALIDATE(manifest.has_config("language"));
    COMPUTE_FLOW_VALIDATE(manifest.has_config("num_active_code_pred_groups"));

    COMPUTE_FLOW_VALIDATE(manifest.has_constant("enable_streaming_mode"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_audio_samples_per_frame"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_max_frames_speech_dec"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_max_frames"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("sample_rate"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("tts_pad_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("tts_bos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("tts_eos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_eos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_bos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_pad_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_think_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_no_think_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_think_bos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("codec_think_eos_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("im_start_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("im_end_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("assistant_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("newline_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_chinese_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_english_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_japanese_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_korean_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_german_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_french_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_russian_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_portuguese_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_spanish_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_italian_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("lang_auto_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("min_ref_audio_duration_sec"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("max_ref_audio_duration_sec"));

    const std::string ref_text = read_text_file(*manifest.get_asset("ref_text_file"));
    if(!ref_text.empty()) {
        COMPUTE_FLOW_VALIDATE(manifest.has_module("speech_tokenizer_encoder"));
        COMPUTE_FLOW_VALIDATE(manifest.has_module("code_pred_group_embeds"));
        COMPUTE_FLOW_VALIDATE(manifest.get_constant<bool>("enable_streaming_mode"));
    }

    return {};
}

ValidationResult Qwen3TTS::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "tts");
    COMPUTE_FLOW_VALIDATE(info.name == "Qwen3TTS");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);

    // Get Qwen3 TTS variant (Currently only custom voice and base model are supported)
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("qwen3_tts_variant"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("qwen3_tts_variants_file"));

    const std::string* variants_file = manifest.get_asset("qwen3_tts_variants_file");
    COMPUTE_FLOW_VALIDATE(variants_file != nullptr);

    const std::string qwen3_tts_variant = manifest.get_constant<std::string>("qwen3_tts_variant");
    COMPUTE_FLOW_VALIDATE(variant_in_table(qwen3_tts_variant, *variants_file) != 0);

    COMPUTE_FLOW_VALIDATE(qwen3_tts_variant == "qwen3_tts_12hz_0.6b_base" ||
                    qwen3_tts_variant == "qwen3_tts_12hz_0.6b_custom_voice");

    if(qwen3_tts_variant == "qwen3_tts_12hz_0.6b_custom_voice") {
        return validate_custom_voice(manifest);
    } else {
        return validate_base(manifest);
    }
}

void Qwen3TTS::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    const std::string qwen3_tts_variant = manifest.get_constant<std::string>("qwen3_tts_variant");

    _num_active_code_pred_groups = manifest.get_config<int32_t>("num_active_code_pred_groups");
    COMPUTE_FLOW_ASSERT(_num_active_code_pred_groups > 0);
    COMPUTE_FLOW_ASSERT(_num_active_code_pred_groups <= _num_code_pred_groups);

    if(qwen3_tts_variant == "qwen3_tts_12hz_0.6b_base") {
        const std::string language = manifest.get_config<std::string>("language");
        const std::string languages_map_file = *manifest.get_asset("languages_map_file");

        _lang_token_id = get_token_id_from_json(manifest, languages_map_file, language);
    } else {
        const std::string language = manifest.get_config<std::string>("language");
        const std::string speaker  = manifest.get_config<std::string>("speaker");
        const std::string languages_map_file = *manifest.get_asset("languages_map_file");
        const std::string speakers_map_file  = *manifest.get_asset("speakers_map_file");

        _lang_token_id = get_token_id_from_json(manifest, languages_map_file, language);
        _spk_token_id  = get_token_id_from_json(manifest, speakers_map_file, speaker);
    }
}

void Qwen3TTS::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {

    for(const auto& model : manifest.models()) {
        const std::string model_name = model.model_name;
        const RunnerType runner_type = model.framework_type;
        const LMRunnerType lm_runner_type = model.lm_runner_type;

        if(runner_type != RunnerType::Unknown) {
            _runner[model_name] = make_runner(runner_type);
            _runner[model_name]->setup(model, profiler);
        } else if(lm_runner_type != LMRunnerType::Unknown) {
            _lm_runner[model_name] = make_lm_runner(lm_runner_type);
            _lm_runner[model_name]->setup(model, profiler);
        } else {
            COMPUTE_FLOW_ASSERT(false);
        }
    }
}

void Qwen3TTS::prepare_impl_custom_voice(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    _qwen3_tts_variant           = Qwen3TTSVariant::CustomVoice_0_6B;
    _sample_rate                 = static_cast<int64_t>(manifest.get_constant<int32_t>("sample_rate"));
    _enable_streaming_mode       = manifest.get_constant<bool>("enable_streaming_mode");
    _num_max_frames_speech_dec   = manifest.get_constant<int32_t>("num_max_frames_speech_dec");
    _num_max_frames              = manifest.get_constant<int32_t>("num_max_frames");
    _num_audio_samples_per_frame = manifest.get_constant<int32_t>("num_audio_samples_per_frame");
    _num_code_pred_groups        = manifest.get_constant<int32_t>("num_code_pred_groups");
    _num_active_code_pred_groups = manifest.get_config<int32_t>("num_active_code_pred_groups");
    _tts_pad_token_id            = manifest.get_constant<int32_t>("tts_pad_token_id");
    _tts_bos_token_id            = manifest.get_constant<int32_t>("tts_bos_token_id");
    _tts_eos_token_id            = manifest.get_constant<int32_t>("tts_eos_token_id");
    _codec_eos_token_id          = manifest.get_constant<int32_t>("codec_eos_token_id");
    _codec_bos_token_id          = manifest.get_constant<int32_t>("codec_bos_token_id");
    _codec_pad_token_id          = manifest.get_constant<int32_t>("codec_pad_token_id");
    _codec_think_token_id        = manifest.get_constant<int32_t>("codec_think_token_id");
    _codec_no_think_token_id     = manifest.get_constant<int32_t>("codec_no_think_token_id");
    _codec_think_bos_token_id    = manifest.get_constant<int32_t>("codec_think_bos_token_id");
    _codec_think_eos_token_id    = manifest.get_constant<int32_t>("codec_think_eos_token_id");
    _im_start_token_id           = manifest.get_constant<int32_t>("im_start_token_id");
    _im_end_token_id             = manifest.get_constant<int32_t>("im_end_token_id");
    _assistant_token_id          = manifest.get_constant<int32_t>("assistant_token_id");
    _newline_token_id            = manifest.get_constant<int32_t>("newline_token_id");

    COMPUTE_FLOW_ASSERT(_num_active_code_pred_groups > 0);
    COMPUTE_FLOW_ASSERT(_num_active_code_pred_groups <= _num_code_pred_groups);

    // Get configs
    const std::string language = manifest.get_config<std::string>("language");
    const std::string speaker  = manifest.get_config<std::string>("speaker");
    const std::string languages_map_file = *manifest.get_asset("languages_map_file");
    const std::string speakers_map_file  = *manifest.get_asset("speakers_map_file");

    _lang_token_id = get_token_id_from_json(manifest, languages_map_file, language);
    _spk_token_id  = get_token_id_from_json(manifest, speakers_map_file, speaker);

    auto t_cod_fmt = _runner["speech_decoder_pre"]->get_input_format("codes_t");
    auto t_emb_fmt = _lm_runner["talker"]->get_input_format("inputs_embeds");

    _k_n_embd_sz = t_emb_fmt.shape[t_emb_fmt.shape.size() - 1];

    const int64_t idx_num_groups = t_cod_fmt.shape.size() - 1;
    const int64_t idx_num_frames = t_cod_fmt.shape.size() - 2;

    COMPUTE_FLOW_ASSERT(t_cod_fmt.shape[idx_num_groups] == _num_code_pred_groups);

    // If the speech decoder frame count is statically defined, use it to override
    // _num_max_frames_speech_dec.
    // Otherwise, we should set the frame count to _num_max_frames_speech_dec
    if(t_cod_fmt.shape[idx_num_frames] > 0) {
        _num_max_frames_speech_dec = t_cod_fmt.shape[idx_num_frames];
    } else {
        t_cod_fmt.shape[idx_num_frames] = _num_max_frames_speech_dec;
    }

    TensorFormat t_wav_fmt;
    t_wav_fmt.dt = DataType::F32;
    if(!_enable_streaming_mode) {
        t_wav_fmt.shape = {_num_max_frames * _num_audio_samples_per_frame};
    } else {
        t_wav_fmt.shape = {0};
    }

    for(auto&i : t_emb_fmt.shape) {
        i = std::max((int64_t)1, i);
    }

    for(auto&i : t_cod_fmt.shape) {
        i = std::max((int64_t)1, i);
    }

    const auto t_pre_conv_fmt = _runner["speech_decoder_pre"]->get_input_format("pre_conv_history");
    const auto t_lat_fmt      = _runner["speech_decoder_post"]->get_input_format("latent_buffer");
    _internal_memory.allocate("out_raw_audio", t_wav_fmt);
    _internal_memory.allocate("step_embd", t_emb_fmt);
    _internal_memory.allocate("tts_pad_embed", t_emb_fmt);
    _internal_memory.allocate("tts_bos_embed", t_emb_fmt);
    _internal_memory.allocate("tts_eos_embed", t_emb_fmt);
    _internal_memory.allocate("codec_pad_embed", t_emb_fmt);
    _internal_memory.allocate("audio_codes", t_cod_fmt);
    _internal_memory.allocate("pre_conv_zeroed", t_pre_conv_fmt);
    _internal_memory.allocate("latent_zeroed", t_lat_fmt);

    TensorData t_tts_pad_embed   = _internal_memory.get_tensor("tts_pad_embed");
    TensorData t_tts_bos_embed   = _internal_memory.get_tensor("tts_bos_embed");
    TensorData t_tts_eos_embed   = _internal_memory.get_tensor("tts_eos_embed");
    TensorData t_codec_pad_embed = _internal_memory.get_tensor("codec_pad_embed");
    TensorData t_pre_conv_zeroed = _internal_memory.get_tensor("pre_conv_zeroed");
    TensorData t_latent_zeroed   = _internal_memory.get_tensor("latent_zeroed");

    run_text_embedding(&_tts_pad_token_id, 1, (float*)t_tts_pad_embed.data);
    run_text_embedding(&_tts_bos_token_id, 1, (float*)t_tts_bos_embed.data);
    run_text_embedding(&_tts_eos_token_id, 1, (float*)t_tts_eos_embed.data);
    run_codec_embedding(&_codec_pad_token_id, 1, (float*)t_codec_pad_embed.data);

    memset(t_pre_conv_zeroed.data, 0, t_pre_conv_fmt.get_nbytes());
    memset(t_latent_zeroed.data, 0, t_lat_fmt.get_nbytes());
}

void Qwen3TTS::prepare_impl_base(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    _sample_rate                 = static_cast<int64_t>(manifest.get_constant<int32_t>("sample_rate"));
    _enable_streaming_mode       = manifest.get_constant<bool>("enable_streaming_mode");
    _num_max_frames_speech_dec   = manifest.get_constant<int32_t>("num_max_frames_speech_dec");
    _num_max_frames              = manifest.get_constant<int32_t>("num_max_frames");
    _num_audio_samples_per_frame = manifest.get_constant<int32_t>("num_audio_samples_per_frame");
    _num_code_pred_groups        = manifest.get_constant<int32_t>("num_code_pred_groups");
    _num_active_code_pred_groups = manifest.get_config<int32_t>("num_active_code_pred_groups");
    _tts_pad_token_id            = manifest.get_constant<int32_t>("tts_pad_token_id");
    _tts_bos_token_id            = manifest.get_constant<int32_t>("tts_bos_token_id");
    _tts_eos_token_id            = manifest.get_constant<int32_t>("tts_eos_token_id");
    _codec_eos_token_id          = manifest.get_constant<int32_t>("codec_eos_token_id");
    _codec_bos_token_id          = manifest.get_constant<int32_t>("codec_bos_token_id");
    _codec_pad_token_id          = manifest.get_constant<int32_t>("codec_pad_token_id");
    _codec_think_token_id        = manifest.get_constant<int32_t>("codec_think_token_id");
    _codec_no_think_token_id     = manifest.get_constant<int32_t>("codec_no_think_token_id");
    _codec_think_bos_token_id    = manifest.get_constant<int32_t>("codec_think_bos_token_id");
    _codec_think_eos_token_id    = manifest.get_constant<int32_t>("codec_think_eos_token_id");
    _im_start_token_id           = manifest.get_constant<int32_t>("im_start_token_id");
    _im_end_token_id             = manifest.get_constant<int32_t>("im_end_token_id");
    _assistant_token_id          = manifest.get_constant<int32_t>("assistant_token_id");
    _newline_token_id            = manifest.get_constant<int32_t>("newline_token_id");

    COMPUTE_FLOW_ASSERT(_num_active_code_pred_groups > 0);
    COMPUTE_FLOW_ASSERT(_num_active_code_pred_groups <= _num_code_pred_groups);

    _max_ref_audio_duration_sec = manifest.get_constant<int32_t>("max_ref_audio_duration_sec");
    _min_ref_audio_duration_sec = manifest.get_constant<int32_t>("min_ref_audio_duration_sec");

    // Get configs
    const std::string language = manifest.get_config<std::string>("language");

    // Get assets
    const std::string speakers_embeddings_path = *manifest.get_asset("speakers_embeddings_path");
    const std::string languages_map_file       = *manifest.get_asset("languages_map_file");
    const std::string ref_audio_path           = *manifest.get_asset("ref_audio_file");
    const std::string ref_text                 = read_text_file(*manifest.get_asset("ref_text_file"));
    const bool icl_mode                        = !ref_text.empty();

    if(icl_mode) {
        COMPUTE_FLOW_ASSERT(_enable_streaming_mode);
    }

    _lang_token_id = get_token_id_from_json(manifest, languages_map_file, language);
    _spk_token_id  = -1;

    auto t_cod_fmt = _runner["speech_decoder_pre"]->get_input_format("codes_t");
    auto t_emb_fmt = _lm_runner["talker"]->get_input_format("inputs_embeds");

    _k_n_embd_sz = t_emb_fmt.shape[t_emb_fmt.shape.size() - 1];

    TensorFormat t_spk_emb_fmt = TensorFormat{ DataType::F32, { 1, _k_n_embd_sz } };
    _internal_memory.allocate("speaker_embeddings", t_spk_emb_fmt);
    auto t_speaker_embedding = _internal_memory.get_tensor("speaker_embeddings");

    const uint64_t ref_audio_hash = hash_file_fnv1a(ref_audio_path);
    const std::string spk_embedding_file = find_file_in_directory(speakers_embeddings_path, std::to_string(ref_audio_hash));

    if(!spk_embedding_file.empty()) {
        load_embeddings_from_file(spk_embedding_file, t_speaker_embedding);
    } else {
        load_speaker_embedding_from_ref_audio(ref_audio_path, t_speaker_embedding);

        std::string new_embedding_file = speakers_embeddings_path + "/" + std::to_string(ref_audio_hash) + "-" +
                                            std::filesystem::path(ref_audio_path).stem().string() + ".npy";

        npy::npy_data<float> d;
        d.shape = {1, static_cast<unsigned long int>(_k_n_embd_sz)};
        d.data.resize(t_speaker_embedding.format.get_nbytes());
        memcpy(d.data.data(), t_speaker_embedding.data, t_speaker_embedding.format.get_nbytes());
        npy::write_npy(new_embedding_file, d);
    }

    const int64_t idx_num_groups = t_cod_fmt.shape.size() - 1;
    const int64_t idx_num_frames = t_cod_fmt.shape.size() - 2;

    COMPUTE_FLOW_ASSERT(t_cod_fmt.shape[idx_num_groups] == _num_code_pred_groups);

    // If the speech decoder frame count is statically defined, use it to override
    // _num_max_frames_speech_dec.
    // Otherwise, we should set the frame count to _num_max_frames_speech_dec
    if(t_cod_fmt.shape[idx_num_frames] > 0) {
        _num_max_frames_speech_dec = t_cod_fmt.shape[idx_num_frames];
    } else {
        t_cod_fmt.shape[idx_num_frames] = _num_max_frames_speech_dec;
    }

    TensorFormat t_wav_fmt;
    t_wav_fmt.dt = DataType::F32;
    if(!_enable_streaming_mode) {
        t_wav_fmt.shape = {_num_max_frames * _num_audio_samples_per_frame};
    } else {
        t_wav_fmt.shape = {0};
    }

    for(auto&i : t_emb_fmt.shape) {
        i = std::max((int64_t)1, i);
    }

    for(auto&i : t_cod_fmt.shape) {
        i = std::max((int64_t)1, i);
    }

    const auto t_pre_conv_fmt = _runner["speech_decoder_pre"]->get_input_format("pre_conv_history");
    const auto t_lat_fmt      = _runner["speech_decoder_post"]->get_input_format("latent_buffer");
    _internal_memory.allocate("out_raw_audio", t_wav_fmt);
    _internal_memory.allocate("step_embd", t_emb_fmt);
    _internal_memory.allocate("tts_pad_embed", t_emb_fmt);
    _internal_memory.allocate("tts_bos_embed", t_emb_fmt);
    _internal_memory.allocate("tts_eos_embed", t_emb_fmt);
    _internal_memory.allocate("codec_pad_embed", t_emb_fmt);
    _internal_memory.allocate("audio_codes", t_cod_fmt);
    _internal_memory.allocate("pre_conv_zeroed", t_pre_conv_fmt);
    _internal_memory.allocate("latent_zeroed", t_lat_fmt);

    TensorData t_tts_pad_embed   = _internal_memory.get_tensor("tts_pad_embed");
    TensorData t_tts_bos_embed   = _internal_memory.get_tensor("tts_bos_embed");
    TensorData t_tts_eos_embed   = _internal_memory.get_tensor("tts_eos_embed");
    TensorData t_codec_pad_embed = _internal_memory.get_tensor("codec_pad_embed");
    TensorData t_pre_conv_zeroed = _internal_memory.get_tensor("pre_conv_zeroed");
    TensorData t_latent_zeroed   = _internal_memory.get_tensor("latent_zeroed");

    run_text_embedding(&_tts_pad_token_id, 1, (float*)t_tts_pad_embed.data);
    run_text_embedding(&_tts_bos_token_id, 1, (float*)t_tts_bos_embed.data);
    run_text_embedding(&_tts_eos_token_id, 1, (float*)t_tts_eos_embed.data);
    run_codec_embedding(&_codec_pad_token_id, 1, (float*)t_codec_pad_embed.data);

    memset(t_pre_conv_zeroed.data, 0, t_pre_conv_fmt.get_nbytes());
    memset(t_latent_zeroed.data, 0, t_lat_fmt.get_nbytes());

    if(icl_mode) {
        icl_load_ref_codes_from_ref_audio_base(ref_audio_path);
        icl_precompute_embeddings_base(ref_text);
        icl_precompute_talker_prefix_state_base();
        _t_state_pre_conv = t_pre_conv_zeroed;
        _t_state_latent = t_latent_zeroed;
        icl_precompute_speech_decoder_state_base(
            _internal_memory.get_tensor("audio_codes"),
            _internal_memory.get_tensor("out_raw_audio"),
            0);
        _qwen3_tts_variant = Qwen3TTSVariant::Base_ICL_0_6B;
    } else {
        _qwen3_tts_variant = Qwen3TTSVariant::Base_0_6B;
    }
}

void Qwen3TTS::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    std::string qwen3_tts_variant = manifest.get_constant<std::string>("qwen3_tts_variant");

    if(qwen3_tts_variant == "qwen3_tts_12hz_0.6b_base") {
        prepare_impl_base(manifest, profiler);
    } else {
        prepare_impl_custom_voice(manifest, profiler);
    }
}

void Qwen3TTS::warmup_impl(ProfilerView profiler) {
    TensorDict input;
    input.declare_tensor("text_prompt");

    const std::string warmup_text = "warmup";
    input["text_prompt"].data = (void*)warmup_text.c_str();
    input["text_prompt"].format.dt = DataType::String;
    input["text_prompt"].format.shape = {static_cast<int64_t>(warmup_text.size())};

    run_impl(input, profiler);
}

TensorDict Qwen3TTS::run_impl_custom_voice(const TensorDict& input, ProfilerView profiler) {

    const bool is_first_run = _is_first_run_streaming_mode;

    TensorData t_tts_pad_embed   = _internal_memory.get_tensor("tts_pad_embed");
    TensorData t_tts_bos_embed   = _internal_memory.get_tensor("tts_bos_embed");
    TensorData t_tts_eos_embed   = _internal_memory.get_tensor("tts_eos_embed");
    TensorData t_codec_pad_embed = _internal_memory.get_tensor("codec_pad_embed");
    TensorData t_audio_out       = _internal_memory.get_tensor("out_raw_audio");
    TensorData t_step_embd       = _internal_memory.get_tensor("step_embd");
    TensorData t_codes           = _internal_memory.get_tensor("audio_codes");
    TensorData t_pre_conv_zeroed = _internal_memory.get_tensor("pre_conv_zeroed");
    TensorData t_latent_zeroed   = _internal_memory.get_tensor("latent_zeroed");

    // Get the maximum number of output samples to write in the t_audio_out
    // to avoid out of bounds writes in non-streaming mode
    const int64_t max_output_audio_samples = t_audio_out.format.shape[0];

    if(is_first_run) {
        // Run tokenizer
        const std::string text_prompt = tensor_to_string(input["text_prompt"]);
        const std::string normalized_prompt = ensure_prompt_ends_with_punct(text_prompt);

        TensorData t_prompt;
        t_prompt.data = (void *)normalized_prompt.c_str();
        t_prompt.format.dt = DataType::String;
        t_prompt.format.shape = { static_cast<int64_t>(normalized_prompt.size()) };

        _runner["tokenizer"]->set_input("text_prompt", t_prompt);
        _runner["tokenizer"]->run();
        const auto t_text_tokens = _runner["tokenizer"]->get_output("token_ids");

        // Assemble: [im_start, assistant, \n] + text_tokens + [im_end, \n, im_start, assistant, \n] */
        const int64_t role_len       = 3; // [im_start, assistant, \n]
        const int64_t text_token_len = t_text_tokens.format.get_num_elems();
        const int64_t suffix_len     = 5; // [im_end, \n, im_start, assistant, \n]
        const int64_t all_len = role_len + text_token_len + suffix_len;

        std::vector<int32_t> all_tokens;
        all_tokens.resize(all_len);
        int32_t pos_t = 0;
        all_tokens[pos_t++] = _im_start_token_id;       /* <|im_start|> */
        all_tokens[pos_t++] = _assistant_token_id;      /* assistant */
        all_tokens[pos_t++] = _newline_token_id;        /* \n */

        memcpy(all_tokens.data() + pos_t, t_text_tokens.data, text_token_len * sizeof(int32_t));

        pos_t += text_token_len;
        all_tokens[pos_t++] = _im_end_token_id;         /* <|im_end|> */
        all_tokens[pos_t++] = _newline_token_id;        /* \n */
        all_tokens[pos_t++] = _im_start_token_id;       /* <|im_start|> */
        all_tokens[pos_t++] = _assistant_token_id;      /* assistant */
        all_tokens[pos_t++] = _newline_token_id;        /* \n */

        // Build codec-side prefix:
        // CustomVoice with language: [THINK, THINK_BOS, _lang_token_id, THINK_EOS, speaker, PAD, BOS]
        // CustomVoice without language: [NO_THINK, THINK_BOS, THINK_EOS, speaker, PAD, BOS]
        // VoiceDesign with language: [THINK, THINK_BOS, _lang_token_id, THINK_EOS, PAD, BOS]
        // VoiceDesign without language: [NO_THINK, THINK_BOS, THINK_EOS, PAD, BOS]
        std::vector<int32_t> codec_tokens;
        codec_tokens.reserve(16);

        if (_lang_token_id >= 0) {
            codec_tokens.push_back(_codec_think_token_id);
            codec_tokens.push_back(_codec_think_bos_token_id);
            codec_tokens.push_back(_lang_token_id);
            codec_tokens.push_back(_codec_think_eos_token_id);
        } else {
            codec_tokens.push_back(_codec_no_think_token_id);
            codec_tokens.push_back(_codec_think_bos_token_id);
            codec_tokens.push_back(_codec_think_eos_token_id);
        }

        /* Speaker position in codec prefix:
        * - CustomVoice: discrete speaker token from spk_id
        * - VoiceDesign: no speaker
        * - Voice Clone: continuous speaker embedding replaces token
        *   We use -1 as sentinel; the embedding loop handles it specially. */
        //if (ctx->voice_clone && ctx->speaker_embedding) {
        //    codec_tokens[codec_len++] = -1;  /* placeholder — will use speaker_embedding */
        //} else if (!ctx->voice_design) {
        //    codec_tokens[codec_len++] = ctx->speaker_id;
        //}

        codec_tokens.push_back(_spk_token_id);
        codec_tokens.push_back(_codec_pad_token_id);
        codec_tokens.push_back(_codec_bos_token_id);

        // Build prefill. The input of LM consists of the following sections, depending on the mode
        //
        // Section 1: Role prefix (3 positions) - text-only, NO codec pairing
        // Section 2: tts_pad*(codec_len-2) + tts_bos paired with codec[0..codec_len-2]
        //
        // Normal mode (non-ICL):
        //   Section 3: text_content[0..N-1] + tts_eos paired with codec_pad * (N+1)
        //   Section 4: tts_pad + codec_bos (1 position)
        const int64_t codec_len = codec_tokens.size();

        const int64_t sec1_len = role_len;
        const int64_t sec2_len = codec_len - 1; /* codec tokens without the last (BOS) */
        const int64_t sec3_len = text_token_len + 1; /* text + eos */
        const int64_t sec4_len = 1; /* bos */

        const int64_t prefill_len = sec1_len + sec2_len + sec3_len + sec4_len;

        auto talker_in_fmt = _lm_runner["talker"]->get_input_format("inputs_embeds");
        talker_in_fmt.shape[0] = prefill_len;
        COMPUTE_FLOW_ASSERT((prefill_len * _k_n_embd_sz) == talker_in_fmt.get_num_elems());

        _internal_memory.allocate("input_embeds", talker_in_fmt);

        float* input_embeds = _internal_memory.get_data_as<float>("input_embeds");

        // Convert all tokens to embedding. We have prefill_len tokens and, each one will generate an
        // embedding of size _k_n_embd_sz

        int64_t pos_embedding = 0;

        // Section 1 - Role prefix tokens
        run_text_embedding(all_tokens.data(), sec1_len, input_embeds + pos_embedding);
        pos_embedding += (sec1_len * _k_n_embd_sz);

        // Section 2 - codec prefix without the trailing codec_bos.
        // In voice-clone mode, the speaker slot is filled by the externally supplied speaker embeddings.
        float* sec2_embeds = input_embeds + pos_embedding;

        run_codec_embedding(codec_tokens.data(), sec2_len, sec2_embeds);

        add_special_embedding((float*)t_tts_pad_embed.data, sec2_len - 1, _k_n_embd_sz, sec2_embeds);
        add_special_embedding((float*)t_tts_bos_embed.data, 1, _k_n_embd_sz,
                            sec2_embeds + ((sec2_len - 1) * _k_n_embd_sz));

        pos_embedding += (sec2_len * _k_n_embd_sz);

        // Section 3 - text content + tts_eos, all paired with codec_pad
        run_text_embedding(all_tokens.data() + sec1_len, sec3_len - 1, input_embeds + pos_embedding);
        add_special_embedding((float*)t_codec_pad_embed.data, sec3_len - 1, _k_n_embd_sz, input_embeds + pos_embedding);
        pos_embedding += ((sec3_len - 1) * _k_n_embd_sz);
        run_codec_embedding(&_codec_pad_token_id, 1, input_embeds + pos_embedding);
        add_special_embedding((float*)t_tts_eos_embed.data, 1, _k_n_embd_sz, input_embeds + pos_embedding);
        pos_embedding += _k_n_embd_sz;

        // Section 4 - tts_pad + codec_bos
        run_codec_embedding(&_codec_bos_token_id, 1, input_embeds + pos_embedding);
        add_special_embedding((float*)t_tts_pad_embed.data, 1, _k_n_embd_sz, input_embeds + pos_embedding);
        pos_embedding += _k_n_embd_sz;

        COMPUTE_FLOW_ASSERT((prefill_len * _k_n_embd_sz) == pos_embedding);

        auto t_prefill_emb = _internal_memory.get_tensor("input_embeds");

        // Run encode
        _lm_runner["talker"]->set_input("inputs_embeds", t_prefill_emb);
        _lm_runner["talker"]->run();

        // Initialize the audio output shape to zero; in non-streaming mode,
        // it will grow with each invocation of the speech tokenizer.
        t_audio_out.format.shape[0] = 0;

        // Release the memory for the input embedding because not needed anymore
        _internal_memory.release("input_embeds");

        // Initialize the states of the speech tokenizer to zero
        _t_state_pre_conv = t_pre_conv_zeroed;
        _t_state_latent = t_latent_zeroed;
    }

    // Generate the audio frames
    int64_t num_frames_to_gen = _num_max_frames;

    if(_enable_streaming_mode) {
        _is_first_run_streaming_mode = false;
        // If the user doesn't reset the state, streaming mode stops after completion.
        num_frames_to_gen = is_generation_complete()? 0 : _num_max_frames_speech_dec;
    }

    for (int64_t frame_idx = 0; frame_idx < num_frames_to_gen; ++frame_idx) {

        int32_t* codes = (int32_t*)t_codes.data + (frame_idx % _num_max_frames_speech_dec) * _num_code_pred_groups;

        auto t_last_hidden_state = _lm_runner["talker"]->get_output("hidden_state");
        auto t_code0             = _lm_runner["talker"]->get_output("token_id");

        // The first code is from the talker. The following codes are from the code predictor
        codes[0] = *static_cast<const int32_t*>(t_code0.data);

        if (codes[0] == _codec_eos_token_id) {
            const int64_t frame_pos        = frame_idx % _num_max_frames_speech_dec;
            const int64_t frames_processed = frame_pos;
            const int64_t frames_remaining = _num_max_frames_speech_dec - frames_processed;
            constexpr int32_t k_speech_decoder_pad_code_id = 0;

            // Leftover frame 0: _num_code_pred_groups - 1, because code[0] is current token.
            // Leftover frames N: (frames_remaining - 1) * _num_code_pred_groups
            const int64_t pad_count =
                ((frames_remaining - 1) * _num_code_pred_groups) + (_num_code_pred_groups - 1);

            // speech_decoder_pre expects codebook IDs in [0, 2047]. Replace EOS/specials.
            codes[0] = k_speech_decoder_pad_code_id;
            std::fill_n(codes + 1, pad_count, k_speech_decoder_pad_code_id);

            run_speech_decoder(t_codes, frames_processed, t_audio_out, max_output_audio_samples, _enable_streaming_mode);

            // Signal the completion of the generation
            _is_output_ready = true;
            break;
        }

        TensorData t_codes_sliced;
        t_codes_sliced.data         = (void*)codes;
        t_codes_sliced.format.dt    = DataType::I32;
        t_codes_sliced.format.shape = {_num_code_pred_groups};

        // Run code predictor to infer code[1], code[2],...code[14]
        run_code_predictor(t_last_hidden_state, t_codes_sliced, t_step_embd);

        const int64_t num_frames_processed = frame_idx + 1;
        if((num_frames_processed % _num_max_frames_speech_dec == 0) && (frame_idx != 0)) {
            run_speech_decoder(t_codes, num_frames_processed, t_audio_out, max_output_audio_samples, _enable_streaming_mode);
        }

        // Finalize the input to the LM backbone by adding up tts_pad_embed
        custom_op_in_place_add(t_tts_pad_embed, t_step_embd);

        _lm_runner["talker"]->set_input("inputs_embeds", t_step_embd);
        _lm_runner["talker"]->run();
    }

    // In non-streaming mode, is_output_ready is true either we hit the EOG token id
    // or we reached the maximum number of frames to generate
    if(!_enable_streaming_mode) {
        // Signal the completion of the generation
        _is_output_ready = true;
    }

    TensorDict output;
    output.declare_tensor("left_channel");
    output.declare_tensor("right_channel");
    output.declare_tensor("sample_rate");

    output["left_channel"]  = t_audio_out;
    output["right_channel"] = t_audio_out;

    output["sample_rate"].data         = (void*)&_sample_rate;
    output["sample_rate"].format.dt    = DataType::I64;
    output["sample_rate"].format.shape = {1};

    return output;
}

TensorDict Qwen3TTS::run_impl_base(const TensorDict& input, ProfilerView profiler) {

    const bool is_first_run = _is_first_run_streaming_mode;

    TensorData t_tts_pad_embed     = _internal_memory.get_tensor("tts_pad_embed");
    TensorData t_tts_bos_embed     = _internal_memory.get_tensor("tts_bos_embed");
    TensorData t_tts_eos_embed     = _internal_memory.get_tensor("tts_eos_embed");
    TensorData t_codec_pad_embed   = _internal_memory.get_tensor("codec_pad_embed");
    TensorData t_speaker_embedding = _internal_memory.get_tensor("speaker_embeddings");
    TensorData t_audio_out         = _internal_memory.get_tensor("out_raw_audio");
    TensorData t_step_embd         = _internal_memory.get_tensor("step_embd");
    TensorData t_codes             = _internal_memory.get_tensor("audio_codes");
    TensorData t_pre_conv_zeroed   = _internal_memory.get_tensor("pre_conv_zeroed");
    TensorData t_latent_zeroed     = _internal_memory.get_tensor("latent_zeroed");

    // Get the maximum number of output samples to write in the t_audio_out
    // to avoid out of bounds writes in non-streaming mode
    const int64_t max_output_audio_samples = t_audio_out.format.shape[0];

    if(is_first_run) {
        // Run tokenizer
        const std::string text_prompt = tensor_to_string(input["text_prompt"]);
        const std::string normalized_prompt = ensure_prompt_ends_with_punct(text_prompt);

        TensorData t_prompt;
        t_prompt.data = (void *)normalized_prompt.c_str();
        t_prompt.format.dt = DataType::String;
        t_prompt.format.shape = { static_cast<int64_t>(normalized_prompt.size()) };

        _runner["tokenizer"]->set_input("text_prompt", t_prompt);
        _runner["tokenizer"]->run();
        const auto t_text_tokens = _runner["tokenizer"]->get_output("token_ids");

        // Assemble: [im_start, assistant, \n] + text_tokens + [im_end, \n, im_start, assistant, \n] */
        const int64_t role_len       = 3; // [im_start, assistant, \n]
        const int64_t text_token_len = t_text_tokens.format.get_num_elems();
        const int64_t suffix_len     = 5; // [im_end, \n, im_start, assistant, \n]
        const int64_t all_len = role_len + text_token_len + suffix_len;

        std::vector<int32_t> all_tokens;
        all_tokens.resize(all_len);
        int32_t pos_t = 0;
        all_tokens[pos_t++] = _im_start_token_id;       /* <|im_start|> */
        all_tokens[pos_t++] = _assistant_token_id;      /* assistant */
        all_tokens[pos_t++] = _newline_token_id;        /* \n */

        memcpy(all_tokens.data() + pos_t, t_text_tokens.data, text_token_len * sizeof(int32_t));

        pos_t += text_token_len;
        all_tokens[pos_t++] = _im_end_token_id;         /* <|im_end|> */
        all_tokens[pos_t++] = _newline_token_id;        /* \n */
        all_tokens[pos_t++] = _im_start_token_id;       /* <|im_start|> */
        all_tokens[pos_t++] = _assistant_token_id;      /* assistant */
        all_tokens[pos_t++] = _newline_token_id;        /* \n */

        // Build codec-side prefix:
        // CustomVoice with language: [THINK, THINK_BOS, _lang_token_id, THINK_EOS, speaker, PAD, BOS]
        // CustomVoice without language: [NO_THINK, THINK_BOS, THINK_EOS, speaker, PAD, BOS]
        // VoiceDesign with language: [THINK, THINK_BOS, _lang_token_id, THINK_EOS, PAD, BOS]
        // VoiceDesign without language: [NO_THINK, THINK_BOS, THINK_EOS, PAD, BOS]
        std::vector<int32_t> codec_tokens;
        codec_tokens.reserve(16);

        if (_lang_token_id >= 0) {
            codec_tokens.push_back(_codec_think_token_id);
            codec_tokens.push_back(_codec_think_bos_token_id);
            codec_tokens.push_back(_lang_token_id);
            codec_tokens.push_back(_codec_think_eos_token_id);
        } else {
            codec_tokens.push_back(_codec_no_think_token_id);
            codec_tokens.push_back(_codec_think_bos_token_id);
            codec_tokens.push_back(_codec_think_eos_token_id);
        }

        /* Speaker position in codec prefix:
        * - CustomVoice: discrete speaker token from spk_id
        * - VoiceDesign: no speaker
        * - Voice Clone: continuous speaker embedding replaces token
        *   We use -1 as sentinel; the embedding loop handles it specially. */
        //if (ctx->voice_clone && ctx->speaker_embedding) {
        //    codec_tokens[codec_len++] = -1;  /* placeholder — will use speaker_embedding */
        //} else if (!ctx->voice_design) {
        //    codec_tokens[codec_len++] = ctx->speaker_id;
        //}

        codec_tokens.push_back(_codec_pad_token_id);
        codec_tokens.push_back(_codec_bos_token_id);

        // Build prefill. The input of LM consists of the following sections, depending on the mode
        //
        // Section 1: Role prefix (3 positions) - text-only, NO codec pairing
        // Section 2: tts_pad*(codec_len-2) + tts_bos paired with codec[0..codec_len-2]
        //
        // Normal mode (non-ICL):
        //   Section 3: text_content[0..N-1] + tts_eos paired with codec_pad * (N+1)
        //   Section 4: tts_pad + codec_bos (1 position)
        const int64_t codec_len = codec_tokens.size();

        const int64_t sec1_len = role_len;
        const int64_t sec2_len = codec_len;        /* codec tokens with the last (BOS) */
        const int64_t sec3_len = text_token_len + 1;  /* text + eos */
        const int64_t sec4_len = 1;                   /* bos */

        const int64_t prefill_len = sec1_len + sec2_len + sec3_len + sec4_len;

        auto talker_in_fmt = _lm_runner["talker"]->get_input_format("inputs_embeds");
        talker_in_fmt.shape[0] = prefill_len;
        COMPUTE_FLOW_ASSERT((prefill_len * _k_n_embd_sz) == talker_in_fmt.get_num_elems());

        _internal_memory.allocate("input_embeds", talker_in_fmt);

        float* input_embeds = _internal_memory.get_data_as<float>("input_embeds");

        // Convert all tokens to embedding. We have prefill_len tokens and, each one will generate an
        // embedding of size _k_n_embd_sz

        int64_t pos_embedding = 0;

        // Section 1 - Role prefix tokens
        run_text_embedding(all_tokens.data(), sec1_len, input_embeds + pos_embedding);
        pos_embedding += (sec1_len * _k_n_embd_sz);

        // Section 2 - codec prefix without the trailing codec_bos.
        // In voice-clone mode, the speaker slot is filled by the externally supplied speaker embeddings.
        float* sec2_embeds = input_embeds + pos_embedding;

        const int64_t pre_speaker_len = codec_len - 2;
        float* sec2_write_ptr = sec2_embeds;

        if(pre_speaker_len > 0) {
            run_codec_embedding(codec_tokens.data(), pre_speaker_len, sec2_write_ptr);
            sec2_write_ptr += (pre_speaker_len * _k_n_embd_sz);
        }

        memcpy(sec2_write_ptr, t_speaker_embedding.data, t_speaker_embedding.format.get_nbytes());
        sec2_write_ptr += _k_n_embd_sz;

        memcpy(sec2_write_ptr, t_codec_pad_embed.data, t_codec_pad_embed.format.get_nbytes());

        add_special_embedding((float*)t_tts_pad_embed.data, sec2_len - 1, _k_n_embd_sz, sec2_embeds);
        add_special_embedding((float*)t_tts_bos_embed.data, 1, _k_n_embd_sz,
                            sec2_embeds + ((sec2_len - 1) * _k_n_embd_sz));

        pos_embedding += (sec2_len * _k_n_embd_sz);

        // Section 3 - text content + tts_eos, all paired with codec_pad
        run_text_embedding(all_tokens.data() + sec1_len, sec3_len - 1, input_embeds + pos_embedding);
        add_special_embedding((float*)t_codec_pad_embed.data, sec3_len - 1, _k_n_embd_sz, input_embeds + pos_embedding);
        pos_embedding += ((sec3_len - 1) * _k_n_embd_sz);
        run_codec_embedding(&_codec_pad_token_id, 1, input_embeds + pos_embedding);
        add_special_embedding((float*)t_tts_eos_embed.data, 1, _k_n_embd_sz, input_embeds + pos_embedding);
        pos_embedding += _k_n_embd_sz;

        // Section 4 - tts_pad + codec_bos
        run_codec_embedding(&_codec_bos_token_id, 1, input_embeds + pos_embedding);
        add_special_embedding((float*)t_tts_pad_embed.data, 1, _k_n_embd_sz, input_embeds + pos_embedding);
        pos_embedding += _k_n_embd_sz;

        COMPUTE_FLOW_ASSERT((prefill_len * _k_n_embd_sz) == pos_embedding);

        auto t_prefill_emb = _internal_memory.get_tensor("input_embeds");

        // Run encode
        _lm_runner["talker"]->set_input("inputs_embeds", t_prefill_emb);
        _lm_runner["talker"]->run();

        // Initialize the audio output shape to zero; in non-streaming mode,
        // it will grow with each invocation of the speech tokenizer.
        t_audio_out.format.shape[0] = 0;

        // Release the memory for the input embedding because not needed anymore
        _internal_memory.release("input_embeds");

        // Initialize the states of the speech tokenizer to zero
        _t_state_pre_conv = t_pre_conv_zeroed;
        _t_state_latent = t_latent_zeroed;
    }

    // Generate the audio frames
    int64_t num_frames_to_gen = _num_max_frames;

    if(_enable_streaming_mode) {
        _is_first_run_streaming_mode = false;
        // If the user doesn't reset the state, streaming mode stops after completion.
        num_frames_to_gen = is_generation_complete()? 0 : _num_max_frames_speech_dec;
    }

    for (int64_t frame_idx = 0; frame_idx < num_frames_to_gen; ++frame_idx) {

        int32_t* codes = (int32_t*)t_codes.data + (frame_idx % _num_max_frames_speech_dec) * _num_code_pred_groups;

        auto t_last_hidden_state = _lm_runner["talker"]->get_output("hidden_state");
        auto t_code0             = _lm_runner["talker"]->get_output("token_id");

        // The first code is from the talker. The following codes are from the code predictor
        codes[0] = *static_cast<const int32_t*>(t_code0.data);

        if (codes[0] == _codec_eos_token_id) {
            const int64_t frame_pos        = frame_idx % _num_max_frames_speech_dec;
            const int64_t frames_processed = frame_pos;
            const int64_t frames_remaining = _num_max_frames_speech_dec - frames_processed;
            constexpr int32_t k_speech_decoder_pad_code_id = 0;

            // Leftover frame 0: _num_code_pred_groups - 1, because code[0] is current token.
            // Leftover frames N: (frames_remaining - 1) * _num_code_pred_groups
            const int64_t pad_count =
                ((frames_remaining - 1) * _num_code_pred_groups) + (_num_code_pred_groups - 1);

            // speech_decoder_pre expects codebook IDs in [0, 2047]. Replace EOS/specials.
            codes[0] = k_speech_decoder_pad_code_id;
            std::fill_n(codes + 1, pad_count, k_speech_decoder_pad_code_id);

            run_speech_decoder(t_codes, frames_processed, t_audio_out, max_output_audio_samples, _enable_streaming_mode);

            // Signal the completion of the generation
            _is_output_ready = true;
            break;
        }

        TensorData t_codes_sliced;
        t_codes_sliced.data         = (void*)codes;
        t_codes_sliced.format.dt    = DataType::I32;
        t_codes_sliced.format.shape = {_num_code_pred_groups};

        // Run code predictor to infer code[1], code[2],...code[14]
        run_code_predictor(t_last_hidden_state, t_codes_sliced, t_step_embd);

        const int64_t num_frames_processed = frame_idx + 1;
        if((num_frames_processed % _num_max_frames_speech_dec == 0) && (frame_idx != 0)) {
            run_speech_decoder(t_codes, num_frames_processed, t_audio_out, max_output_audio_samples, _enable_streaming_mode);
        }

        // Finalize the input to the LM backbone by adding up tts_pad_embed
        custom_op_in_place_add(t_tts_pad_embed, t_step_embd);

        _lm_runner["talker"]->set_input("inputs_embeds", t_step_embd);
        _lm_runner["talker"]->run();
    }

    // In non-streaming mode, is_output_ready is true either we hit the EOG token id
    // or we reached the maximum number of frames to generate
    if(!_enable_streaming_mode) {
        // Signal the completion of the generation
        _is_output_ready = true;
    }

    TensorDict output;
    output.declare_tensor("left_channel");
    output.declare_tensor("right_channel");
    output.declare_tensor("sample_rate");

    output["left_channel"]  = t_audio_out;
    output["right_channel"] = t_audio_out;
    output["sample_rate"].data         = (void*)&_sample_rate;
    output["sample_rate"].format.dt    = DataType::I64;
    output["sample_rate"].format.shape = {1};

    return output;
}

TensorDict Qwen3TTS::run_impl_base_icl(const TensorDict& input, ProfilerView profiler) {

    const bool is_first_run = _is_first_run_streaming_mode;

    TensorData t_tts_pad_embed   = _internal_memory.get_tensor("tts_pad_embed");
    TensorData t_codec_pad_embed = _internal_memory.get_tensor("codec_pad_embed");
    TensorData t_audio_out       = _internal_memory.get_tensor("out_raw_audio");
    TensorData t_step_embd       = _internal_memory.get_tensor("step_embd");
    TensorData t_codes           = _internal_memory.get_tensor("audio_codes");

    // Get the maximum number of output samples to write in the t_audio_out
    // to avoid out of bounds writes in non-streaming mode.
    const int64_t max_output_audio_samples = t_audio_out.format.shape[0];

    if(is_first_run) {
        const std::string text_prompt = tensor_to_string(input["text_prompt"]);
        const std::string normalized_prompt = ensure_prompt_ends_with_punct(text_prompt);

        TensorData t_prompt;
        t_prompt.data = (void *)normalized_prompt.c_str();
        t_prompt.format.dt = DataType::String;
        t_prompt.format.shape = { static_cast<int64_t>(normalized_prompt.size()) };

        _runner["tokenizer"]->set_input("text_prompt", t_prompt);
        _runner["tokenizer"]->run();
        const auto t_text_tokens = _runner["tokenizer"]->get_output("token_ids");
        const int64_t text_token_len = t_text_tokens.format.get_num_elems();

        const int64_t prefill_len = text_token_len + _icl_prefill_suffix_len;

        auto talker_in_fmt = _lm_runner["talker"]->get_input_format("inputs_embeds");
        talker_in_fmt.shape[0] = prefill_len;
        COMPUTE_FLOW_ASSERT((prefill_len * _k_n_embd_sz) == talker_in_fmt.get_num_elems());

        _internal_memory.allocate("input_embeds", talker_in_fmt);
        float* input_embeds = _internal_memory.get_data_as<float>("input_embeds");

        int64_t pos_embedding = 0;

        if (text_token_len > 0) {
            run_text_embedding(static_cast<const int32_t*>(t_text_tokens.data),
                               text_token_len,
                               input_embeds + pos_embedding);
            add_special_embedding((float*)t_codec_pad_embed.data,
                                  text_token_len,
                                  _k_n_embd_sz,
                                  input_embeds + pos_embedding);
            pos_embedding += text_token_len * _k_n_embd_sz;
        }

        TensorData t_icl_suffix = _internal_memory.get_tensor("icl_prefill_suffix_embeds");
        memcpy(input_embeds + pos_embedding, t_icl_suffix.data, t_icl_suffix.format.get_nbytes());
        pos_embedding += _icl_prefill_suffix_len * _k_n_embd_sz;

        COMPUTE_FLOW_ASSERT((prefill_len * _k_n_embd_sz) == pos_embedding);

        auto t_prefill_emb = _internal_memory.get_tensor("input_embeds");

        _lm_runner["talker"]->restore_sequence_state(kIclTalkerPrefixStateName);
        _lm_runner["talker"]->set_input("inputs_embeds", t_prefill_emb);
        _lm_runner["talker"]->run();

        t_audio_out.format.shape[0] = 0;

        _internal_memory.release("input_embeds");

        _t_state_pre_conv = _internal_memory.get_tensor("icl_primed_pre_conv_state");
        _t_state_latent = _internal_memory.get_tensor("icl_primed_latent_state");
        _lm_runner["speech_decoder_transformer"]->restore_sequence_state(kIclSpeechDecoderStateName);
    }

    int64_t num_frames_to_gen = _num_max_frames;
    bool trim_initial_audio = is_first_run;

    if(_enable_streaming_mode) {
        _is_first_run_streaming_mode = false;
        num_frames_to_gen = is_generation_complete()? 0 : _num_max_frames_speech_dec;
    }

    for (int64_t frame_idx = 0; frame_idx < num_frames_to_gen; ++frame_idx) {

        int32_t* codes = (int32_t*)t_codes.data + (frame_idx % _num_max_frames_speech_dec) * _num_code_pred_groups;

        auto t_last_hidden_state = _lm_runner["talker"]->get_output("hidden_state");
        auto t_code0             = _lm_runner["talker"]->get_output("token_id");

        codes[0] = *static_cast<const int32_t*>(t_code0.data);

        if (codes[0] == _codec_eos_token_id) {
            const int64_t frame_pos        = frame_idx % _num_max_frames_speech_dec;
            const int64_t frames_processed = frame_pos;
            const int64_t frames_remaining = _num_max_frames_speech_dec - frames_processed;
            constexpr int32_t k_speech_decoder_pad_code_id = 0;

            const int64_t pad_count =
                ((frames_remaining - 1) * _num_code_pred_groups) + (_num_code_pred_groups - 1);

            codes[0] = k_speech_decoder_pad_code_id;
            std::fill_n(codes + 1, pad_count, k_speech_decoder_pad_code_id);

            run_speech_decoder(t_codes, frames_processed, t_audio_out, max_output_audio_samples, _enable_streaming_mode);
            if(trim_initial_audio) {
                icl_trim_initial_audio_base(t_audio_out);
                trim_initial_audio = false;
            }

            _is_output_ready = true;
            break;
        }

        TensorData t_codes_sliced;
        t_codes_sliced.data         = (void*)codes;
        t_codes_sliced.format.dt    = DataType::I32;
        t_codes_sliced.format.shape = {_num_code_pred_groups};

        run_code_predictor(t_last_hidden_state, t_codes_sliced, t_step_embd);
        ++_num_generated_frames;

        const int64_t num_frames_processed = frame_idx + 1;
        bool chunk_flushed = false;
        if((num_frames_processed % _num_max_frames_speech_dec == 0) && (frame_idx != 0)) {
            run_speech_decoder(t_codes, num_frames_processed, t_audio_out, max_output_audio_samples, _enable_streaming_mode);
            if(trim_initial_audio) {
                icl_trim_initial_audio_base(t_audio_out);
                trim_initial_audio = false;
            }
            chunk_flushed = true;
        }

        if(_num_generated_frames >= _num_max_frames) {
            if(!chunk_flushed) {
                run_speech_decoder(t_codes, num_frames_processed, t_audio_out, max_output_audio_samples, _enable_streaming_mode);
                if(trim_initial_audio) {
                    icl_trim_initial_audio_base(t_audio_out);
                    trim_initial_audio = false;
                }
            }
            _is_output_ready = true;
            break;
        }

        custom_op_in_place_add(t_tts_pad_embed, t_step_embd);

        _lm_runner["talker"]->set_input("inputs_embeds", t_step_embd);
        _lm_runner["talker"]->run();
    }

    TensorDict output;
    output.declare_tensor("left_channel");
    output.declare_tensor("right_channel");
    output.declare_tensor("sample_rate");

    output["left_channel"]  = t_audio_out;
    output["right_channel"] = t_audio_out;

    output["sample_rate"].data         = (void*)&_sample_rate;
    output["sample_rate"].format.dt    = DataType::I64;
    output["sample_rate"].format.shape = {1};

    return output;
}

TensorDict Qwen3TTS::run_impl(const TensorDict& input, ProfilerView profiler) {

    switch(_qwen3_tts_variant){
        case Qwen3TTSVariant::Base_0_6B: {
            return run_impl_base(input, profiler);
        }
        break;
        case Qwen3TTSVariant::Base_ICL_0_6B: {
            return run_impl_base_icl(input, profiler);
        }
        break;
        case Qwen3TTSVariant::CustomVoice_0_6B: {
            return run_impl_custom_voice(input, profiler);
        }
        break;
        default:
            COMPUTE_FLOW_ASSERT(false);
            throw std::runtime_error("Unsupported qwen3_tts variant.");
    }
}

bool Qwen3TTS::is_generation_complete() const {
    return _is_output_ready;
}

void Qwen3TTS::reset_states_impl(ProfilerView profiler) {
    _is_output_ready             = false;
    _is_first_run_streaming_mode = true;
    _num_generated_frames        = 0;

    for (auto& [name, runner] : _runner) {
        runner->reset_states();
    }

    for (auto& [name, runner] : _lm_runner) {
        runner->reset_states();
    }
}

void Qwen3TTS::custom_op_in_place_add(const TensorData& t_lhs, TensorData& t_rhs_out) {
    const int64_t num_lhs_elems = t_lhs.format.get_num_elems();
    const int64_t num_rhs_elems = t_rhs_out.format.get_num_elems();

    COMPUTE_FLOW_ASSERT(num_lhs_elems == num_rhs_elems);

    float* rhs_out = (float*)t_rhs_out.data;
    float* lhs     = (float*)t_lhs.data;

    for (int32_t i = 0; i < num_rhs_elems; ++i) {
        rhs_out[i] += lhs[i];
    }
}

void Qwen3TTS::run_text_embedding(const int32_t* ids, int64_t n, float* out) {
    // text_embedding: [1, n] int32 → [1, n, 1024] float32
    TensorData t_ids;
    t_ids.data         = (void*)(ids);
    t_ids.format.dt    = DataType::I32;
    t_ids.format.shape = {1, n};

    _runner["text_embedding"]->set_input("token_ids", t_ids);
    _runner["text_embedding"]->run();
    auto t_emb = _runner["text_embedding"]->get_output("embeddings");

    memcpy(out, t_emb.data, t_emb.format.get_nbytes());
}

void Qwen3TTS::run_codec_embedding(const int32_t* ids, int64_t n, float* out) {
    // input_embedding: [1, n] int32 → [1, n, 1024] float32
    TensorData t_ids;
    t_ids.data         = (void*)(ids);
    t_ids.format.dt    = DataType::I32;
    t_ids.format.shape = {1, n};

    _runner["codec_embedding"]->set_input("token_ids", t_ids);
    _runner["codec_embedding"]->run();
    auto t_emb = _runner["codec_embedding"]->get_output("embeddings");

    memcpy(out, t_emb.data, t_emb.format.get_nbytes());
}

void Qwen3TTS::load_speaker_embedding_from_ref_audio(const std::string& path, TensorData& t_speaker_embedding) {

    std::vector<float> mono_raw_wav;
    read_wav(path, mono_raw_wav, _sample_rate);

    // Ensure the input audio length is at least k_min_ref_audio_duration_seconds seconds
    COMPUTE_FLOW_ASSERT(mono_raw_wav.size() >= static_cast<uint64_t>(_sample_rate) * _min_ref_audio_duration_sec);

    // If the reference audio is longer than k_max_ref_audio_duration_seconds, we truncate it.
    const size_t max_ref_samples = static_cast<size_t>(_sample_rate) * _max_ref_audio_duration_sec;
    if (mono_raw_wav.size() > max_ref_samples) {
        mono_raw_wav.resize(max_ref_samples);
    }

    TensorData t_audio;
    t_audio.data = mono_raw_wav.data();
    t_audio.format.dt = DataType::F32;
    t_audio.format.shape = {1, static_cast<int64_t>(mono_raw_wav.size())};

    _runner["mel_spectrogram"]->set_input("audio", t_audio);
    _runner["mel_spectrogram"]->run();
    auto t_mels = _runner["mel_spectrogram"]->get_output("mel_spectrogram");

    _runner["speaker_encoder"]->set_input("mels", t_mels);
    _runner["speaker_encoder"]->run();
    auto t_runtime_embedding = _runner["speaker_encoder"]->get_output("speaker_embedding");

    COMPUTE_FLOW_ASSERT(t_runtime_embedding.format.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(t_runtime_embedding.format.get_nbytes() == t_speaker_embedding.format.get_nbytes());

    memcpy(
        t_speaker_embedding.data,
        t_runtime_embedding.data,
        static_cast<size_t>(t_speaker_embedding.format.get_nbytes()));
}

void Qwen3TTS::icl_load_ref_codes_from_ref_audio_base(const std::string& path) {
    constexpr int64_t k_speech_tokenizer_encoder_min_samples       = 2880;
    constexpr int64_t k_speech_tokenizer_encoder_alignment_samples = 960;
    constexpr int64_t k_speech_tokenizer_encoder_frame_samples     = 1920;

    std::vector<float> mono_raw_wav;
    read_wav(path, mono_raw_wav, _sample_rate);

    COMPUTE_FLOW_ASSERT(mono_raw_wav.size() >= static_cast<uint64_t>(_sample_rate) * _min_ref_audio_duration_sec);

    const size_t max_ref_samples = static_cast<size_t>(_sample_rate) * _max_ref_audio_duration_sec;
    if (mono_raw_wav.size() > max_ref_samples) {
        mono_raw_wav.resize(max_ref_samples);
    }

    const auto encoder_input_fmt = _runner["speech_tokenizer_encoder"]->get_input_format("input_values");
    COMPUTE_FLOW_ASSERT(encoder_input_fmt.dt == DataType::F32);
    COMPUTE_FLOW_ASSERT(encoder_input_fmt.shape.size() == 2);
    COMPUTE_FLOW_ASSERT(encoder_input_fmt.shape[0] == 1);
    COMPUTE_FLOW_ASSERT(encoder_input_fmt.shape[1] > 0);

    const int64_t encoder_input_samples = encoder_input_fmt.shape[1];
    const int64_t total_ref_samples = static_cast<int64_t>(mono_raw_wav.size());

    _num_ref_code_frames = ceil_div(total_ref_samples, k_speech_tokenizer_encoder_frame_samples);
    COMPUTE_FLOW_ASSERT(_num_ref_code_frames > 0);

    _internal_memory.allocate("ref_codes", TensorFormat{DataType::I32, {_num_ref_code_frames, _num_code_pred_groups}});
    int32_t* dst_codes = _internal_memory.get_data_as<int32_t>("ref_codes");

    int64_t ref_frame_offset = 0;

    for (int64_t sample_offset = 0; sample_offset < total_ref_samples; sample_offset += encoder_input_samples) {
        const int64_t chunk_ref_samples = std::min<int64_t>(encoder_input_samples, total_ref_samples - sample_offset);
        const int64_t padded_samples = ceil_div(
            std::max<int64_t>(chunk_ref_samples, k_speech_tokenizer_encoder_min_samples),
            k_speech_tokenizer_encoder_alignment_samples) * k_speech_tokenizer_encoder_alignment_samples;

        COMPUTE_FLOW_ASSERT(padded_samples <= encoder_input_samples);

        std::vector<float> padded_audio(static_cast<size_t>(encoder_input_samples), 0.0f);
        memcpy(
            padded_audio.data(),
            mono_raw_wav.data() + sample_offset,
            static_cast<size_t>(chunk_ref_samples) * sizeof(float));

        TensorData t_audio;
        t_audio.data = padded_audio.data();
        t_audio.format.dt = DataType::F32;
        t_audio.format.shape = {1, encoder_input_samples};

        _runner["speech_tokenizer_encoder"]->set_input("input_values", t_audio);
        _runner["speech_tokenizer_encoder"]->run();
        auto t_codes = _runner["speech_tokenizer_encoder"]->get_output("codes_t");

        COMPUTE_FLOW_ASSERT(t_codes.format.dt == DataType::I32 || t_codes.format.dt == DataType::I64);
        COMPUTE_FLOW_ASSERT(t_codes.format.shape.size() == 2 || t_codes.format.shape.size() == 3);

        bool transpose_codes = false;
        int64_t output_frames = 0;

        if (t_codes.format.shape.size() == 3) {
            COMPUTE_FLOW_ASSERT(t_codes.format.shape[0] == 1);
            if (t_codes.format.shape[2] == _num_code_pred_groups) {
                output_frames = t_codes.format.shape[1];
            } else {
                COMPUTE_FLOW_ASSERT(t_codes.format.shape[1] == _num_code_pred_groups);
                output_frames = t_codes.format.shape[2];
                transpose_codes = true;
            }
        } else {
            if (t_codes.format.shape[1] == _num_code_pred_groups) {
                output_frames = t_codes.format.shape[0];
            } else {
                COMPUTE_FLOW_ASSERT(t_codes.format.shape[0] == _num_code_pred_groups);
                output_frames = t_codes.format.shape[1];
                transpose_codes = true;
            }
        }

        COMPUTE_FLOW_ASSERT(output_frames > 0);

        const int64_t valid_frames = ceil_div(chunk_ref_samples, k_speech_tokenizer_encoder_frame_samples);
        COMPUTE_FLOW_ASSERT(valid_frames <= output_frames);
        COMPUTE_FLOW_ASSERT(ref_frame_offset + valid_frames <= _num_ref_code_frames);

        for (int64_t frame = 0; frame < valid_frames; ++frame) {
            for (int64_t group = 0; group < _num_code_pred_groups; ++group) {
                const int64_t src_idx = transpose_codes ?
                    (group * output_frames) + frame :
                    (frame * _num_code_pred_groups) + group;

                const int64_t dst_idx = ((ref_frame_offset + frame) * _num_code_pred_groups) + group;
                if (t_codes.format.dt == DataType::I32) {
                    const int32_t* src_codes = static_cast<const int32_t*>(t_codes.data);
                    dst_codes[dst_idx] = src_codes[src_idx];
                } else {
                    const int64_t* src_codes = static_cast<const int64_t*>(t_codes.data);
                    dst_codes[dst_idx] = static_cast<int32_t>(src_codes[src_idx]);
                }
            }
        }

        ref_frame_offset += valid_frames;
    }

    COMPUTE_FLOW_ASSERT(ref_frame_offset == _num_ref_code_frames);
}

void Qwen3TTS::icl_precompute_embeddings_base(const std::string& ref_text) {
    TensorData t_tts_pad_embed     = _internal_memory.get_tensor("tts_pad_embed");
    TensorData t_tts_bos_embed     = _internal_memory.get_tensor("tts_bos_embed");
    TensorData t_tts_eos_embed     = _internal_memory.get_tensor("tts_eos_embed");
    TensorData t_codec_pad_embed   = _internal_memory.get_tensor("codec_pad_embed");
    TensorData t_speaker_embedding = _internal_memory.get_tensor("speaker_embeddings");
    TensorData t_ref_codes         = _internal_memory.get_tensor("ref_codes");

    const std::string normalized_ref_text = ensure_prompt_ends_with_punct(ref_text);

    TensorData t_prompt;
    t_prompt.data = const_cast<char*>(normalized_ref_text.c_str());
    t_prompt.format.dt = DataType::String;
    t_prompt.format.shape = { static_cast<int64_t>(normalized_ref_text.size()) };

    _runner["tokenizer"]->set_input("text_prompt", t_prompt);
    _runner["tokenizer"]->run();
    auto t_ref_text_tokens = _runner["tokenizer"]->get_output("token_ids");

    COMPUTE_FLOW_ASSERT(t_ref_text_tokens.format.dt == DataType::I32);
    _ref_text_token_len = t_ref_text_tokens.format.get_num_elems();
    COMPUTE_FLOW_ASSERT(_ref_text_token_len > 0);

    const int32_t* token_ids = static_cast<const int32_t*>(t_ref_text_tokens.data);
    std::vector<int32_t> ref_text_tokens(token_ids, token_ids + _ref_text_token_len);

    std::vector<int32_t> codec_tokens;
    codec_tokens.reserve(16);
    if (_lang_token_id >= 0) {
        codec_tokens.push_back(_codec_think_token_id);
        codec_tokens.push_back(_codec_think_bos_token_id);
        codec_tokens.push_back(_lang_token_id);
        codec_tokens.push_back(_codec_think_eos_token_id);
    } else {
        codec_tokens.push_back(_codec_no_think_token_id);
        codec_tokens.push_back(_codec_think_bos_token_id);
        codec_tokens.push_back(_codec_think_eos_token_id);
    }
    codec_tokens.push_back(_codec_pad_token_id);
    codec_tokens.push_back(_codec_bos_token_id);

    constexpr int64_t role_len = 3;
    const int64_t codec_len = static_cast<int64_t>(codec_tokens.size());

    _icl_prefill_prefix_len = role_len + codec_len + _ref_text_token_len;
    _icl_prefill_suffix_len = 2 + _num_ref_code_frames;

    _internal_memory.allocate(
        "icl_prefill_prefix_embeds",
        TensorFormat{DataType::F32, {_icl_prefill_prefix_len, _k_n_embd_sz}});
    _internal_memory.allocate(
        "icl_prefill_suffix_embeds",
        TensorFormat{DataType::F32, {_icl_prefill_suffix_len, _k_n_embd_sz}});

    float* prefix = _internal_memory.get_data_as<float>("icl_prefill_prefix_embeds");
    float* suffix = _internal_memory.get_data_as<float>("icl_prefill_suffix_embeds");
    int64_t pos_embedding = 0;

    const int32_t role_tokens[role_len] = {_im_start_token_id, _assistant_token_id, _newline_token_id};
    run_text_embedding(role_tokens, role_len, prefix + pos_embedding);
    pos_embedding += role_len * _k_n_embd_sz;

    float* codec_embeds = prefix + pos_embedding;
    const int64_t pre_speaker_len = codec_len - 2;
    float* codec_write_ptr = codec_embeds;

    if(pre_speaker_len > 0) {
        run_codec_embedding(codec_tokens.data(), pre_speaker_len, codec_write_ptr);
        codec_write_ptr += pre_speaker_len * _k_n_embd_sz;
    }

    memcpy(codec_write_ptr, t_speaker_embedding.data, t_speaker_embedding.format.get_nbytes());
    codec_write_ptr += _k_n_embd_sz;
    memcpy(codec_write_ptr, t_codec_pad_embed.data, t_codec_pad_embed.format.get_nbytes());

    add_special_embedding((float*)t_tts_pad_embed.data, codec_len - 1, _k_n_embd_sz, codec_embeds);
    add_special_embedding((float*)t_tts_bos_embed.data, 1, _k_n_embd_sz,
                          codec_embeds + ((codec_len - 1) * _k_n_embd_sz));
    pos_embedding += codec_len * _k_n_embd_sz;

    run_text_embedding(ref_text_tokens.data(), _ref_text_token_len, prefix + pos_embedding);
    add_special_embedding((float*)t_codec_pad_embed.data,
                          _ref_text_token_len,
                          _k_n_embd_sz,
                          prefix + pos_embedding);
    pos_embedding += _ref_text_token_len * _k_n_embd_sz;

    COMPUTE_FLOW_ASSERT(pos_embedding == _icl_prefill_prefix_len * _k_n_embd_sz);

    pos_embedding = 0;
    run_codec_embedding(&_codec_pad_token_id, 1, suffix + pos_embedding);
    add_special_embedding((float*)t_tts_eos_embed.data, 1, _k_n_embd_sz, suffix + pos_embedding);
    pos_embedding += _k_n_embd_sz;

    run_codec_embedding(&_codec_bos_token_id, 1, suffix + pos_embedding);
    add_special_embedding((float*)t_tts_pad_embed.data, 1, _k_n_embd_sz, suffix + pos_embedding);
    pos_embedding += _k_n_embd_sz;

    const int32_t* ref_codes = static_cast<const int32_t*>(t_ref_codes.data);
    for (int64_t frame = 0; frame < _num_ref_code_frames; ++frame) {
        float* frame_embedding = suffix + pos_embedding;
        const int32_t code0 = ref_codes[frame * _num_code_pred_groups];
        run_codec_embedding(&code0, 1, frame_embedding);

        for (int64_t group = 1; group < _num_code_pred_groups; ++group) {
            int32_t token_id = ref_codes[(frame * _num_code_pred_groups) + group];
            int32_t group_idx = static_cast<int32_t>(group);

            TensorData t_token_id;
            t_token_id.data = &token_id;
            t_token_id.format.dt = DataType::I32;
            t_token_id.format.shape = {1};

            TensorData t_group_idx;
            t_group_idx.data = &group_idx;
            t_group_idx.format.dt = DataType::I32;
            t_group_idx.format.shape = {1};

            _runner["code_pred_group_embeds"]->set_input("token_id", t_token_id);
            _runner["code_pred_group_embeds"]->set_input("group_idx", t_group_idx);
            _runner["code_pred_group_embeds"]->run();
            auto t_group_embedding = _runner["code_pred_group_embeds"]->get_output("embeddings");

            COMPUTE_FLOW_ASSERT(t_group_embedding.format.dt == DataType::F32);
            COMPUTE_FLOW_ASSERT(t_group_embedding.format.get_num_elems() == _k_n_embd_sz);

            add_special_embedding(
                static_cast<const float*>(t_group_embedding.data),
                1,
                _k_n_embd_sz,
                frame_embedding);
        }
        add_special_embedding((float*)t_tts_pad_embed.data, 1, _k_n_embd_sz, frame_embedding);

        pos_embedding += _k_n_embd_sz;
    }

    COMPUTE_FLOW_ASSERT(pos_embedding == _icl_prefill_suffix_len * _k_n_embd_sz);
}

void Qwen3TTS::icl_precompute_talker_prefix_state_base() {
    TensorData t_icl_prefix = _internal_memory.get_tensor("icl_prefill_prefix_embeds");

    // This prefix contains only reference-dependent ICL context, so it can be evaluated once in
    // prepare() and restored before each request-specific prompt.
    _lm_runner["talker"]->reset_states();
    _lm_runner["talker"]->set_input("inputs_embeds", t_icl_prefix);
    _lm_runner["talker"]->run();
    _lm_runner["talker"]->save_sequence_state(kIclTalkerPrefixStateName);
}

void Qwen3TTS::icl_precompute_speech_decoder_state_base(TensorData& t_codes,
                                                        TensorData t_audio_out,
                                                        int64_t max_output_audio_samples) {
    TensorData t_ref_codes = _internal_memory.get_tensor("ref_codes");
    const int32_t* ref_codes = static_cast<const int32_t*>(t_ref_codes.data);
    int32_t* chunk_codes = static_cast<int32_t*>(t_codes.data);

    // ICL conditions the talker on reference speech codes. The speech decoder also needs that
    // history: otherwise the first generated chunk is decoded from zero state and can contain a
    // small start artifact. Prime the decoder with the reference codes and discard the produced
    // reference audio so generated streaming output starts with the expected decoder state.
    _lm_runner["speech_decoder_transformer"]->reset_states();
    for (int64_t frame_offset = 0; frame_offset < _num_ref_code_frames; frame_offset += _num_max_frames_speech_dec) {
        const int64_t chunk_frames = std::min<int64_t>(
            _num_max_frames_speech_dec,
            _num_ref_code_frames - frame_offset);
        const int64_t chunk_codes_count = chunk_frames * _num_code_pred_groups;

        memcpy(
            chunk_codes,
            ref_codes + (frame_offset * _num_code_pred_groups),
            static_cast<size_t>(chunk_codes_count) * sizeof(int32_t));

        const int64_t pad_codes_count = (_num_max_frames_speech_dec - chunk_frames) * _num_code_pred_groups;
        if(pad_codes_count > 0) {
            std::fill_n(chunk_codes + chunk_codes_count, pad_codes_count, 0);
        }

        run_speech_decoder(t_codes, chunk_frames, t_audio_out, max_output_audio_samples, true);
    }

    _internal_memory.allocate("icl_primed_pre_conv_state", _t_state_pre_conv.format);
    _internal_memory.allocate("icl_primed_latent_state", _t_state_latent.format);

    TensorData t_primed_pre_conv = _internal_memory.get_tensor("icl_primed_pre_conv_state");
    TensorData t_primed_latent = _internal_memory.get_tensor("icl_primed_latent_state");
    memcpy(t_primed_pre_conv.data, _t_state_pre_conv.data, _t_state_pre_conv.format.get_nbytes());
    memcpy(t_primed_latent.data, _t_state_latent.data, _t_state_latent.format.get_nbytes());

    _lm_runner["speech_decoder_transformer"]->save_sequence_state(kIclSpeechDecoderStateName);
}

void Qwen3TTS::icl_trim_initial_audio_base(TensorData& t_audio_out) {
    COMPUTE_FLOW_ASSERT(t_audio_out.format.dt == DataType::F32);

    const int64_t num_samples = t_audio_out.format.get_num_elems();
    if(num_samples <= 0) {
        return;
    }

    // The first returned ICL chunk starts at a synthetic boundary: reference speech was decoded
    // only to prime state, not emitted. Drop the tiny leading transition region without changing
    // token generation, decoder state, or later streaming chunks.
    constexpr int64_t k_trim_duration_ms = 20;
    const int64_t trim_samples = std::min<int64_t>(num_samples, (_sample_rate * k_trim_duration_ms) / 1000);
    t_audio_out.data = static_cast<float*>(t_audio_out.data) + trim_samples;
    t_audio_out.format.shape[0] = num_samples - trim_samples;
}

void Qwen3TTS::run_code_predictor(const TensorData& t_last_talker_hidden, TensorData& t_codes, TensorData& t_step_embd) {

    int32_t* codes   = ((int32_t*)(t_codes.data));
    float* step_embd = ((float*)(t_step_embd.data));

    COMPUTE_FLOW_ASSERT(_num_active_code_pred_groups > 0);
    COMPUTE_FLOW_ASSERT(_num_active_code_pred_groups <= _num_code_pred_groups);

    if (_num_active_code_pred_groups < _num_code_pred_groups) {
        std::fill(codes + _num_active_code_pred_groups, codes + _num_code_pred_groups, 0);
    }

    if (_num_active_code_pred_groups == 1) {
        run_codec_embedding(&codes[0], 1, step_embd);
        return;
    }

    _lm_runner["code_pred"]->reset_states();

    // Pos0 ----------------------------------------------------- s
    _lm_runner["code_pred"]->set_input("inputs_embeds", t_last_talker_hidden);
    _lm_runner["code_pred"]->run();
    // Pos0 ----------------------------------------------------- e

    // Pos1 ----------------------------------------------------- s
    run_codec_embedding(&codes[0], 1, step_embd);
    _lm_runner["code_pred"]->set_input("inputs_embeds", t_step_embd);
    _lm_runner["code_pred"]->run();
    auto t_hidden = _lm_runner["code_pred"]->get_output("hidden_state");

    _runner["code_pred_post_1"]->set_input("code_t", t_hidden);
    _runner["code_pred_post_1"]->run();

    auto t_token_id_1 = _runner["code_pred_post_1"]->get_output("token_id");
    auto t_group_embd = _runner["code_pred_post_1"]->get_output("embeddings");

    COMPUTE_FLOW_ASSERT(t_token_id_1.format.dt == DataType::I32);
    codes[1] = *static_cast<const int32_t*>(t_token_id_1.data);

    custom_op_in_place_add(t_group_embd, t_step_embd);
    // Pos1 ----------------------------------------------------- e

    // PosN ----------------------------------------------------- s
    for (int64_t pos = 2; pos < _num_active_code_pred_groups; ++pos) {
        _lm_runner["code_pred"]->set_input("inputs_embeds", t_group_embd);
        _lm_runner["code_pred"]->run();
        t_hidden = _lm_runner["code_pred"]->get_output("hidden_state");

        const std::string code_pred_heads_name = "code_pred_post_" + std::to_string(pos);
        _runner[code_pred_heads_name]->set_input("code_t", t_hidden);
        _runner[code_pred_heads_name]->run();

        auto t_token_id = _runner[code_pred_heads_name]->get_output("token_id");
        t_group_embd    = _runner[code_pred_heads_name]->get_output("embeddings");

        COMPUTE_FLOW_ASSERT(t_token_id.format.dt == DataType::I32);
        codes[pos] = *static_cast<const int32_t*>(t_token_id.data);

        custom_op_in_place_add(t_group_embd, t_step_embd);
    }
    // PosN ----------------------------------------------------- e
}

void Qwen3TTS::run_speech_decoder(const TensorData& t_codes, int64_t num_frames_processed, TensorData& t_audio_out, int64_t max_output_audio_samples, bool en_streaming_mode) {

    // ── Stage 1 — Pre speech tokenizer ────────────────────────────────────────────────
    _runner["speech_decoder_pre"]->set_input("codes_t",          t_codes);
    _runner["speech_decoder_pre"]->set_input("pre_conv_history", _t_state_pre_conv);
    _runner["speech_decoder_pre"]->run();

    auto t_hidden       = _runner["speech_decoder_pre"]->get_output("hidden");
    auto t_new_pre_conv = _runner["speech_decoder_pre"]->get_output("new_pre_conv_history");
    _t_state_pre_conv = t_new_pre_conv;

    // ── Stage 2 — Transformer  ───────────────────────
    // Reshape hidden from [1, T, C] to [T, C] for the LM runner (2-D embedding input).
    const auto t_main_in_fmt = _lm_runner["speech_decoder_transformer"]->get_input_format("inputs_embeds");
    const int64_t expected_embd = t_main_in_fmt.shape.back();
    COMPUTE_FLOW_ASSERT(t_hidden.format.shape.size() == 3);
    COMPUTE_FLOW_ASSERT(t_hidden.format.shape[2] == expected_embd);

    TensorData t_embeds;
    t_embeds.data         = t_hidden.data;
    t_embeds.format.dt    = t_hidden.format.dt;
    t_embeds.format.shape = {t_hidden.format.shape[1], t_hidden.format.shape[2]};

    _lm_runner["speech_decoder_transformer"]->set_input("inputs_embeds", t_embeds);
    _lm_runner["speech_decoder_transformer"]->run();

    auto t_transformer_out = _lm_runner["speech_decoder_transformer"]->get_output("hidden_state_seq");

    // The LM runner returns [T, C], while the post model expects [1, T, C].
    TensorData t_transformer_out_3d = t_transformer_out;
    if (t_transformer_out.format.shape.size() == 2) {
        t_transformer_out_3d.format.shape = {
            1,
            t_transformer_out.format.shape[0],
            t_transformer_out.format.shape[1]
        };
    } else if (t_transformer_out.format.shape.size() == 3) {
        t_transformer_out_3d = t_transformer_out;
    } else {
        COMPUTE_FLOW_ASSERT(false);
    }

    // ── Stage 3 — Post speech tokenizer (LiteRT) ───────────────────────────────────────────────
    // Inputs: transformer_out [1,16,512] f32, latent_buffer [1,1024,25] f32
    // Outputs: padded_audio [65536] f32, valid_samples [1] int64, new_latent_buffer [1,1024,25] f32
    _runner["speech_decoder_post"]->set_input("transformer_out", t_transformer_out_3d);
    _runner["speech_decoder_post"]->set_input("latent_buffer",   _t_state_latent);
    _runner["speech_decoder_post"]->run();

    auto t_padded_audio  = _runner["speech_decoder_post"]->get_output("padded_audio");
    auto t_valid_samples = _runner["speech_decoder_post"]->get_output("valid_samples");
    auto t_new_latent    = _runner["speech_decoder_post"]->get_output("new_latent_buffer");

    _t_state_latent = t_new_latent;

    // The post model outputs audio with CNN context from previous chunks prepended, followed by
    // exactly k_chunk_size frames of new audio. For a partial (leftover) chunk, only the first
    // num_frames_processed frames are real speech; the trailing zero-padded frames must be excluded.
    // Correct slice: skip CNN context, then take the first num_frames_processed frames' worth of samples.
    const int64_t k_chunk_size            = _num_max_frames_speech_dec;
    const int64_t num_valid_samples       = *static_cast<const int64_t*>(t_valid_samples.data);
    const int64_t num_cnn_context_samples = std::max<int64_t>(0, num_valid_samples - (k_chunk_size * _num_audio_samples_per_frame));
    const int64_t num_new_samples         = std::min(num_valid_samples - num_cnn_context_samples,
                                                 (num_frames_processed * _num_audio_samples_per_frame));
    const int64_t start_idx               = num_cnn_context_samples;
    void* t_wav_start_ptr   = (void*)((float*)(t_padded_audio.data) + start_idx);

    if(en_streaming_mode) {
        t_audio_out.data         = static_cast<float*>(t_padded_audio.data) + start_idx;
        t_audio_out.format.dt    = DataType::F32;
        t_audio_out.format.shape = {num_new_samples};
    } else {
        COMPUTE_FLOW_ASSERT(t_audio_out.format.shape[0] + num_new_samples < max_output_audio_samples);

        memcpy(
            (float*)(t_audio_out.data) + t_audio_out.format.shape[0],
            t_wav_start_ptr,
            num_new_samples * sizeof(float));

        t_audio_out.format.shape[0] += num_new_samples;
    }

}

void Qwen3TTS::add_special_embedding(const float* special_embedding, int64_t num_tokens, int64_t hidden, float* out) {
    for(int64_t num_tokens_idx = 0; num_tokens_idx < num_tokens; ++num_tokens_idx) {
        for(int64_t hidden_idx = 0; hidden_idx < hidden; ++hidden_idx) {
            out[hidden_idx + (num_tokens_idx * hidden)] = out[hidden_idx + (num_tokens_idx * hidden)] + special_embedding[hidden_idx];
        }
    }
}
} // namespace compute_flow::bricks::tts
