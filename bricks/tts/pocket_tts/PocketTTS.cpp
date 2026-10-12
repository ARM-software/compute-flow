//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "PocketTTS.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <npy.hpp>

namespace compute_flow::bricks::tts {

constexpr int32_t k_brick_major_version = 1;
constexpr int64_t k_fade_out_ms         = 10;
constexpr float   k_pi                  = 3.14159265358979323846f;

static void apply_cosine_fade_out(float* audio, int64_t num_audio_samples, int64_t num_fade_samples) {
    if(audio == nullptr || num_audio_samples <= 0 || num_fade_samples <= 0) {
        return;
    }

    const int64_t fade_samples = std::min(num_audio_samples, num_fade_samples);
    const int64_t fade_start   = num_audio_samples - fade_samples;

    if(fade_samples == 1) {
        audio[fade_start] = 0.0f;
        return;
    }

    for(int64_t i = 0; i < fade_samples; ++i) {
        const float phase = static_cast<float>(i) / static_cast<float>(fade_samples - 1);
        const float gain  = 0.5f * (1.0f + std::cos(k_pi * phase));
        audio[fade_start + i] *= gain;
    }
}

static void load_embeddings_from_file(const std::string& path, TensorData& bos_embedding) {
    // Load embeddings from a .npy file
    npy::npy_data d = npy::read_npy<float>(path);
    COMPUTE_FLOW_ASSERT(d.shape[0] == bos_embedding.format.shape[0]);
    COMPUTE_FLOW_ASSERT(d.shape[1] == bos_embedding.format.shape[1]);
    COMPUTE_FLOW_ASSERT(d.shape[2] == bos_embedding.format.shape[2]);

    memcpy(bos_embedding.data, d.data.data(), bos_embedding.format.get_nbytes());
}

static std::string normalize_unicode_punctuation(const std::string& src) {
    std::string dst;
    dst.reserve(src.size());

    for(size_t idx = 0; idx < src.size();) {
        if(src.compare(idx, 2, "\xC2\xA0") == 0) {
            dst += ' ';
            idx += 2;
            continue;
        }

        if(src.compare(idx, 3, "\xE2\x80\xA6") == 0) {
            dst += "...";
            idx += 3;
            continue;
        }

        if(src.compare(idx, 3, "\xE2\x80\x98") == 0 ||
           src.compare(idx, 3, "\xE2\x80\x99") == 0 ||
           src.compare(idx, 3, "\xE2\x80\x9A") == 0 ||
           src.compare(idx, 3, "\xE2\x80\x9B") == 0) {
            dst += '\'';
            idx += 3;
            continue;
        }

        if(src.compare(idx, 3, "\xE2\x80\x9C") == 0 ||
           src.compare(idx, 3, "\xE2\x80\x9D") == 0 ||
           src.compare(idx, 3, "\xE2\x80\x9E") == 0 ||
           src.compare(idx, 3, "\xE2\x80\x9F") == 0) {
            dst += '"';
            idx += 3;
            continue;
        }

        if(src.compare(idx, 3, "\xE2\x80\x90") == 0 ||
           src.compare(idx, 3, "\xE2\x80\x91") == 0 ||
           src.compare(idx, 3, "\xE2\x80\x92") == 0 ||
           src.compare(idx, 3, "\xE2\x80\x93") == 0 ||
           src.compare(idx, 3, "\xE2\x80\x94") == 0 ||
           src.compare(idx, 3, "\xE2\x88\x92") == 0) {
            dst += '-';
            idx += 3;
            continue;
        }

        if(src.compare(idx, 3, "\xE2\x80\x89") == 0 ||
           src.compare(idx, 3, "\xE2\x80\xAF") == 0) {
            dst += ' ';
            idx += 3;
            continue;
        }

        dst += src[idx++];
    }

    return dst;
}

void PocketTTS::custom_op_fill_random_box_muller_transform_f32(const TensorData& dst) {
    // Box-Muller transform, with adjacent cosine/sine samples and double intermediates.
    constexpr double k_full_circle = 6.283185307179586476925286766559;
    float* output = static_cast<float*>(dst.data);
    const uint64_t num_values = get_num_elems(dst.format.shape);
    uint64_t position = 0;

    while (position < num_values) {
        // 1-U lies in (0, 1], so the logarithm is defined even when U is zero.
        const double survival = 1.0 - std::generate_canonical<double, 53>(_rng);
        const double angle = k_full_circle * std::generate_canonical<double, 53>(_rng);
        const double amplitude = static_cast<double>(_rng_std) * std::sqrt(-2.0 * std::log(survival));

        output[position++] = static_cast<float>(_rng_mean + amplitude * std::cos(angle));

        // An odd-length call discards the final sine sample.
        if (position < num_values) {
            output[position++] = static_cast<float>(_rng_mean + amplitude * std::sin(angle));
        }
    }
}

static int64_t preprocess_prompt(const std::string& src, std::string& dst) {
    int64_t num_words = 0;
    std::string source_without_line_breaks;
    const std::string normalized_src = normalize_unicode_punctuation(src);

    // Remove line-break markers before preserving the original stream-based preprocessing below.
    for(size_t idx = 0; idx < normalized_src.size(); ++idx) {
        const char c = normalized_src[idx];
        if((c == '/' || c == '\\') && idx + 1 < normalized_src.size() &&
           (normalized_src[idx + 1] == 'n' || normalized_src[idx + 1] == 'r')) {
            source_without_line_breaks += ' ';
            ++idx;
            continue;
        }

        if(c == '\n' || c == '\r') {
            source_without_line_breaks += ' ';
            continue;
        }

        source_without_line_breaks += c;
    }

    // Count number of words and clean spaces in the input text.
    dst.clear();
    std::istringstream stream(source_without_line_breaks);
    std::string word;

    while (stream >> word) {
        ++num_words;
        dst += word + " ";
    }

    // Strip whitespace from the beginning and the end of the text.
    std::string normalized_text = dst;
    if(!normalized_text.empty() && normalized_text.back() == ' ') {
        normalized_text.pop_back();
    }

    // Make sure it starts with an uppercase letter (if the first character is a letter).
    if(!normalized_text.empty() && std::isalpha(static_cast<unsigned char>(normalized_text[0]))) {
        normalized_text[0] =
            static_cast<char>(std::toupper(static_cast<unsigned char>(normalized_text[0])));
    }

    auto is_prompt_punctuation = [](char c) {
        return c == ',' || c == '.' || c == '!' || c == '?' || c == ':' || c == ';';
    };

    std::string punctuation_normalized_text;
    for(size_t idx = 0; idx < normalized_text.size(); ++idx) {
        const char c = normalized_text[idx];
        if(is_prompt_punctuation(c)) {
            while(!punctuation_normalized_text.empty() &&
                  punctuation_normalized_text.back() == ' ') {
                punctuation_normalized_text.pop_back();
            }
            punctuation_normalized_text += c;
            if(idx + 1 < normalized_text.size() && normalized_text[idx + 1] != ' ' &&
               !is_prompt_punctuation(normalized_text[idx + 1])) {
                punctuation_normalized_text += ' ';
            }
            continue;
        }

        punctuation_normalized_text += c;
    }
    normalized_text = punctuation_normalized_text;

    // Make sure the prompt ends with a punctuation. If not, we add a period at the end.
    // If it ends with a letter or a digit, we add a period at the end.
    if (!normalized_text.empty()) {
        char last_char = normalized_text.back();
        if (std::isalnum(static_cast<unsigned char>(last_char))) {
            normalized_text += ".";
        }
    }

    bool capitalize_next = true;
    for(char& c : normalized_text) {
        if(std::isalpha(static_cast<unsigned char>(c))) {
            if(capitalize_next) {
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            }
            capitalize_next = false;
            continue;
        }

        if(c == '.' || c == '!' || c == '?') {
            capitalize_next = true;
        }
    }

    // The model does not perform well with small number of tokens.
    // To mitigate this, we can add some empty spaces at the beginning of the prompt.
    if(num_words < 5) {
        normalized_text = std::string(8, ' ') + normalized_text;
    }

    dst = normalized_text;

    return num_words;
}

static int64_t get_num_frames_after_eos(int64_t num_words) {
    // Find number of frames after EOS based on number of words in the input text
    int64_t frames_after_eos_guess = 2;

    if(num_words <= 4) {
        frames_after_eos_guess += 3; // short text
    } else {
        frames_after_eos_guess += 1; // long text
    }

    return frames_after_eos_guess;
}

ValidationResult PocketTTS::validate(const ComputeFlowManifest& manifest) const {
    const BrickInfo& info = manifest.get_brick_info();
    COMPUTE_FLOW_VALIDATE(info.type == "tts");
    COMPUTE_FLOW_VALIDATE(info.name == "PocketTTS");
    COMPUTE_FLOW_VALIDATE(info.version.major == k_brick_major_version);

    COMPUTE_FLOW_VALIDATE(manifest.num_modules() == 6);

    COMPUTE_FLOW_VALIDATE(manifest.has_module("spm"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("text_encoder"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("flowlm"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("simplemlpadaln"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("latent_proj"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("mimi_decoder"));

    // Validate presence of the model path
    COMPUTE_FLOW_VALIDATE(manifest["spm"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["text_encoder"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["flowlm"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["simplemlpadaln"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["latent_proj"].model_path != "");
    COMPUTE_FLOW_VALIDATE(manifest["mimi_decoder"].model_path != "");

    // Validate the target framework.
    COMPUTE_FLOW_VALIDATE(manifest["spm"].framework_type == RunnerType::SentencePieceEncode);
    COMPUTE_FLOW_VALIDATE(manifest["text_encoder"].framework_type == RunnerType::ExecuTorch ||
                     manifest["text_encoder"].framework_type == RunnerType::ONNXRT);
    COMPUTE_FLOW_VALIDATE(manifest["flowlm"].lm_runner_type == LMRunnerType::Llama_cpp);
    COMPUTE_FLOW_VALIDATE(manifest["simplemlpadaln"].framework_type == RunnerType::ExecuTorch ||
                     manifest["simplemlpadaln"].framework_type == RunnerType::ONNXRT);
    COMPUTE_FLOW_VALIDATE(manifest["latent_proj"].framework_type == RunnerType::ExecuTorch ||
                     manifest["latent_proj"].framework_type == RunnerType::ONNXRT);
    COMPUTE_FLOW_VALIDATE(manifest["mimi_decoder"].framework_type == RunnerType::ExecuTorch ||
                     manifest["mimi_decoder"].framework_type == RunnerType::ONNXRT ||
                     manifest["mimi_decoder"].framework_type == RunnerType::LiteRT);

    // Check whether the models only have 1 accelerator type
    // At the moment, only the CPU is supported
    COMPUTE_FLOW_VALIDATE(manifest["spm"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["text_encoder"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["flowlm"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["simplemlpadaln"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["latent_proj"].acc_config.priorities.size() == 1);
    COMPUTE_FLOW_VALIDATE(manifest["mimi_decoder"].acc_config.priorities.size() == 1);

    // Only CPU is supported as accelerator type
    COMPUTE_FLOW_VALIDATE(manifest["text_encoder"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["spm"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["flowlm"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["simplemlpadaln"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["latent_proj"].acc_config.has_accelerator(AcceleratorType::Cpu));
    COMPUTE_FLOW_VALIDATE(manifest["mimi_decoder"].acc_config.has_accelerator(AcceleratorType::Cpu));

    // Validate presence of required metadata
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("seed"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("rng_std"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("num_max_frames"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("rng_mean"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("eos_threshold"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("enable_streaming_mode"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("sample_rate"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("trailing_silence_ms"));

    // Validate presence of required asset
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("voice_files_path"));
    COMPUTE_FLOW_VALIDATE(manifest.has_asset("bos_embedding_file"));
    COMPUTE_FLOW_VALIDATE(manifest.has_config("voice"));

    const int64_t seed = manifest.get_constant<int64_t>("seed");
    const float rng_std = manifest.get_constant<float>("rng_std");

    COMPUTE_FLOW_VALIDATE(seed >= 0);
    COMPUTE_FLOW_VALIDATE(std::isfinite(rng_std) && rng_std >= 0.0f);

    return {};
}

void PocketTTS::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    const std::string voice            = manifest.get_config<std::string>("voice");
    const std::string voice_files_path = *manifest.get_asset("voice_files_path");
    const std::filesystem::path voice_file =
        std::filesystem::path(voice_files_path) / (voice + ".npy");

    const npy::npy_data d = npy::read_npy<float>(voice_file.string());

    COMPUTE_FLOW_ASSERT(d.shape.size() == 3);

    const TensorFormat voice_style_format =
        get_flowlm_conditioning_format(static_cast<int64_t>(d.shape[1]));

    COMPUTE_FLOW_ASSERT(voice_style_format.shape[2] == d.shape[2]);

    auto t_voice_style = _internal_memory.get_tensor("voice_style");

    if(t_voice_style.format.shape != voice_style_format.shape) {
        _internal_memory.release("voice_style");
        _internal_memory.allocate("voice_style", voice_style_format);
        t_voice_style = _internal_memory.get_tensor("voice_style");
    }

    COMPUTE_FLOW_ASSERT(t_voice_style.format.dt == voice_style_format.dt);
    COMPUTE_FLOW_ASSERT(t_voice_style.format.shape == voice_style_format.shape);

    memcpy(t_voice_style.data, d.data.data(), voice_style_format.get_nbytes());

}

void PocketTTS::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    RunnerType spm_runner_type            = manifest["spm"].framework_type;
    RunnerType text_encoder_runner_type   = manifest["text_encoder"].framework_type;
    RunnerType simplemlpadaln_runner_type = manifest["simplemlpadaln"].framework_type;
    RunnerType latent_proj_runner_type    = manifest["latent_proj"].framework_type;
    RunnerType mimi_decoder_runner_type   = manifest["mimi_decoder"].framework_type;

    _runner["spm"]            = make_runner(spm_runner_type);
    _runner["text_encoder"]   = make_runner(text_encoder_runner_type);
    _runner["simplemlpadaln"] = make_runner(simplemlpadaln_runner_type);
    _runner["latent_proj"]    = make_runner(latent_proj_runner_type);
    _runner["mimi_decoder"]   = make_runner(mimi_decoder_runner_type);
    _lm_runner["flowlm"]      = make_lm_runner(manifest["flowlm"].lm_runner_type);

    _runner["spm"]->setup(manifest["spm"], profiler);
    _runner["text_encoder"]->setup(manifest["text_encoder"], profiler);
    _runner["simplemlpadaln"]->setup(manifest["simplemlpadaln"], profiler);
    _runner["latent_proj"]->setup(manifest["latent_proj"], profiler);
    _runner["mimi_decoder"]->setup(manifest["mimi_decoder"], profiler);
    _lm_runner["flowlm"]->setup(manifest["flowlm"], profiler);
}

void PocketTTS::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _seed                             = manifest.get_constant<int64_t>("seed");
    _rng_std                          = manifest.get_constant<float>("rng_std");
    _num_max_frames                   = manifest.get_constant<int64_t>("num_max_frames");
    _rng_mean                         = manifest.get_constant<float>("rng_mean");
    _eos_threshold                    = manifest.get_constant<float>("eos_threshold");
    _enable_streaming_mode            = manifest.get_constant<bool>("enable_streaming_mode");
    _sample_rate                      = manifest.get_constant<int64_t>("sample_rate");
    const int64_t trailing_silence_ms = manifest.get_constant<int64_t>("trailing_silence_ms");

    const int64_t num_trailing_silence_samples =
        (static_cast<int64_t>(_sample_rate) * trailing_silence_ms) / 1000;
    _fade_out_samples = (static_cast<int64_t>(_sample_rate) * k_fade_out_ms) / 1000;

    const std::string bos_embedding_file = *manifest.get_asset("bos_embedding_file");

    TensorFormat noise_format         = _runner["simplemlpadaln"]->get_input_format("noise");
    TensorFormat bos_embedding_format = get_flowlm_conditioning_format(1);

    _internal_memory.allocate("noise", noise_format);
    _internal_memory.allocate("bos_embedding", bos_embedding_format);
    _internal_memory.allocate("trailing_silence", TensorFormat{ DataType::F32, { num_trailing_silence_samples } });

    auto t_trailing_silence = _internal_memory.get_tensor("trailing_silence");
    memset(t_trailing_silence.data, 0, t_trailing_silence.format.get_nbytes());

    auto t_bos_embedding = _internal_memory.get_tensor("bos_embedding");
    load_embeddings_from_file(bos_embedding_file, t_bos_embedding);

    const std::string voice            = manifest.get_config<std::string>("voice");
    const std::string voice_files_path = *manifest.get_asset("voice_files_path");
    const std::filesystem::path voice_file =
        std::filesystem::path(voice_files_path) / (voice + ".npy");

    const npy::npy_data d = npy::read_npy<float>(voice_file.string());

    COMPUTE_FLOW_ASSERT(d.shape.size() == 3);

    const TensorFormat voice_style_format =
        get_flowlm_conditioning_format(static_cast<int64_t>(d.shape[1]));

    COMPUTE_FLOW_ASSERT(voice_style_format.shape[2] == d.shape[2]);

    _internal_memory.allocate("voice_style", voice_style_format);

    update_config_impl(manifest, profiler);
}

void PocketTTS::warmup_impl(ProfilerView profiler) {
    (void)profiler;
}

TensorDict PocketTTS::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)profiler;
    if(_enable_streaming_mode) {
        return run_inference_streaming(input);
    } else {
        return run_inference(input);
    }
}

void PocketTTS::reset_states_impl(ProfilerView profiler) {
    // Reset the states
    for (auto& [name, runner] : _runner) {
        runner->reset_states();
    }

    for (auto& [name, runner] : _lm_runner) {
        runner->reset_states();
    }

    // Clear all past audio frames
    _generated_audio_frames.clear();

    // Reset the output ready flag
    _is_output_ready = false;

    // Reset the first run streaming mode flag
    _is_first_run_streaming_mode = true;

    // Reset the pending final silence flag
    _pending_trailing_silence = false;

    // Reset the last generated audio chunk tracked for the streaming fade-out.
    _last_audio_chunk_data        = nullptr;
    _last_audio_chunk_num_samples = 0;
    _last_audio_chunk_faded       = false;
}

bool PocketTTS::is_generation_complete() const {
    return _is_output_ready;
}

TensorDict PocketTTS::run_inference(const TensorDict& input) {

    _rng.seed(_seed);

    // Preprocess text prompt
    const std::string txt_prompt((const char*)input["text_prompt"].data);
    std::string       preprocessed_txt_prompt;

    const int64_t num_words            = preprocess_prompt(txt_prompt, preprocessed_txt_prompt);
    const int64_t num_frames_after_eos = get_num_frames_after_eos(num_words);

    TensorData t_preprocessed_prompt;
    t_preprocessed_prompt.data = (void *)preprocessed_txt_prompt.c_str();
    t_preprocessed_prompt.format.dt = DataType::String;
    t_preprocessed_prompt.format.shape = { (int64_t)preprocessed_txt_prompt.size() };

    // START SPM ---------------------------------
    _runner["spm"]->set_input("text_prompt", t_preprocessed_prompt);

    _runner["spm"]->run();

    auto t_spm_out = _runner["spm"]->get_output("ids");
    // END SPM ---------------------------------

    int64_t num_token_ids = t_spm_out.format.shape[0];

    _internal_memory.allocate("token_ids_i64", TensorFormat{ DataType::I64, { 1, num_token_ids } });
    auto t_token_ids_i64 = _internal_memory.get_tensor("token_ids_i64");

    // Convert the SPM output to I64
    auto* dst = static_cast<int64_t*>(t_token_ids_i64.data);
    auto* src = static_cast<const int32_t*>(t_spm_out.data);

    for (int64_t i = 0; i < num_token_ids; ++i) {
        dst[i] = static_cast<int64_t>(src[i]);
    }

    // START Text encoder ---------------------------------
    _runner["text_encoder"]->set_input("token_ids", t_token_ids_i64);
    _runner["text_encoder"]->run();

    auto t_text_encoder_out = _runner["text_encoder"]->get_output("embeddings");
    // END Text encoder ---------------------------------

    // Get the voice style and bos embedding tensors
    auto t_voice_style   = _internal_memory.get_tensor("voice_style");
    auto t_bos_embedding = _internal_memory.get_tensor("bos_embedding");
    auto t_noise         = _internal_memory.get_tensor("noise");

    // (1) Run FlowLM with voice style conditioning (ignore output)
    set_flowlm_conditioning(t_voice_style);
    _lm_runner["flowlm"]->run();
    custom_op_fill_random_box_muller_transform_f32(t_noise);

    // (2) Run FlowLM with the output of text encoder (ignore output)
    set_flowlm_conditioning(t_text_encoder_out);
    _lm_runner["flowlm"]->run();
    custom_op_fill_random_box_muller_transform_f32(t_noise);

    // (3) Run FlowLM with BoS token to generate audio
    set_flowlm_conditioning(t_bos_embedding);
    _lm_runner["flowlm"]->run();

    auto flowlm_out = get_flowlm_output();

    // Generate random noise for SimpleMLPAdaLN
    custom_op_fill_random_box_muller_transform_f32(t_noise);

    _runner["simplemlpadaln"]->set_input("x", flowlm_out);
    _runner["simplemlpadaln"]->set_input("noise", t_noise);
    _runner["simplemlpadaln"]->run();

    // Get SimpleMLPAdaLN output (next_latent)
    auto t_simplemlpadaln_out = _runner["simplemlpadaln"]->get_output("next_latent");
    auto t_eos_logit          = _runner["simplemlpadaln"]->get_output("eos_logit");

    _eos_detected          = false;
    _eos_frames_countdown  = num_frames_after_eos;
    bool should_decode_bos = true;
    const float bos_eos_logit_value = *((float*)t_eos_logit.data);
    if(bos_eos_logit_value > _eos_threshold) {
        _eos_detected = true;
    }

    if(_eos_detected) {
        if(_eos_frames_countdown > 0) {
            --_eos_frames_countdown;
        } else {
            _is_output_ready = true;
            should_decode_bos = false;
        }
    }

    if(should_decode_bos) {
        _runner["mimi_decoder"]->set_input("latent", t_simplemlpadaln_out);
        _runner["mimi_decoder"]->run();

        // Get the Mimi Decoder output (raw audio chunk)
        auto t_mimi_decoder_out = _runner["mimi_decoder"]->get_output("audio_chunk");

        // Append the generated audio chunk to the vector
        int64_t num_elems = get_num_elems(t_mimi_decoder_out.format.shape);
        float*  t_mimi_decoder_out_ptr = (float*)t_mimi_decoder_out.data;
        for(int64_t i = 0; i < num_elems; ++i) {
            _generated_audio_frames.push_back(t_mimi_decoder_out_ptr[i]);
        }

        // START Latent projection ---------------------------------
        _runner["latent_proj"]->set_input("x", t_simplemlpadaln_out);
        _runner["latent_proj"]->run();
        // END Latent projection ---------------------------------
    }

    // Generate the audio frames
    for(int64_t frame_idx = 0; frame_idx < _num_max_frames; ++frame_idx) {

        auto t_latent_out = _runner["latent_proj"]->get_output("latent");

        // START FlowLM ---------------------------------
        set_flowlm_conditioning(t_latent_out);
        _lm_runner["flowlm"]->run();

        auto flowlm_out = get_flowlm_output();
        // END FlowLM ---------------------------------

        // Genereate random noise for SimpleMLPAdaLN
        auto t_noise = _internal_memory.get_tensor("noise");
        custom_op_fill_random_box_muller_transform_f32(t_noise);

        // START SimpleMLPadLn ---------------------------------
        _runner["simplemlpadaln"]->set_input("x", flowlm_out);
        _runner["simplemlpadaln"]->set_input("noise", t_noise);
        _runner["simplemlpadaln"]->run();

        auto t_simplemlpadaln_out = _runner["simplemlpadaln"]->get_output("next_latent");
        auto t_eos_logit          = _runner["simplemlpadaln"]->get_output("eos_logit");
        // END SimpleMLPadLn ---------------------------------

        // Check for end of sentence before decoding the current frame. This matches
        // upstream Pocket-TTS, which does not enqueue the frame that crosses the
        // frames_after_eos stop boundary.
        const float eos_logit_value = *((float*)t_eos_logit.data);
        if(eos_logit_value > _eos_threshold) {
            _eos_detected = true;
        }

        if(_eos_detected) {
            if(_eos_frames_countdown > 0) {
                --_eos_frames_countdown;
            } else {
                _is_output_ready = true;
                break;
            }
        }

        // START Mini decoder ---------------------------------
        _runner["mimi_decoder"]->set_input("latent", t_simplemlpadaln_out);
        _runner["mimi_decoder"]->run();

        // Ge the the raw audio chunk
        auto t_mimi_decoder_out = _runner["mimi_decoder"]->get_output("audio_chunk");
        // END Mini decoder ---------------------------------

        // Append the generated audio chunk to the vector
        int64_t num_elems = get_num_elems(t_mimi_decoder_out.format.shape);
        float*  t_mimi_decoder_out_ptr = (float*)t_mimi_decoder_out.data;

        for(int64_t i = 0; i < num_elems; ++i) {
            _generated_audio_frames.push_back(t_mimi_decoder_out_ptr[i]);
        }

        // START Latent projection ---------------------------------
        _runner["latent_proj"]->set_input("x", t_simplemlpadaln_out);
        _runner["latent_proj"]->run();
        // END Latent projection ---------------------------------
    }

    apply_cosine_fade_out(_generated_audio_frames.data(),
                          static_cast<int64_t>(_generated_audio_frames.size()),
                          _fade_out_samples);

    // Append the silence
    auto t_trailing_silence = _internal_memory.get_tensor("trailing_silence");
    for(int64_t i = 0; i < get_num_elems(t_trailing_silence.format.shape); ++i) {
        _generated_audio_frames.push_back(((float*)(t_trailing_silence.data))[i]);
    }

    _is_output_ready = true;

    TensorDict output;

    // Return the output
    output.declare_tensor("left_channel");
    output.declare_tensor("right_channel");
    output.declare_tensor("sample_rate");

    const int64_t num_audio_samples = _generated_audio_frames.size();

    output["left_channel"].data         = (void*)_generated_audio_frames.data();
    output["left_channel"].format.dt    = DataType::F32;
    output["left_channel"].format.shape = {static_cast<int64_t>(num_audio_samples)};

    output["right_channel"].data         = (void*)_generated_audio_frames.data();
    output["right_channel"].format.dt    = DataType::F32;
    output["right_channel"].format.shape = {static_cast<int64_t>(num_audio_samples)};

    output["sample_rate"].data         = (void*)&_sample_rate;
    output["sample_rate"].format.dt    = DataType::I64;
    output["sample_rate"].format.shape = {1};

    return output;
}

TensorDict PocketTTS::run_inference_streaming(const TensorDict& input) {

    if(_pending_trailing_silence) {
        auto t_trailing_silence = _internal_memory.get_tensor("trailing_silence");

        if(_last_audio_chunk_data != nullptr && !_last_audio_chunk_faded) {
            apply_cosine_fade_out(_last_audio_chunk_data, _last_audio_chunk_num_samples, _fade_out_samples);
            _last_audio_chunk_faded = true;
        }

        _pending_trailing_silence = false;
        _is_output_ready          = true;

        TensorDict output;
        output.declare_tensor("left_channel");
        output.declare_tensor("right_channel");
        output.declare_tensor("sample_rate");

        const int64_t num_audio_samples = get_num_elems(t_trailing_silence.format.shape);

        output["left_channel"].data         = t_trailing_silence.data;
        output["left_channel"].format.dt    = DataType::F32;
        output["left_channel"].format.shape = { num_audio_samples };

        output["right_channel"].data         = t_trailing_silence.data;
        output["right_channel"].format.dt    = DataType::F32;
        output["right_channel"].format.shape = { num_audio_samples };

        output["sample_rate"].data         = (void*)&_sample_rate;
        output["sample_rate"].format.dt    = DataType::I64;
        output["sample_rate"].format.shape = {1};

        return output;
    }

    if(_is_first_run_streaming_mode) {

        _rng.seed(_seed);

        // Preprocess text prompt
        const std::string txt_prompt((const char*)input["text_prompt"].data);
        std::string       preprocessed_txt_prompt;

        const int64_t num_words            = preprocess_prompt(txt_prompt, preprocessed_txt_prompt);
        const int64_t num_frames_after_eos = get_num_frames_after_eos(num_words);

        TensorData t_preprocessed_prompt;
        t_preprocessed_prompt.data = (void *)preprocessed_txt_prompt.c_str();
        t_preprocessed_prompt.format.dt = DataType::String;
        t_preprocessed_prompt.format.shape = { (int64_t)preprocessed_txt_prompt.size() };

        // START SPM ---------------------------------
        _runner["spm"]->set_input("text_prompt", t_preprocessed_prompt);

        _runner["spm"]->run();

        auto t_spm_out = _runner["spm"]->get_output("ids");
        // END SPM ---------------------------------

        int64_t num_token_ids = t_spm_out.format.shape[0];

        _internal_memory.allocate("token_ids_i64", TensorFormat{ DataType::I64, { 1, num_token_ids } });
        auto t_token_ids_i64 = _internal_memory.get_tensor("token_ids_i64");

        // Convert the SPM output to I64
        auto* dst = static_cast<int64_t*>(t_token_ids_i64.data);
        auto* src = static_cast<const int32_t*>(t_spm_out.data);

        for (int64_t i = 0; i < num_token_ids; ++i) {
            dst[i] = static_cast<int64_t>(src[i]);
        }

        // START Text encoder ---------------------------------
        _runner["text_encoder"]->set_input("token_ids", t_token_ids_i64);
        _runner["text_encoder"]->run();

        auto t_text_encoder_out = _runner["text_encoder"]->get_output("embeddings");
        // END Text encoder ---------------------------------

        // Get the voice style and bos embedding tensors
        auto t_voice_style   = _internal_memory.get_tensor("voice_style");
        auto t_bos_embedding = _internal_memory.get_tensor("bos_embedding");
        auto t_noise         = _internal_memory.get_tensor("noise");

        // (1) Run FlowLM with voice style conditioning (ignore output)
        set_flowlm_conditioning(t_voice_style);
        _lm_runner["flowlm"]->run();
        custom_op_fill_random_box_muller_transform_f32(t_noise);

        // (2) Run FlowLM with the output of text encoder (ignore output)
        set_flowlm_conditioning(t_text_encoder_out);
        _lm_runner["flowlm"]->run();
        custom_op_fill_random_box_muller_transform_f32(t_noise);

        // (3) Run FlowLM with BoS token to generate audio
        set_flowlm_conditioning(t_bos_embedding);
        _lm_runner["flowlm"]->run();

        auto flowlm_out = get_flowlm_output();

        // Generate random noise for SimpleMLPAdaLN
        custom_op_fill_random_box_muller_transform_f32(t_noise);

        _runner["simplemlpadaln"]->set_input("x", flowlm_out);
        _runner["simplemlpadaln"]->set_input("noise", t_noise);
        _runner["simplemlpadaln"]->run();

        // Get SimpleMLPAdaLN output (next_latent)
        auto t_simplemlpadaln_out = _runner["simplemlpadaln"]->get_output("next_latent");
        auto t_eos_logit          = _runner["simplemlpadaln"]->get_output("eos_logit");

        _eos_detected          = false;
        _eos_frames_countdown  = num_frames_after_eos;
        bool should_decode_bos = true;
        bool should_fade_out_bos = false;
        const float bos_eos_logit_value = *((float*)t_eos_logit.data);

        if(bos_eos_logit_value > _eos_threshold) {
            _eos_detected = true;
        }

        if(_eos_detected) {
            if(_eos_frames_countdown > 0) {
                --_eos_frames_countdown;
                should_fade_out_bos = _eos_frames_countdown == 0;
            } else {
                _pending_trailing_silence = true;
                should_decode_bos         = false;
            }
        }

        _is_first_run_streaming_mode = false;

        if(!should_decode_bos) {
            auto t_trailing_silence = _internal_memory.get_tensor("trailing_silence");

            TensorDict output;
            output.declare_tensor("left_channel");
            output.declare_tensor("right_channel");
            output.declare_tensor("sample_rate");

            output["left_channel"].data         = t_trailing_silence.data;
            output["left_channel"].format.dt    = DataType::F32;
            output["left_channel"].format.shape = { 0 };

            output["right_channel"].data         = t_trailing_silence.data;
            output["right_channel"].format.dt    = DataType::F32;
            output["right_channel"].format.shape = { 0 };

            output["sample_rate"].data         = (void*)&_sample_rate;
            output["sample_rate"].format.dt    = DataType::I64;
            output["sample_rate"].format.shape = {1};

            return output;
        }

        if(should_decode_bos) {
            _runner["mimi_decoder"]->set_input("latent", t_simplemlpadaln_out);
            _runner["mimi_decoder"]->run();

            // Get the Mimi Decoder output (raw audio chunk)
            auto t_mimi_decoder_out = _runner["mimi_decoder"]->get_output("audio_chunk");

            // START Latent projection ---------------------------------
            _runner["latent_proj"]->set_input("x", t_simplemlpadaln_out);
            _runner["latent_proj"]->run();
            // END Latent projection ---------------------------------

            // Return the output
            TensorDict output;
            output.declare_tensor("left_channel");
            output.declare_tensor("right_channel");
            output.declare_tensor("sample_rate");

            const int64_t num_audio_samples = get_num_elems(t_mimi_decoder_out.format.shape);
            auto*         audio_ptr         = static_cast<float*>(t_mimi_decoder_out.data);

            if(should_fade_out_bos) {
                apply_cosine_fade_out(audio_ptr, num_audio_samples, _fade_out_samples);
            }

            _last_audio_chunk_data        = audio_ptr;
            _last_audio_chunk_num_samples = num_audio_samples;
            _last_audio_chunk_faded       = should_fade_out_bos;

            output["left_channel"].data         = t_mimi_decoder_out.data;
            output["left_channel"].format.dt    = DataType::F32;
            output["left_channel"].format.shape = {static_cast<int64_t>(num_audio_samples)};

            output["right_channel"].data         = t_mimi_decoder_out.data;
            output["right_channel"].format.dt    = DataType::F32;
            output["right_channel"].format.shape = {static_cast<int64_t>(num_audio_samples)};

            output["sample_rate"].data         = (void*)&_sample_rate;
            output["sample_rate"].format.dt    = DataType::I64;
            output["sample_rate"].format.shape = {1};

            return output;
        }
    }

    // Generate a single audio frame
    if(_is_output_ready == false) {
        // If in streaming mode, the run method only generates a single frame
        auto t_latent_out = _runner["latent_proj"]->get_output("latent");

        // START FlowLM ---------------------------------
        set_flowlm_conditioning(t_latent_out);
        _lm_runner["flowlm"]->run();

        auto flowlm_out = get_flowlm_output();
        // END FlowLM ---------------------------------

        // Genereate random noise for SimpleMLPAdaLN
        auto t_noise = _internal_memory.get_tensor("noise");
        custom_op_fill_random_box_muller_transform_f32(t_noise);

        // START SimpleMLPadLn ---------------------------------
        _runner["simplemlpadaln"]->set_input("x", flowlm_out);
        _runner["simplemlpadaln"]->set_input("noise", t_noise);
        _runner["simplemlpadaln"]->run();

        auto t_simplemlpadaln_out = _runner["simplemlpadaln"]->get_output("next_latent");
        auto t_eos_logit          = _runner["simplemlpadaln"]->get_output("eos_logit");
        // END SimpleMLPadLn ---------------------------------

        // Check for end of sentence before decoding the current frame. This matches
        // upstream Pocket-TTS, which does not enqueue the frame that crosses the
        // frames_after_eos stop boundary.
        bool should_fade_out_current_chunk = false;
        const float eos_logit_value = *((float*)t_eos_logit.data);
        if(eos_logit_value > _eos_threshold) {
            _eos_detected = true;
        }

        if(_eos_detected) {
            if(_eos_frames_countdown > 0) {
                --_eos_frames_countdown;
                should_fade_out_current_chunk = _eos_frames_countdown == 0;
            } else {
                _pending_trailing_silence = true;

                if(_last_audio_chunk_data != nullptr && !_last_audio_chunk_faded) {
                    apply_cosine_fade_out(_last_audio_chunk_data,
                                          _last_audio_chunk_num_samples,
                                          _fade_out_samples);
                    _last_audio_chunk_faded = true;
                }

                auto t_trailing_silence = _internal_memory.get_tensor("trailing_silence");

                TensorDict output;
                output.declare_tensor("left_channel");
                output.declare_tensor("right_channel");
                output.declare_tensor("sample_rate");

                output["left_channel"].data         = t_trailing_silence.data;
                output["left_channel"].format.dt    = DataType::F32;
                output["left_channel"].format.shape = { 0 };

                output["right_channel"].data         = t_trailing_silence.data;
                output["right_channel"].format.dt    = DataType::F32;
                output["right_channel"].format.shape = { 0 };

                output["sample_rate"].data         = (void*)&_sample_rate;
                output["sample_rate"].format.dt    = DataType::I64;
                output["sample_rate"].format.shape = {1};

                return output;
            }
        }

        // START Mini decoder ---------------------------------
        _runner["mimi_decoder"]->set_input("latent", t_simplemlpadaln_out);
        _runner["mimi_decoder"]->run();

        // Ge the the raw audio chunk
        auto t_mimi_decoder_out = _runner["mimi_decoder"]->get_output("audio_chunk");
        // END Mini decoder ---------------------------------

        // START Latent projection ---------------------------------
        _runner["latent_proj"]->set_input("x", t_simplemlpadaln_out);
        _runner["latent_proj"]->run();
        // END Latent projection ---------------------------------

        // Return the output
        TensorDict output;
        output.declare_tensor("left_channel");
        output.declare_tensor("right_channel");
        output.declare_tensor("sample_rate");

        const int64_t num_audio_samples = get_num_elems(t_mimi_decoder_out.format.shape);
        auto*         audio_ptr         = static_cast<float*>(t_mimi_decoder_out.data);

        if(should_fade_out_current_chunk) {
            apply_cosine_fade_out(audio_ptr, num_audio_samples, _fade_out_samples);
        }

        _last_audio_chunk_data        = audio_ptr;
        _last_audio_chunk_num_samples = num_audio_samples;
        _last_audio_chunk_faded       = should_fade_out_current_chunk;

        output["left_channel"].data         = t_mimi_decoder_out.data;
        output["left_channel"].format.dt    = DataType::F32;
        output["left_channel"].format.shape = {static_cast<int64_t>(num_audio_samples)};

        output["right_channel"].data         = t_mimi_decoder_out.data;
        output["right_channel"].format.dt    = DataType::F32;
        output["right_channel"].format.shape = {static_cast<int64_t>(num_audio_samples)};

        output["sample_rate"].data         = (void*)&_sample_rate;
        output["sample_rate"].format.dt    = DataType::I64;
        output["sample_rate"].format.shape = {1};

        return output;
    } else {
        auto t_trailing_silence = _internal_memory.get_tensor("trailing_silence");

        TensorDict output;
        output.declare_tensor("left_channel");
        output.declare_tensor("right_channel");
        output.declare_tensor("sample_rate");

        output["left_channel"].data         = t_trailing_silence.data;
        output["left_channel"].format.dt    = DataType::F32;
        output["left_channel"].format.shape = { 0 };

        output["right_channel"].data         = t_trailing_silence.data;
        output["right_channel"].format.dt    = DataType::F32;
        output["right_channel"].format.shape = { 0 };

        return output;
    }
}

int64_t PocketTTS::get_num_elems(const std::vector<int64_t>& values) {
    int64_t result = 1;
    for (int64_t v : values) {
        result *= v;
    }
    return result;
}

TensorFormat PocketTTS::get_flowlm_conditioning_format(int64_t sequence_length) {
    TensorFormat format = _lm_runner["flowlm"]->get_input_format("conditioning");
    COMPUTE_FLOW_ASSERT(format.shape.size() == 2);
    format.shape = {1, sequence_length, format.shape[1]};
    return format;
}

void PocketTTS::set_flowlm_conditioning(TensorData tensor) {
    COMPUTE_FLOW_ASSERT(tensor.format.shape.size() == 3);
    COMPUTE_FLOW_ASSERT(tensor.format.shape[0] == 1);
    tensor.format.shape = { tensor.format.shape[1], tensor.format.shape[2] };

    _lm_runner["flowlm"]->set_input("conditioning", tensor);
}

TensorData PocketTTS::get_flowlm_output() {
    TensorData output = _lm_runner["flowlm"]->get_output("x");
    COMPUTE_FLOW_ASSERT(output.format.shape.size() == 1);
    output.format.shape = { 1, 1, output.format.shape[0] };

    return output;
}

} // namespace compute_flow::bricks::tts
