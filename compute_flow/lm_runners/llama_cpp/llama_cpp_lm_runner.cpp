//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifdef COMPUTE_FLOW_HAS_LLAMACPP

#include "compute_flow/lm_runners/llama_cpp/llama_cpp_lm_runner.h"
#include "compute_flow/lm_runners/llama_cpp/llama_cpp_tensor_indices.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <vector>

namespace compute_flow {

namespace {

std::once_flag g_llama_backend_once;

std::string profile_op_name(const ggml_tensor& tensor) {
    std::string name = ggml_op_desc(&tensor);
    name += '[';

    bool all_types_match = true;
    for (const ggml_tensor* source : tensor.src) {
        if (source != nullptr && source->type != tensor.type) {
            all_types_match = false;
            break;
        }
    }

    if (all_types_match) {
        name += ggml_type_name(tensor.type);
    } else {
        bool is_first_source = true;
        for (const ggml_tensor* source : tensor.src) {
            if (source == nullptr) continue;
            if (!is_first_source) name += ',';
            name += ggml_type_name(source->type);
            is_first_source = false;
        }
        name += "->";
        name += ggml_type_name(tensor.type);
    }

    if (tensor.op == GGML_OP_MUL_MAT && tensor.src[1] != nullptr) {
        name += tensor.src[1]->ne[1] == 1 ? ",GEMV" : ",GEMM";
    }
    name += ']';
    return name;
}

// TODO: Should this a helper function offered by Compute Flow?
std::string tensor_to_string(const TensorData& t) {
    if (t.format.dt != DataType::String) {
        throw std::invalid_argument("LlamaCppRunner: expected DataType::String for text tensor");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument("LlamaCppRunner: text tensor has null data");
    }
    if (t.format.shape.empty()) {
        throw std::invalid_argument("LlamaCppRunner: text tensor has empty shape");
    }
    const int64_t nbytes = t.format.shape[0];
    if (nbytes < 0) {
        throw std::invalid_argument("LlamaCppRunner: text tensor has dynamic/negative size");
    }
    return std::string(static_cast<const char*>(t.data), static_cast<size_t>(nbytes));
}

[[noreturn]] void ThrowNotImplemented(const char* api_name) {
    throw std::runtime_error(std::string("LlamaCppLMRunner: not implemented: ") + api_name);
}

struct LcgSamplerCtx {
    uint32_t seed;
    uint32_t seed_cur;
    uint32_t rng_state;
};

static float quickselect_kth_largest(float* arr, int n, int k) {
    int lo = 0;
    int hi = n - 1;
    while (lo < hi) {
        float pivot = arr[lo + (hi - lo) / 2];
        int i = lo;
        int j = hi;
        int p = lo;
        while (i <= j) {
            if (arr[i] > pivot) {
                const float t = arr[p];
                arr[p] = arr[i];
                arr[i] = t;
                ++p;
                ++i;
            } else if (arr[i] < pivot) {
                const float t = arr[i];
                arr[i] = arr[j];
                arr[j] = t;
                --j;
            } else {
                ++i;
            }
        }
        if (k - 1 < p) {
            hi = p - 1;
        } else if (k - 1 > j) {
            lo = j + 1;
        } else {
            return pivot;
        }
    }
    return arr[lo];
}

struct TopKThresholdTiesCtx {
    int32_t k;
    std::vector<float> tmp;
};

struct MinGeneratedTokensCtx {
    const llama_vocab* vocab;
    const int32_t* generated_tokens;
    int32_t min_generated_tokens;
};

static const char* min_generated_tokens_name(const struct llama_sampler* /*smpl*/) {
    return "min-generated-tokens";
}

static void min_generated_tokens_apply(struct llama_sampler* smpl, llama_token_data_array* cur_p) {
    auto* ctx = static_cast<MinGeneratedTokensCtx*>(smpl->ctx);
    if (ctx == nullptr || ctx->vocab == nullptr || ctx->generated_tokens == nullptr ||
        *ctx->generated_tokens >= ctx->min_generated_tokens) {
        return;
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        if (llama_vocab_is_eog(ctx->vocab, cur_p->data[i].id)) {
            cur_p->data[i].logit = -std::numeric_limits<float>::infinity();
        }
    }
    cur_p->sorted = false;
}

static struct llama_sampler* min_generated_tokens_clone(const struct llama_sampler* smpl);

static void min_generated_tokens_free(struct llama_sampler* smpl) {
    delete static_cast<MinGeneratedTokensCtx*>(smpl->ctx);
}

static struct llama_sampler_i min_generated_tokens_i = {
    /* .name              = */ min_generated_tokens_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ min_generated_tokens_apply,
    /* .reset             = */ nullptr,
    /* .clone             = */ min_generated_tokens_clone,
    /* .free              = */ min_generated_tokens_free,
};

static struct llama_sampler* llama_sampler_init_min_generated_tokens(
    const llama_vocab* vocab,
    const int32_t* generated_tokens,
    int32_t min_generated_tokens) {
    return llama_sampler_init(
        /* .iface = */ &min_generated_tokens_i,
        /* .ctx   = */ new MinGeneratedTokensCtx{
            /* .vocab                = */ vocab,
            /* .generated_tokens     = */ generated_tokens,
            /* .min_generated_tokens = */ min_generated_tokens,
        });
}

static struct llama_sampler* min_generated_tokens_clone(const struct llama_sampler* smpl) {
    const auto* ctx = static_cast<const MinGeneratedTokensCtx*>(smpl->ctx);
    return llama_sampler_init_min_generated_tokens(ctx->vocab, ctx->generated_tokens, ctx->min_generated_tokens);
}

static const char* top_k_threshold_ties_name(const struct llama_sampler* /*smpl*/) {
    return "top-k-threshold-ties";
}

static void top_k_threshold_ties_apply(struct llama_sampler* smpl, llama_token_data_array* cur_p) {
    auto* ctx = static_cast<TopKThresholdTiesCtx*>(smpl->ctx);
    if (ctx->k <= 0 || cur_p->size == 0) {
        return;
    }

    const int32_t n = static_cast<int32_t>(cur_p->size);
    if (ctx->k >= n) {
        return;
    }

    if (static_cast<int32_t>(ctx->tmp.size()) < n) {
        ctx->tmp.resize(static_cast<size_t>(n));
    }

    for (int32_t i = 0; i < n; ++i) {
        ctx->tmp[static_cast<size_t>(i)] = cur_p->data[i].logit;
    }

    const float threshold = quickselect_kth_largest(ctx->tmp.data(), n, ctx->k);

    size_t out = 0;
    for (int32_t i = 0; i < n; ++i) {
        if (cur_p->data[i].logit >= threshold) {
            cur_p->data[out] = cur_p->data[i];
            ++out;
        }
    }

    cur_p->size = out;
    cur_p->sorted = false;
}

static struct llama_sampler* top_k_threshold_ties_clone(const struct llama_sampler* smpl);

static void top_k_threshold_ties_free(struct llama_sampler* smpl) {
    delete static_cast<TopKThresholdTiesCtx*>(smpl->ctx);
}

static struct llama_sampler_i top_k_threshold_ties_i = {
    /* .name              = */ top_k_threshold_ties_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ top_k_threshold_ties_apply,
    /* .reset             = */ nullptr,
    /* .clone             = */ top_k_threshold_ties_clone,
    /* .free              = */ top_k_threshold_ties_free,
};

static struct llama_sampler* llama_sampler_init_top_k_threshold_ties(int32_t k) {
    if (k <= 0) {
        // Match llama.cpp behavior for disabled top-k without relying on newer APIs.
        return llama_sampler_init_top_k(k);
    }

    return llama_sampler_init(
        /* .iface = */ &top_k_threshold_ties_i,
        /* .ctx   = */ new TopKThresholdTiesCtx{
            /* .k   = */ k,
            /* .tmp = */ {},
        });
}

static struct llama_sampler* top_k_threshold_ties_clone(const struct llama_sampler* smpl) {
    const auto* ctx = static_cast<const TopKThresholdTiesCtx*>(smpl->ctx);
    return llama_sampler_init_top_k_threshold_ties(ctx->k);
}

static uint32_t get_rng_seed_lcg(uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        static bool is_rd_prng = std::random_device().entropy() == 0;
        if (is_rd_prng) {
            return static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
        }
        std::random_device rd;
        return rd();
    }
    return seed;
}

static inline float lcg_rand_uniform(uint32_t& state) {
    state = state * 1103515245u + 12345u;
    return static_cast<float>((state >> 16) & 0x7FFFu) / 32768.0f;
}

static const char* lcg_sampler_name(const struct llama_sampler* /*smpl*/) {
    return "dist_lcg";
}

static void lcg_sampler_apply(struct llama_sampler* smpl, llama_token_data_array* cur_p) {
    auto* ctx = static_cast<LcgSamplerCtx*>(smpl->ctx);

    if (cur_p->size == 0) {
        cur_p->selected = -1;
        return;
    }

    cur_p->selected = 0;
    if (cur_p->size == 1) {
        cur_p->data[0].p = 1.0f;
        return;
    }

    float max_l = cur_p->data[0].logit;
    if (!cur_p->sorted) {
        for (size_t i = 1; i < cur_p->size; ++i) {
            max_l = std::max(max_l, cur_p->data[i].logit);
        }
    }

    double sum_cum = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float p = expf(cur_p->data[i].logit - max_l);
        cur_p->data[i].p = p;
        sum_cum += p;
    }

    const double rnd = static_cast<double>(lcg_rand_uniform(ctx->rng_state));
    const double sum_tgt = sum_cum * rnd;

    bool found = false;
    double sum_run = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (!found) {
            sum_run += cur_p->data[i].p;
            if (sum_run >= sum_tgt) {
                cur_p->selected = static_cast<int>(i);
                found = true;
            }
        }
        cur_p->data[i].p = static_cast<float>(cur_p->data[i].p / sum_cum);
    }

    if (!found) {
        cur_p->selected = static_cast<int>(cur_p->size - 1);
    }
}

static void lcg_sampler_reset(struct llama_sampler* smpl) {
    auto* ctx = static_cast<LcgSamplerCtx*>(smpl->ctx);
    ctx->seed_cur = get_rng_seed_lcg(ctx->seed);
    ctx->rng_state = ctx->seed_cur;
}

static struct llama_sampler* lcg_sampler_clone(const struct llama_sampler* smpl);

static void lcg_sampler_free(struct llama_sampler* smpl) {
    delete static_cast<LcgSamplerCtx*>(smpl->ctx);
}

static struct llama_sampler_i lcg_sampler_i = {
    /* .name              = */ lcg_sampler_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ lcg_sampler_apply,
    /* .reset             = */ lcg_sampler_reset,
    /* .clone             = */ lcg_sampler_clone,
    /* .free              = */ lcg_sampler_free,
};

static struct llama_sampler* llama_sampler_init_dist_lcg(uint32_t seed) {
    const uint32_t seed_cur = get_rng_seed_lcg(seed);
    return llama_sampler_init(
        /* .iface = */ &lcg_sampler_i,
        /* .ctx   = */ new LcgSamplerCtx{
            /* .seed      = */ seed,
            /* .seed_cur  = */ seed_cur,
            /* .rng_state = */ seed_cur,
        });
}

static struct llama_sampler* lcg_sampler_clone(const struct llama_sampler* smpl) {
    const auto* ctx = static_cast<const LcgSamplerCtx*>(smpl->ctx);
    auto* result = llama_sampler_init_dist_lcg(ctx->seed);
    auto* result_ctx = static_cast<LcgSamplerCtx*>(result->ctx);
    result_ctx->seed_cur = ctx->seed_cur;
    result_ctx->rng_state = ctx->rng_state;
    return result;
}
}  // namespace

bool LlamaCppLMRunner::profile_op(ggml_tensor* tensor, bool ask, void* user_data) {
    auto* runner = static_cast<LlamaCppLMRunner*>(user_data);
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();

    if (ask) {
        runner->_op_start_time_ns = now_ns;
        return true;
    }

    const std::string op_name = profile_op_name(*tensor);
    runner->_profiler.add_nested_sample(
        "llama_cpp", runner->_model_name, op_name, now_ns - runner->_op_start_time_ns);
    return true;
}

void LlamaCppLMRunner::setup(const ModelData& model, ProfilerView profiler) {
    _model_name = model.model_name;
    _profiler   = profiler;
    _saved_states.clear();

    if (model.lm_params.max_tokens_to_gen < -1) {
        throw std::runtime_error("LlamaCppRunner: max_tokens_to_gen must be >= -1");
    }
    _max_tokens_to_gen = model.lm_params.max_tokens_to_gen;

    const std::string model_path  = model.model_path;
    AcceleratorConfig accelerator = model.acc_config;

    std::call_once(g_llama_backend_once, []() {
        llama_backend_init();
    });

    if(accelerator.has_accelerator(AcceleratorType::Cpu) == false) {
        throw std::runtime_error("LlamaCppRunner: CPU only is supported");
    }

    const auto cpu = std::get<CpuOptions>(accelerator.priorities[0].params);

    // Load the model
    llama_model_params model_params = llama_model_default_params();
    _model.reset(llama_load_model_from_file(model_path.c_str(), model_params));

    if (!_model) {
        throw std::runtime_error("LlamaCppRunner: failed to load llama.cpp model: " + model_path);
    }

    _embedding_position_count =
        model.lm_params.embedding_position_mode == LMEmbeddingPositionMode::MRoPEBroadcast ? 4 : 1;

    // Initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = model.lm_params.max_context_tokens;
    ctx_params.n_batch         = model.lm_params.batch_size;
    ctx_params.n_threads       = static_cast<int32_t>(cpu.num_threads);
    ctx_params.n_threads_batch = static_cast<int32_t>(cpu.num_threads);
    ctx_params.embeddings      = true;
    if (profiler.is_enabled() && profiler.is_per_op_profiling_enabled()) {
        ctx_params.cb_eval           = profile_op;
        ctx_params.cb_eval_user_data = this;
    }

    _ctx.reset(llama_new_context_with_model(_model.get(), ctx_params));

    if (!_ctx) {
        throw std::runtime_error("LlamaCppRunner: failed to create llama.cpp context for: " + model_path);
    }

    if(model.lm_params.sampling_policy != LMSamplingPolicy::Disabled) {

        _is_sampler_enabled = true;
        // Append behavior is controlled by runner_params.sampler.append_generated_token in the manifest.
        _append_generated_token = model.lm_params.append_generated_token;
        _min_generated_tokens = model.lm_params.min_generated_tokens;
        _generated_tokens = 0;

        // Initialize the sampler
        llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (chain == nullptr) {
            throw std::runtime_error("LlamaCppRunner: failed to create llama sampler chain");
        }

        if (model.lm_params.repetition_penalty != 1.0f) {
            int32_t penalty_last_n = 0;
            if (model.lm_params.max_context_tokens > 0) {
                const int64_t max_ctx = model.lm_params.max_context_tokens;
                const int64_t cap = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
                penalty_last_n = static_cast<int32_t>(std::min(max_ctx, cap));
            }
            llama_sampler_chain_add(chain,
                llama_sampler_init_penalties(penalty_last_n,
                                            model.lm_params.repetition_penalty,
                                            0.0f,
                                            0.0f));
        }

        if (!model.lm_params.suppress_token_ranges.empty()) {
            const llama_vocab* vocab = llama_model_get_vocab(_model.get());
            if (vocab == nullptr) {
                throw std::runtime_error("LlamaCppRunner: failed to fetch llama.cpp vocab");
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            if (n_vocab <= 0) {
                throw std::runtime_error("LlamaCppRunner: invalid vocab size");
            }

            size_t total = 0;
            for (const auto& [start, end] : model.lm_params.suppress_token_ranges) {
                if (start < 0 || end < 0 || start >= n_vocab || end >= n_vocab) {
                    throw std::runtime_error("LlamaCppRunner: suppress_token_ranges out of vocab bounds");
                }
                total += static_cast<size_t>(end - start + 1);
            }

            std::vector<llama_logit_bias> logit_bias;
            logit_bias.reserve(total);
            const float suppress_bias = -std::numeric_limits<float>::infinity();
            for (const auto& [start, end] : model.lm_params.suppress_token_ranges) {
                for (int32_t token = start; token <= end; ++token) {
                    logit_bias.push_back({ token, suppress_bias });
                }
            }

            if (!logit_bias.empty()) {
                llama_sampler_chain_add(
                    chain,
                    llama_sampler_init_logit_bias(n_vocab,
                                                  static_cast<int32_t>(logit_bias.size()),
                                                  logit_bias.data()));
            }
        }

        if (_min_generated_tokens > 0) {
            const llama_vocab* vocab = llama_model_get_vocab(_model.get());
            if (vocab == nullptr) {
                throw std::runtime_error("LlamaCppRunner: failed to fetch llama.cpp vocab");
            }
            llama_sampler_chain_add(chain,
                llama_sampler_init_min_generated_tokens(vocab, &_generated_tokens, _min_generated_tokens));
        }

        switch(model.lm_params.sampling_policy) {
            case LMSamplingPolicy::Greedy: {
                llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            }
            break;
            case LMSamplingPolicy::Distribution: {
                const auto top_k = model.lm_params.distribution_params.top_k;
                const auto top_p = model.lm_params.distribution_params.top_p;
                const auto temp  = model.lm_params.distribution_params.temperature;
                const auto seed  = model.lm_params.distribution_params.seed;
                const auto rng_strategy = model.lm_params.distribution_params.rng_strategy;
                const auto top_k_strategy = model.lm_params.distribution_params.top_k_strategy;
                // Match HF generate() order: temperature, top-k, top-p.
                llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
                if (top_k > 0) {
                    switch (top_k_strategy) {
                        case LMTopKStrategy::StrictK:
                            llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
                            break;
                        case LMTopKStrategy::ThresholdTies:
                            llama_sampler_chain_add(chain, llama_sampler_init_top_k_threshold_ties(top_k));
                            break;
                        default:
                            throw std::runtime_error(
                                "LlamaCppRunner: unknown top_k_strategy (expected: strict_k, threshold_ties)");
                    }
                }
                if (top_p > 0.0f && top_p <= 1.0f) {
                    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
                }
                switch (rng_strategy) {
                    case LMRNGStrategy::Mt19937:
                        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
                        break;
                    case LMRNGStrategy::Lcg:
                        llama_sampler_chain_add(chain, llama_sampler_init_dist_lcg(seed));
                        break;
                    default:
                        throw std::runtime_error(
                            "LlamaCppRunner: unknown rng_strategy (expected: mt19937, lcg)");
                }
            }
            break;
            default: {
                throw std::runtime_error("LlamaCppRunner: Unsupported sampling policy");
            }
        }
        _sampler.reset(chain);
    } else {
        _is_sampler_enabled = false;
    }

    // Map input/output indices from manifest model section.
    for (const auto& [name, idx] : model.inputs) {
        map_input(name, static_cast<size_t>(idx));
    }

    for (const auto& [name, idx] : model.outputs) {
        map_output(name, static_cast<size_t>(idx));
    }

    reset_states();
}

const std::string& LlamaCppLMRunner::model_name() const {
    return _model_name;
}

void LlamaCppLMRunner::map_input(std::string_view key, size_t idx) {
    // llama.cpp does not manage input tensors internally.
    // Inputs are supplied by the user during decoding,
    // so we maintain our own index-to-input mapping.
    // Input semantics are inferred from idx to avoid relying on key names.
    std::string k(key);
    if (_in_tensor_idx.count(k)) {
        throw std::invalid_argument("Input key already exists: " + k);
    }

    if (_inputs.size() <= idx) {
        _inputs.resize(idx + 1);
    }

    _in_tensor_idx.emplace(std::move(k), idx);

    auto& tensor = _inputs[idx];
    tensor.data = nullptr;
    tensor.user_hosted = true;

    switch (idx) {
        case k_input_token_ids_idx:
            tensor.format.dt = DataType::I32;
            tensor.format.shape = { 1 };
            return;
        case k_input_embeds_idx: {
            const int32_t n_embd = llama_model_n_embd_inp(_model.get());
            if (n_embd <= 0) {
                throw std::runtime_error("LlamaCppRunner: invalid embedding dimension for inputs_embeds");
            }
            tensor.format.dt = DataType::F32;
            tensor.format.shape = { -1, n_embd };
            return;
        }
        default:
            throw std::invalid_argument("LlamaCppRunner: unsupported input index: " + std::to_string(idx));
    }
}

void LlamaCppLMRunner::map_input(std::string_view /*key*/, std::string_view /*tensor_name*/) {
    ThrowNotImplemented("map_input(key, tensor_name)");
}

void LlamaCppLMRunner::map_output(std::string_view key, size_t idx) {
    // llama.cpp does not expose persistent output tensors.
    // Outputs (e.g., logits) are produced during decode and accessed
    // through runtime buffers, so we maintain our own indexed mapping.
    // Output semantics are inferred from idx to avoid relying on key names.
    std::string k(key);
    if (_out_tensor_idx.count(k)) {
        throw std::invalid_argument("Output key already exists: " + k);
    }

    if (_outputs.size() <= idx) {
        _outputs.resize(idx + 1);
    }

    _out_tensor_idx.emplace(std::move(k), idx);

    auto& tensor = _outputs[idx];

    switch (idx) {
        case k_output_token_id_idx:
            tensor.data = &_last_generated_token_id;
            tensor.user_hosted = false;
            tensor.format.dt = DataType::I32;
            tensor.format.shape = { 1 }; // One token ID at the time
            return;
        case k_output_hidden_state_idx: {
            const int32_t n_embd = llama_model_n_embd_inp(_model.get());
            if (n_embd <= 0) {
                throw std::runtime_error("LlamaCppRunner: invalid embedding dimension for hidden_state output");
            }
            if (_last_hidden_state.size() != static_cast<size_t>(n_embd)) {
                _last_hidden_state.assign(static_cast<size_t>(n_embd), 0.0f);
            }
            tensor.data = _last_hidden_state.data();
            tensor.user_hosted = false;
            tensor.format.dt = DataType::F32;
            tensor.format.shape = { n_embd };
            return;
        }
        case k_output_logits_idx: {
            const int32_t vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(_model.get()));
            if (vocab_size <= 0) {
                throw std::runtime_error("LlamaCppRunner: invalid vocab size for logits output");
            }
            _logits_vocab_size = vocab_size;
            _last_logits.assign(static_cast<size_t>(vocab_size), 0.0f);
            tensor.data = _last_logits.data();
            tensor.user_hosted = false;
            tensor.format.dt = DataType::F32;
            tensor.format.shape = { vocab_size };
            return;
        }
        case k_output_hidden_state_seq_idx: {
            // Full-sequence hidden states: shape is dynamic [-1, n_embd], updated after each run().
            const int32_t n_embd = llama_model_n_embd_inp(_model.get());
            if (n_embd <= 0) {
                throw std::runtime_error("LlamaCppRunner: invalid embedding dimension for hidden_state_seq output");
            }
            tensor.data           = nullptr;
            tensor.user_hosted    = false;
            tensor.format.dt      = DataType::F32;
            tensor.format.shape   = { -1, n_embd };
            return;
        }
        case k_output_pooled_embedding_idx: {
            const int32_t n_embd = llama_model_n_embd_inp(_model.get());
            if (n_embd <= 0) {
                throw std::runtime_error("LlamaCppRunner: invalid embedding dimension for pooled_embedding output");
            }
            _last_pooled_embedding.assign(static_cast<size_t>(n_embd), 0.0f);
            tensor.data = _last_pooled_embedding.data();
            tensor.user_hosted = false;
            tensor.format.dt = DataType::F32;
            tensor.format.shape = { n_embd };
            return;
        }
        default:
            throw std::invalid_argument("LlamaCppRunner: unsupported output index: " + std::to_string(idx));
    }
}

void LlamaCppLMRunner::map_output(std::string_view /*key*/, std::string_view /*tensor_name*/) {
    ThrowNotImplemented("map_output(key, tensor_name)");
}

void LlamaCppLMRunner::set_input(const std::string& key, const TensorData& tensor) {
    auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) {
        throw std::out_of_range("Input key not found: " + key);
    }

    if (tensor.format.has_dynamic_shape()) {
        throw std::invalid_argument("Input tensor shape must be static for key: " + key);
    }

    const size_t idx = it->second;
    if (_inputs.size() <= idx) {
        throw std::runtime_error("LlamaCppRunner: internal input mapping out of range");
    }

    // TODO: We should check whether the format is what we expect

    _inputs[idx] = tensor;
    _has_new_input = true;
}

TensorData LlamaCppLMRunner::get_output(const std::string& key) {
    auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) {
        throw std::out_of_range("Output key not found: " + key);
    }

    const size_t idx = it->second;
    if (_outputs.size() <= idx) {
        throw std::runtime_error("LlamaCppRunner: internal output mapping out of range");
    }

    return _outputs[idx];
}

TensorFormat LlamaCppLMRunner::get_input_format(const std::string& key) {
    auto it = _in_tensor_idx.find(key);
    if (it == _in_tensor_idx.end()) {
        throw std::out_of_range("Input key not found: " + key);
    }

    const size_t idx = it->second;
    if (_inputs.size() <= idx) {
        throw std::runtime_error("LlamaCppRunner: internal input mapping out of range");
    }
    return _inputs[idx].format;
}

TensorFormat LlamaCppLMRunner::get_output_format(const std::string& key) {
    auto it = _out_tensor_idx.find(key);
    if (it == _out_tensor_idx.end()) {
        throw std::out_of_range("Output key not found: " + key);
    }

    const size_t idx = it->second;
    if (_outputs.size() <= idx) {
        throw std::runtime_error("LlamaCppRunner: internal output mapping out of range");
    }
    return _outputs[idx].format;
}

void LlamaCppLMRunner::reset_states() {
    if (_ctx) {
        llama_memory_clear(llama_get_memory(_ctx.get()), true);
    }
    if (_sampler) {
        llama_sampler_reset(_sampler.get());
    }

    _last_generated_token_id = -1;
    _is_eog = true;
    _is_first_run = true;
    _has_new_input = false;
    _generated_tokens = 0;
}

void LlamaCppLMRunner::run() {
    auto model_scope = _profiler.measure_model("llama_cpp", _model_name);

    if (!_ctx) {
        throw std::runtime_error("LlamaCppRunner: run called before setup()");
    }

    const TensorData* token_ids_td = nullptr;
    const TensorData* embeds_td = nullptr;

    if (_inputs.size() > k_input_token_ids_idx) {
        token_ids_td = &_inputs[k_input_token_ids_idx];
    }
    if (_inputs.size() > k_input_embeds_idx) {
        embeds_td = &_inputs[k_input_embeds_idx];
    }

    const TensorData* active_td = nullptr;
    bool use_embeddings = false;
    const bool has_token_ids = (token_ids_td != nullptr && token_ids_td->data != nullptr);
    if (embeds_td != nullptr && embeds_td->data != nullptr) {
        active_td = embeds_td;
        use_embeddings = true;
    }
    if (has_token_ids && active_td == nullptr) {
        active_td = token_ids_td;
        use_embeddings = false;
    }
    if (active_td == nullptr) {
        throw std::runtime_error("LlamaCppRunner: input tensor data is null");
    }
    if (active_td->format.shape.empty()) {
        throw std::runtime_error("LlamaCppRunner: input shape is empty");
    }

    // Re-encode when the caller provides new input; llama.cpp does not track input tensors.
    const bool was_generation_complete = is_end_of_generation();
    const bool should_encode = _is_eog || was_generation_complete || _has_new_input;
    if (should_encode) {
        if (was_generation_complete) {
            _generated_tokens = 0;
        }
        if (!use_embeddings) {
            // Token IDs
            if (active_td->format.dt != DataType::I32) {
                throw std::runtime_error("LlamaCppRunner: token ids must be DataType::I32");
            }
            if (active_td->format.shape.size() != 1) {
                throw std::runtime_error("LlamaCppRunner: token ids must be a 1D tensor");
            }
            if (active_td->format.shape[0] <= 0) {
                throw std::runtime_error("LlamaCppRunner: input token ids length must be positive");
            }
            const auto* token_ids_ptr = reinterpret_cast<const llama_token*>(active_td->data);
            const int32_t num_tokens  = static_cast<int32_t>(active_td->format.shape[0]);

            const int32_t n_batch = llama_n_batch(_ctx.get());
            const int32_t batch_size = std::max<int32_t>(1, n_batch);
            int32_t offset = 0;
            int32_t last_batch_tokens = 0;
            while (offset < num_tokens) {
                const int32_t n = std::min(num_tokens - offset, batch_size);
                const int32_t n_ctx = llama_n_ctx(_ctx.get());
                const int32_t n_ctx_used = static_cast<int32_t>(
                    llama_memory_seq_pos_max(llama_get_memory(_ctx.get()), 0) + 1);

                if (n_ctx_used + n > n_ctx) {
                    throw std::runtime_error("LlamaCppRunner: context size exceeded during encode");
                }

                llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(token_ids_ptr + offset),
                                                        static_cast<int32_t>(n));

                const int32_t rc = llama_decode(_ctx.get(), batch);
                if (rc != 0) {
                    if(rc == 1) {
                        throw std::runtime_error("LlamaCppRunner: llama_decode returned code 1: no KV slot available; context likely full");
                    } else {
                        throw std::runtime_error("LlamaCppRunner: llama_decode generic error");
                    }
                }
                if (_sampler) {
                    for (int32_t i = 0; i < n; ++i) {
                        llama_sampler_accept(_sampler.get(), token_ids_ptr[offset + i]);
                    }
                }
                last_batch_tokens = n;
                offset += n;
            }

            if (!_last_hidden_state.empty() && last_batch_tokens > 0) {
                const float* emb = llama_get_embeddings_ith(_ctx.get(), last_batch_tokens - 1);
                if (emb != nullptr) {
                    const int32_t n_embd = llama_model_n_embd_inp(_model.get());
                    std::memcpy(_last_hidden_state.data(), emb, static_cast<size_t>(n_embd) * sizeof(float));
                }
            }
            if (!_last_logits.empty()) {
                const float* logits = llama_get_logits_ith(_ctx.get(), -1);
                if (logits != nullptr) {
                    std::memcpy(_last_logits.data(), logits,
                                static_cast<size_t>(_logits_vocab_size) * sizeof(float));
                }
            }
        } else {
            // Embeddings
            const auto* embd_ptr = reinterpret_cast<const float*>(active_td->data);
            const llama_token* token_ids_ptr = nullptr;

            if (active_td->format.shape.size() != 1 && active_td->format.shape.size() != 2) {
                throw std::runtime_error("LlamaCppRunner: embeddings must be a 1D or 2D tensor");
            }

            const int32_t n_embd = llama_model_n_embd_inp(_model.get());
            if (n_embd <= 0) {
                throw std::runtime_error("LlamaCppRunner: invalid embedding dimension");
            }

            int64_t num_tokens64 = 1;
            int64_t embd_dim = 0;
            if (active_td->format.shape.size() == 1) {
                embd_dim = active_td->format.shape[0];
            } else {
                num_tokens64 = active_td->format.shape[0];
                embd_dim = active_td->format.shape[1];
            }

            if (num_tokens64 <= 0 || embd_dim <= 0) {
                throw std::runtime_error("LlamaCppRunner: embedding shape must be positive");
            }
            if (num_tokens64 > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("LlamaCppRunner: embedding token count exceeds int32_t");
            }
            if (embd_dim != n_embd) {
                throw std::runtime_error("LlamaCppRunner: embedding dimension mismatch");
            }

            const int32_t num_tokens = static_cast<int32_t>(num_tokens64);
            if (has_token_ids) {
                if (token_ids_td->format.dt != DataType::I32) {
                    throw std::runtime_error("LlamaCppRunner: token ids must be DataType::I32");
                }
                if (token_ids_td->format.shape.size() != 1) {
                    throw std::runtime_error("LlamaCppRunner: token ids must be a 1D tensor");
                }
                if (token_ids_td->format.shape[0] != num_tokens64) {
                    throw std::runtime_error("LlamaCppRunner: token ids length must match embeddings");
                }
                token_ids_ptr = reinterpret_cast<const llama_token*>(token_ids_td->data);
            }
            const int32_t n_batch = llama_n_batch(_ctx.get());
            const int32_t batch_size = std::max<int32_t>(1, n_batch);
            int32_t offset = 0;
            int32_t last_batch_tokens = 0;

            while (offset < num_tokens) {
                const int32_t n = std::min(num_tokens - offset, batch_size);
                const int32_t n_ctx = llama_n_ctx(_ctx.get());
                const int32_t n_ctx_used = static_cast<int32_t>(
                    llama_memory_seq_pos_max(llama_get_memory(_ctx.get()), 0) + 1);

                if (n_ctx_used + n > n_ctx) {
                    throw std::runtime_error("LlamaCppRunner: context size exceeded during embedding prefill");
                }

                const llama_pos* pos_ptr = nullptr;
                std::vector<llama_pos> pos_tmp;

                pos_tmp.resize(static_cast<size_t>(n) * static_cast<size_t>(_embedding_position_count));
                for (int32_t axis = 0; axis < _embedding_position_count; ++axis) {
                    for (int32_t i = 0; i < n; ++i) {
                        pos_tmp[static_cast<size_t>(axis) * static_cast<size_t>(n) + static_cast<size_t>(i)] =
                            n_ctx_used + i;
                    }
                }
                pos_ptr = pos_tmp.data();

                std::vector<int32_t> n_seq_id(static_cast<size_t>(n), 1);
                std::vector<llama_seq_id> seq_id_0(1, 0);
                std::vector<llama_seq_id*> seq_id(static_cast<size_t>(n) + 1, nullptr);
                std::vector<int8_t> logits(static_cast<size_t>(n), 0);

                // When the full hidden-state sequence is requested (e.g. speech decoder transformer),
                // enable logits for EVERY token so that llama_get_embeddings_ith returns valid data
                // for all positions, not just the last one.
                const bool capture_seq = (_out_tensor_idx.count("hidden_state_seq") > 0);
                for (int32_t i = 0; i < n; ++i) {
                    seq_id[static_cast<size_t>(i)] = seq_id_0.data();
                    logits[static_cast<size_t>(i)] = (capture_seq || i == n - 1) ? 1 : 0;
                }
                seq_id[static_cast<size_t>(n)] = nullptr;

                llama_batch batch = {
                    /*n_tokens=*/ n,
                    /*tokens  =*/ nullptr,
                    /*embd    =*/ const_cast<float*>(embd_ptr + static_cast<int64_t>(offset) * n_embd),
                    /*pos     =*/ const_cast<llama_pos*>(pos_ptr),
                    /*n_seq_id=*/ n_seq_id.data(),
                    /*seq_id  =*/ seq_id.data(),
                    /*logits  =*/ logits.data(),
                };

                const int32_t rc = llama_decode(_ctx.get(), batch);
                if (rc != 0) {
                    if(rc == 1) {
                        throw std::runtime_error("LlamaCppRunner: llama_decode returned code 1: no KV slot available; context likely full");
                    } else {
                        throw std::runtime_error("LlamaCppRunner: llama_decode generic error");
                    }
                }
                if (_sampler && token_ids_ptr != nullptr) {
                    for (int32_t i = 0; i < n; ++i) {
                        llama_sampler_accept(_sampler.get(), token_ids_ptr[offset + i]);
                    }
                }

                last_batch_tokens = n;
                offset += n;
            }

            if (!_last_hidden_state.empty() && last_batch_tokens > 0) {
                const float* emb = llama_get_embeddings_ith(_ctx.get(), last_batch_tokens - 1);
                if (emb != nullptr) {
                    std::memcpy(_last_hidden_state.data(), emb, static_cast<size_t>(n_embd) * sizeof(float));
                }
            }
            // Extract full-sequence hidden states when the caller has mapped "hidden_state_seq" (index 3).
            // NOTE: This only works correctly when num_tokens <= n_batch (single decode call).
            // For Qwen3-TTS speech decoder with chunk_size=16 and default n_batch=256 this always holds.
            {
                auto seq_it = _out_tensor_idx.find("hidden_state_seq");
                if (seq_it != _out_tensor_idx.end()) {
                    const size_t seq_idx = seq_it->second;
                    _last_hidden_state_seq.resize(static_cast<size_t>(num_tokens) * static_cast<size_t>(n_embd));
                    for (int32_t i = 0; i < num_tokens; ++i) {
                        const float* emb_i = llama_get_embeddings_ith(_ctx.get(), i);
                        if (emb_i != nullptr) {
                            std::memcpy(
                                _last_hidden_state_seq.data() + static_cast<size_t>(i) * static_cast<size_t>(n_embd),
                                emb_i,
                                static_cast<size_t>(n_embd) * sizeof(float));
                        }
                    }
                    _outputs[seq_idx].data         = _last_hidden_state_seq.data();
                    _outputs[seq_idx].format.shape = { num_tokens, n_embd };
                }
            }
            if (!_last_logits.empty()) {
                const float* logits = llama_get_logits_ith(_ctx.get(), -1);
                if (logits != nullptr) {
                    std::memcpy(_last_logits.data(), logits,
                                static_cast<size_t>(_logits_vocab_size) * sizeof(float));
                }
            }
        }

        if (!_last_pooled_embedding.empty()) {
            const float* pooled_embedding = llama_get_embeddings_seq(_ctx.get(), 0);
            if (pooled_embedding == nullptr) {
                throw std::runtime_error(
                    "LlamaCppRunner: pooled_embedding output requires a model with a configured pooling type");
            }
            std::memcpy(_last_pooled_embedding.data(), pooled_embedding,
                        _last_pooled_embedding.size() * sizeof(float));
        }

        _is_first_run = false;
        _has_new_input = false;
    }

    // Choose a token from those logits (the logits correspond to the current step).
    const bool reached_generation_limit =
        _max_tokens_to_gen >= 0 && _generated_tokens >= _max_tokens_to_gen;
    if (_is_sampler_enabled && !reached_generation_limit) {
        // llama_sampler_sample already calls llama_sampler_accept internally.
        _last_generated_token_id = llama_sampler_sample(_sampler.get(), _ctx.get(), -1);
        if (_outputs.size() > k_output_token_id_idx) {
            _outputs[k_output_token_id_idx].format.shape = { 1 };
        }
    } else if (_is_sampler_enabled) {
        _last_generated_token_id = -1;
        if (_outputs.size() > k_output_token_id_idx) {
            _outputs[k_output_token_id_idx].format.shape = { 0 };
        }
    }

    const auto* vocab = llama_model_get_vocab(_model.get());
    if (_last_generated_token_id >= 0) {
        _is_eog = llama_vocab_is_eog(vocab, _last_generated_token_id);
        if (!_is_eog) {
            ++_generated_tokens;
        }
    } else {
        _is_eog = false;
    }

    if (_is_sampler_enabled && _append_generated_token && !_is_eog) {
        const int32_t n_ctx = llama_n_ctx(_ctx.get());
        const int32_t n_ctx_used = static_cast<int32_t>(llama_memory_seq_pos_max(llama_get_memory(_ctx.get()), 0) + 1);
        if (n_ctx_used + 1 > n_ctx) {
            throw std::runtime_error("LlamaCppRunner: context size exceeded during next_token_id");
        }

        llama_batch batch = llama_batch_get_one(&_last_generated_token_id, 1);

        // Runs the model forward and produces logits for the next step
        const int32_t rc = llama_decode(_ctx.get(), batch);

        if (rc != 0) {
            if(rc == 1) {
                throw std::runtime_error("LlamaCppRunner: llama_decode returned code 1: no KV slot available; context likely full");
            } else {
                throw std::runtime_error("LlamaCppRunner: llama_decode generic error");
            }
        }

        if (!_last_hidden_state.empty()) {
            const float* emb = llama_get_embeddings_ith(_ctx.get(), 0);
            if (emb != nullptr) {
                const int32_t n_embd = llama_model_n_embd_inp(_model.get());
                std::memcpy(_last_hidden_state.data(), emb, static_cast<size_t>(n_embd) * sizeof(float));
            }
        }
        if (!_last_logits.empty()) {
            const float* logits = llama_get_logits_ith(_ctx.get(), -1);
            if (logits != nullptr) {
                std::memcpy(_last_logits.data(), logits,
                            static_cast<size_t>(_logits_vocab_size) * sizeof(float));
            }
        }
    }

}

TensorFormat LlamaCppLMRunner::get_preprocessing_format(const TensorData& input_tensor) {
    if (!_model) {
        throw std::runtime_error("LlamaCppRunner: get_preprocessing_format called before setup()");
    }

    // TODO: Retrieve the input modality from input_key
    // The tensor modality should be stored in the manifest
    // const LMRunnerModality input_modality = get_modality(input_key);
    const LMRunnerModality input_modality = LMRunnerModality::Text;

    switch(input_modality) {
        case LMRunnerModality::Text: {
            return get_tokenize_ids_format(input_tensor);
        }
        break;
        default:
            throw std::runtime_error("LlamaCppRunner: Unsupported modality");
    }
}

TensorFormat LlamaCppLMRunner::get_postprocessing_format(const TensorData& input_tensor) {
    if (!_model) {
        throw std::runtime_error("LlamaCppRunner: get_postprocessing_format called before setup()");
    }

    // TODO: Retrieve the output modality from the output key (manifest).
    const LMRunnerModality output_modality = LMRunnerModality::Text;

    switch (output_modality) {
        case LMRunnerModality::Text:
            return get_detokenize_format(input_tensor);
        default:
            throw std::runtime_error("LlamaCppRunner: Unsupported modality");
    }
}

void LlamaCppLMRunner::preprocessing(const TensorData& input_tensor, TensorData& output_tensor) {
    if (!_model) {
        throw std::runtime_error("LlamaCppRunner: preprocessing called before setup()");
    }

    // TODO: Retrieve the input modality from input_key
    // The tensor modality should be stored in the manifest
    // const LMRunnerModality input_modality = get_modality(input_key);
    const LMRunnerModality input_modality = LMRunnerModality::Text;

    switch(input_modality) {
        case LMRunnerModality::Text: {
            return tokenize_ids(input_tensor, output_tensor);
        }
        break;
        default:
            throw std::runtime_error("LlamaCppRunner: Unsupported modality");
    }
}

void LlamaCppLMRunner::postprocessing(const TensorData& input_tensor, TensorData& output_tensor) {
    if (!_model) {
        throw std::runtime_error("LlamaCppRunner: postprocessing called before setup()");
    }

    // TODO: Retrieve the output modality from the output key (manifest).
    const LMRunnerModality output_modality = LMRunnerModality::Text;

    switch (output_modality) {
        case LMRunnerModality::Text:
            detokenize(input_tensor, output_tensor);
            return;
        default:
            throw std::runtime_error("LlamaCppRunner: Unsupported modality");
    }
}

void LlamaCppLMRunner::save_sequence_state(const std::string& name) {
    if (!_ctx) {
        throw std::runtime_error("LlamaCppRunner: save_sequence_state called before setup()");
    }
    if (name.empty()) {
        throw std::invalid_argument("LlamaCppRunner: sequence state name must not be empty");
    }

    const size_t state_size = llama_state_seq_get_size(_ctx.get(), 0);
    if (state_size == 0) {
        throw std::runtime_error("LlamaCppRunner: failed to size sequence state");
    }

    std::vector<uint8_t> state(state_size);
    const size_t written = llama_state_seq_get_data(_ctx.get(), state.data(), state.size(), 0);
    if (written == 0) {
        throw std::runtime_error("LlamaCppRunner: failed to save sequence state");
    }

    state.resize(written);
    _saved_states[name] = std::move(state);
}

void LlamaCppLMRunner::restore_sequence_state(const std::string& name) {
    if (!_ctx) {
        throw std::runtime_error("LlamaCppRunner: restore_sequence_state called before setup()");
    }

    const auto it = _saved_states.find(name);
    if (it == _saved_states.end()) {
        throw std::out_of_range("LlamaCppRunner: saved sequence state not found: " + name);
    }

    llama_memory_clear(llama_get_memory(_ctx.get()), true);
    const size_t read = llama_state_seq_set_data(_ctx.get(), it->second.data(), it->second.size(), 0);
    if (read == 0) {
        throw std::runtime_error("LlamaCppRunner: failed to restore sequence state: " + name);
    }

    if (_sampler) {
        llama_sampler_reset(_sampler.get());
    }

    _last_generated_token_id = -1;
    _is_eog = false;
    _is_first_run = false;
    _has_new_input = false;
    _generated_tokens = 0;
}

bool LlamaCppLMRunner::is_end_of_generation() const {
    const bool reached_generation_limit =
        _max_tokens_to_gen >= 0 && _generated_tokens >= _max_tokens_to_gen;
    return !_is_first_run && (_is_eog || reached_generation_limit);
}

TensorFormat LlamaCppLMRunner::get_tokenize_ids_format(const TensorData& text) {
    const std::string text_str = tensor_to_string(text);

    const auto* vocab      = llama_model_get_vocab(_model.get());
    const int32_t text_len = static_cast<int32_t>(text_str.size());

    // Check whether it is a new sequence as the llama_token may need to prepend BOS
    const bool is_new_sequence = llama_memory_seq_pos_max(llama_get_memory(_ctx.get()), 0) == -1;

    llama_token mock_input[1];

    // We call llama_tokenize with 0 for the n_tokens_max input to know how many tokens will be generated
    // TODO: Does it change the context???
    const int32_t num_tokens = llama_tokenize(vocab,
                                              text_str.c_str(),
                                              text_len,
                                              &mock_input[0],
                                              0,
                                              is_new_sequence,
                                              /*parse_special*/ true);

    if (num_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("LlamaCppRunner: llama_tokenize overflow");
    }

    if (num_tokens >= 0) {
        throw std::runtime_error("LlamaCppRunner: llama_tokenize failed. Expected negative value");
    }

    TensorFormat format;
    format.dt = DataType::I32;
    format.shape = { -num_tokens };

    return format;
}

void LlamaCppLMRunner::tokenize_ids(const TensorData& text, TensorData& ids) {
    const std::string text_str = tensor_to_string(text);

    const auto* vocab      = llama_model_get_vocab(_model.get());
    const int32_t text_len = static_cast<int32_t>(text_str.size());

    // Check whether it is a new sequence as the llama_token may need to prepend BOS
    const bool is_new_sequence = llama_memory_seq_pos_max(llama_get_memory(_ctx.get()), 0) == -1;

    const auto ids_fmt_expected = get_tokenize_ids_format(text);

    if(ids_fmt_expected.shape[0] != ids.format.shape[0]) {
        throw std::runtime_error("LlamaCppRunner: llama_tokenize failed. Output tensor shape mismatch");
    }

    if(ids_fmt_expected.shape.size() != ids.format.shape.size()) {
        throw std::runtime_error("LlamaCppRunner: llama_tokenize failed. Output tensor number of dimensions mismatch");
    }

    if(ids_fmt_expected.dt != ids.format.dt) {
        throw std::runtime_error("LlamaCppRunner: llama_tokenize failed. Output tensor datatype mismatch");
    }

    const int32_t num_tokens_written = llama_tokenize(vocab,
                                                      text_str.c_str(),
                                                      text_len,
                                                      (llama_token*)ids.data,
                                                      ids.format.shape[0],
                                                      is_new_sequence,
                                                      /*parse_special*/ true);

    if(ids.format.shape[0] != num_tokens_written) {
        throw std::runtime_error("LlamaCppRunner: llama_tokenize failed. The number of tokens written does not match the output shape");
    }
}

TensorFormat LlamaCppLMRunner::get_detokenize_format(const TensorData& token_ids) {
    if (token_ids.data == nullptr) {
        throw std::runtime_error("LlamaCppRunner: postprocessing input tensor has null data");
    }
    if (token_ids.format.dt != DataType::I32) {
        throw std::runtime_error("LlamaCppRunner: postprocessing supports only DataType::I32 tokens");
    }
    if (token_ids.format.shape.empty()) {
        throw std::runtime_error("LlamaCppRunner: postprocessing input tensor has empty shape");
    }
    if (token_ids.format.shape.size() != 1) {
        throw std::runtime_error("LlamaCppRunner: postprocessing supports only 1D token tensors");
    }

    const int64_t n_tokens64 = token_ids.format.shape[0];
    if (n_tokens64 < 0) {
        throw std::runtime_error("LlamaCppRunner: postprocessing input token count must be non-negative");
    }
    if (n_tokens64 > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("LlamaCppRunner: postprocessing input token count exceeds int32_t");
    }

    const auto* vocab = llama_model_get_vocab(_model.get());
    const auto* tokens = reinterpret_cast<const llama_token*>(token_ids.data);
    const int32_t n_tokens = static_cast<int32_t>(n_tokens64);

    char scratch = '\0';
    const int32_t detok_len = llama_detokenize(
        vocab,
        tokens,
        n_tokens,
        &scratch,
        0,
        /*remove_special*/ true,
        /*unparse_special*/ false);

    int64_t nbytes = 0;
    if (detok_len < 0) {
        nbytes = -static_cast<int64_t>(detok_len);
    } else {
        nbytes = detok_len;
    }

    TensorFormat format;
    format.dt = DataType::String;
    format.shape = { nbytes };
    return format;
}

void LlamaCppLMRunner::detokenize(const TensorData& token_ids, TensorData& text) {
    if (text.data == nullptr) {
        throw std::runtime_error("LlamaCppRunner: postprocessing output tensor has null data");
    }

    const TensorFormat required_format = get_detokenize_format(token_ids);

    if (text.format.dt != DataType::String) {
        throw std::runtime_error("LlamaCppRunner: postprocessing output must be DataType::String");
    }
    if (text.format.shape.empty()) {
        throw std::runtime_error("LlamaCppRunner: postprocessing output tensor has empty shape");
    }
    if (text.format.shape[0] < required_format.shape[0]) {
        throw std::runtime_error("LlamaCppRunner: postprocessing output tensor too small");
    }
    if (required_format.shape[0] > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("LlamaCppRunner: postprocessing output exceeds int32_t");
    }

    const auto* vocab = llama_model_get_vocab(_model.get());
    const auto* tokens = reinterpret_cast<const llama_token*>(token_ids.data);
    const int32_t n_tokens = static_cast<int32_t>(token_ids.format.shape[0]);
    char* out_ptr = static_cast<char*>(text.data);
    const int32_t out_len = static_cast<int32_t>(text.format.shape[0]);

    const int32_t detok_len = llama_detokenize(
        vocab,
        tokens,
        n_tokens,
        out_ptr,
        out_len,
        /*remove_special*/ true,
        /*unparse_special*/ false);

    if (detok_len < 0) {
        throw std::runtime_error("LlamaCppRunner: postprocessing output buffer too small");
    }

    text.format.dt = DataType::String;
    if (text.format.shape.empty()) {
        text.format.shape = { detok_len };
    } else {
        text.format.shape[0] = detok_len;
    }
}

void LlamaCppLMRunner::LlamaModelDeleter::operator()(llama_model* m) const noexcept {
    if (m != nullptr) {
        llama_model_free(m);
    }
}

void LlamaCppLMRunner::LlamaContextDeleter::operator()(llama_context* c) const noexcept {
    if (c != nullptr) {
        llama_free(c);
    }
}

void LlamaCppLMRunner::LlamaSamplerDeleter::operator()(llama_sampler* s) const noexcept {
    if (s != nullptr) {
        llama_sampler_free(s);
    }
}

}  // namespace compute_flow

#endif  // COMPUTE_FLOW_HAS_LLAMACPP
