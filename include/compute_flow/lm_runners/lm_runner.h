//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "compute_flow/runners/runner.h"

namespace compute_flow {

/**
 * @brief Supported modalities for the LMRunner.
 */
enum class LMRunnerModality {
    Text,
    Audio,
    Image,
    Video,
    Unknown
};

/**
 * @brief Interface for executing Language Model (LM) engines.
 *
 * The LMRunner defines the execution contract between the brick and a
 * concrete LM engine implementation. It encapsulates model setup,
 * modality preprocessing/postprocessing, inference execution, and
 * state management.
 *
 * @section usage Expected Usage Pattern
 *
 * A typical invocation flow is as follows:
 *
 * @code
 * LMRunner runner;
 *
 * // 1. Initialize the model
 * runner.setup(model_data, profiler);
 *
 * // 2. Get the TensorFormat for the preprocessing output
 * TensorFormat preprocessing_format
 * runner.get_preprocessing_format(input_external);
 *
 * // 2.a Allocate/prepare the memory for the output of the preprocessing call
 *
 * // 3. Convert external input modality to internal representation
 * TensorData input_internal (preprocessing_format);
 * runner.preprocessing(input_tensor
 *                            output_tensor);
 *
 * // 4. Provide internal representation to the model
 * runner.set_input(input_key, output_tensor);
 *
 * // 5. Execute inference
 * runner.run();
 *
 * // 6. Retrieve internal output representation
 * TensorData output_internal =
 *     runner.get_output("text");
 *
 * // 7. Optionally convert to external modality
 * TensorData output_external;
 * runner.postprocessing(output_internal,
 *                              output_external);
 * @endcode
 *
 * @section internal_representation Internal Representation
 *
 * The LM core operates on a model-specific internal representation,
 * which may consist of:
 *   - Discrete token IDs (integer tensors), or
 *   - Continuous embeddings / feature tensors (floating point).
 *
 * The preprocessing and postprocessing methods provide explicit
 * conversion between external modalities and this internal representation.
 *
 * @section advanced_usage Advanced Usage
 *
 * Advanced workflows are supported by exposing the internal representation:
 *   - Prompt caching (reuse of preprocessed inputs)
 *   - Incremental or streaming decoding
 *   - Token-level manipulation or constrained generation
 *
 * Callers may cache or concatenate internal inputs before invoking
 * @ref run(), and may optionally bypass postprocessing when operating
 * directly on the model’s internal outputs.
 *
 * @section state_management State Management
 *
 * Sequence boundaries and KV-cache management are controlled via
 * @ref reset_states(). Preprocessing and postprocessing are purely
 * modality transformations and do not implicitly reset model state.
 *
 * @note The exact behavior of preprocessing, postprocessing, and
 *       supported modalities depends on the concrete LM engine
 *       implementation.
 */
class ILMRunner : public virtual IRunner {
public:
    virtual ~ILMRunner() = default;

    /** @brief LM engine–specific extension interface.
     *
     * The following methods are part of the LM engine abstraction layer.
     * They are exposed to the brick to decouple it from any concrete LM
     * engine implementation (e.g., llama.cpp, LiteRT-LM, or others).
     *
     * By defining these capabilities at the interface level, the brick
     * remains engine-agnostic and can seamlessly switch between different
     * LM backends without requiring modifications to its internal logic.
     *
     * This abstraction ensures:
     *   - Interchangeability of LM engines
     *   - Clear separation between brick logic and engine-specific behavior
     *   - Long-term maintainability and extensibility
     *
     * @note The exact behavior of these methods is defined by the
     *       concrete LM engine implementation.
    */

    /**
     * @brief Get the required TensorFormat for preprocessing.
     *
     * Throws std::runtime_error if the TensorFormat cannot be determined.
     *
     * @param[in]  input_tensor   User-space tensor containing the external input data.
     *
     * @return the required TensorFormat for the preprocessing output
     *
     * @note Buffers are expected to be caller-allocated user-space memory.
     */
    virtual TensorFormat get_preprocessing_format(const TensorData& input_tensor) = 0;

    /**
     * @brief Get the required TensorFormat for postprocessing.
     *
     * Throws std::runtime_error if the TensorFormat cannot be determined.
     *
     * @param[in]  input_tensor   User-space tensor containing the internal output data.
     *
     * @return the required TensorFormat for the postprocessing output
     *
     * @note Buffers are expected to be caller-allocated user-space memory.
     */
    virtual TensorFormat get_postprocessing_format(const TensorData& input_tensor) = 0;

    /**
     * @brief Convert an external input modality into the internal model representation.
     *
     * Transforms the provided input tensor from the specified external
     * modality (e.g., text, audio, image, video) into the internal
     * representation required by the LM engine for inference.
     *
     * Both @p input_tensor and @p output_tensor are hosted in user space
     * (caller-owned memory). The implementation must not assume device-side
     * allocation and must not retain raw pointers beyond the scope of this call
     * unless explicitly documented.
     *
     * Depending on the model architecture, the internal representation may
     * consist of discrete token IDs or continuous embeddings/features.
     *
     * The resulting internal representation is written to @p output_tensor and may
     * be cached, reused, concatenated, or otherwise manipulated by the caller
     * prior to invoking @ref run().
     *
     * @param[in]  input_tensor   User-space tensor containing the external input data.
     * @param[out] output_tensor  User-space tensor populated with the internal model
     *                            representation corresponding to @p input_tensor.
     *
     * @warning The exact internal representation is model-specific and
     *          defined by the underlying LM engine implementation.
     */
    virtual void preprocessing(const TensorData& input_tensor, TensorData& output_tensor) = 0;

    /**
     * @brief Convert the internal model output representation into an external modality.
     *
     * Transforms the model's internal output representation (e.g., token IDs
     * or floating-point tensors) into the requested external modality
     * (e.g., text string, audio waveform, image tensor).
     *
     * Both @p input_tensor and @p output_tensor are hosted in user space
     * (caller-owned memory). The implementation must not assume device-side
     * allocation and must not retain raw pointers beyond the scope of this call
     * unless explicitly documented.
     *
     * This method is typically invoked after @ref run() and @ref get_output()
     * when the caller requires the result in an application-level format.
     *
     * @param[in]  input_tensor    User-space tensor containing the internal model output representation.
     * @param[out] output_tensor   User-space tensor populated with the converted external representation.
     *
     * @note The output tensor will be resized as needed.
     *
     * @warning Conversion behavior depends on the capabilities of the
     *          underlying LM engine and supported modalities.
     */
    virtual void postprocessing(const TensorData& input_tensor, TensorData& output_tensor) = 0;

    /**
     * @brief Save the current sequence continuation state under a caller-defined name.
     *
     * A sequence state captures the backend-owned data needed to continue
     * running the active sequence from its current position. For transformer
     * backends this usually includes the KV cache for the active sequence, but
     * callers must treat it as an opaque LM-runner state rather than a portable
     * KV-cache buffer.
     *
     * This API is intended for prompt-prefix reuse: run a static prefix once,
     * save its sequence state, then restore it before appending request-specific
     * inputs. It is not a full model snapshot and does not save caller-owned
     * input/output tensors.
     *
     * Saved states are runner-owned and remain available until they are
     * overwritten with the same name, the runner is destroyed, or the runner is
     * reconfigured with @ref setup().
     *
     * @param[in] name Stable identifier for the saved sequence state.
     */
    virtual void save_sequence_state(const std::string& name) = 0;

    /**
     * @brief Restore a previously saved sequence continuation state.
     *
     * Restoring replaces the active sequence state with the runner-owned state
     * saved under @p name. After restore, the next @ref run() call continues
     * from the restored sequence position.
     *
     * The caller is responsible for restoring a state created by the same
     * runner configuration and for appending inputs that are valid after that
     * prefix. Restoring a state does not restore caller-owned input/output
     * tensors and does not imply that any already-supplied input tensor should
     * be reused.
     *
     * @param[in] name Stable identifier passed to @ref save_sequence_state().
     */
    virtual void restore_sequence_state(const std::string& name) = 0;

    virtual bool is_end_of_generation() const = 0;
};

// Lightweight adapter that wraps an existing LM runner implementation.
template <class Impl>
class LMRunnerAdapter final : public RunnerAdapter<Impl>, public virtual ILMRunner {
public:
    LMRunnerAdapter() = default;

    LMRunnerAdapter(const LMRunnerAdapter&) = delete;
    LMRunnerAdapter& operator=(const LMRunnerAdapter&) = delete;
    LMRunnerAdapter(LMRunnerAdapter&&) noexcept = default;
    LMRunnerAdapter& operator=(LMRunnerAdapter&&) noexcept = default;

    TensorFormat get_preprocessing_format(const TensorData& input_tensor) override {
        return this->impl().get_preprocessing_format(input_tensor);
    }
    void preprocessing(const TensorData& input_tensor, TensorData& output_tensor) override {
        this->impl().preprocessing(input_tensor, output_tensor);
    }
    TensorFormat get_postprocessing_format(const TensorData& input_tensor) override {
        return this->impl().get_postprocessing_format(input_tensor);
    }
    void postprocessing(const TensorData& input_tensor, TensorData& output_tensor) override {
        this->impl().postprocessing(input_tensor, output_tensor);
    }
    void save_sequence_state(const std::string& name) override { this->impl().save_sequence_state(name); }
    void restore_sequence_state(const std::string& name) override { this->impl().restore_sequence_state(name); }
    bool is_end_of_generation() const override { return this->impl().is_end_of_generation(); }
};

std::unique_ptr<ILMRunner> make_lm_runner(LMRunnerType type);

}  // namespace compute_flow
