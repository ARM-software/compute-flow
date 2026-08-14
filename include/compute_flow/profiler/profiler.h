//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <climits>
#include <cstdint>
#include <iostream>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace compute_flow {

/**
 * @brief Supported output formats for profiling information.
 */
enum class ProfilingFormat {
    /**
     * @brief Human-readable formatted output.
     */
    Pretty,

    /**
     * @brief Comma-separated values format for tooling and analysis.
     */
    Csv
};

/**
 * @brief Time unit used when writing profiling results.
 */
enum class ProfilingUnit {
    /**
     * @brief Seconds.
     */
    Seconds,

    /**
     * @brief Milliseconds.
     */
    Milliseconds,

    /**
     * @brief Microseconds.
     */
    Microseconds,

    /**
     * @brief Nanoseconds.
     */
    Nanoseconds
};

/**
 * @brief Options controlling profiling behavior.
 */
struct ProfilingOptions {
    /**
     * @brief Enable profiling.
     *
     * @note When disabled, profiling-related calls should have minimal overhead.
     */
    bool enabled = true;

    /**
     * @brief Enable per-operation profiling, if supported.
     *
     * @warning This can introduce significant runtime overhead depending
     *          on the backend and number of operations.
     */
    bool enable_op_profiling = false;

    /**
     * @brief Enable annotations in an external timeline profiler.
     *
     * @note This is a no-op when Compute Flow was built without a timeline backend.
     */
    bool enable_timeline_annotations = true;

    /**
     * @brief Output format for profiling information.
     */
    ProfilingFormat format = ProfilingFormat::Pretty;

    /**
     * @brief Time unit used when writing profiling information.
     */
    ProfilingUnit unit = ProfilingUnit::Milliseconds;
};

class Profiler {
public:
    using clock = std::chrono::steady_clock;

    class Scope {
    public:
        Scope() = default;
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;

        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&& other) noexcept;
        ~Scope();

    private:
        friend class Profiler;

        Scope(Profiler& profiler,
              std::uint64_t id,
              std::string runner,
              std::string parent,
              std::string name,
              bool records_scope,
              uint32_t timeline_channel);

        void finish() noexcept;

        Profiler* _profiler = nullptr;
        std::uint64_t _id = 0;
        std::string _runner;
        std::string _parent;
        std::string _name;
        clock::time_point _start;
        bool _records_scope = false;
        uint32_t _timeline_channel = 0;
    };

    Profiler() = default;

    void configure(const ProfilingOptions& options);

    bool is_enabled() const {
        return _options.enabled;
    }

    bool is_per_op_profiling_enabled() const {
        return _options.enable_op_profiling;
    }

    /**
     * @brief Check whether an external timeline backend is compiled in and enabled.
     */
    bool is_timeline_annotations_enabled() const;

    Scope measure(std::string_view name);

    /**
     * @brief Measure a model invocation using a runner.
     *
     * @param runner Runner name.
     * @param name Model name.
     * @return Profiling scope handle.
     */
    Scope measure_model(std::string_view runner, std::string_view name);

    void start_timer(std::string_view name);
    void end_timer(std::string_view name);
    void cancel_active(std::string_view name);

    void start_timer(std::string_view runner, std::string_view name);
    void end_timer(std::string_view runner, std::string_view name);
    void cancel_active(std::string_view runner, std::string_view name);
    void add_nested_sample(std::string_view runner,
                           std::string_view parent,
                           std::string_view name,
                           long long elapsed_ns);

    void reset();

    std::string report(std::size_t runner_width = 16, std::size_t name_width = 50) const;
    std::string report_csv() const;
    std::string report_visual(std::size_t runner_width = 16,
                              std::size_t name_width   = 50,
                              std::size_t bar_width    = 30,
                              std::size_t top_ops      = 5,
                              bool enable_op_profiling = false) const;
    void display_profile(std::ostream& os = std::cout,
                         std::size_t runner_width = 16,
                         std::size_t name_width   = 50) const;
    void display_profile_visual(std::ostream& os = std::cout,
                                std::size_t runner_width = 16,
                                std::size_t name_width   = 50,
                                std::size_t bar_width    = 30,
                                std::size_t top_ops      = 5,
                                bool enable_op_profiling = false) const;
    void display_profile_visual(bool enable_op_profiling) const {
        display_profile_visual(std::cout, 16, 50, 30, 5, enable_op_profiling);
    }
    void write(std::ostream& os) const;

private:
    struct Stats {
        std::string runner;
        std::string parent;
        std::string name;
        long long total_ns = 0;
        long long min_ns   = LLONG_MAX;
        long long max_ns   = 0;
        std::uint64_t count = 0;
        std::vector<clock::time_point> stack;
    };

    struct ActiveScope {
        std::uint64_t id;
        std::string name;
    };

    struct ActiveModel {
        uint64_t id;
    };

    struct ActiveTimerAnnotation {
        std::string runner;
        std::string name;
        uint32_t channel;
    };

    struct StatsSection {
        std::string_view scope;
        std::string_view table;
        std::uint64_t scope_calls = 0;
        const std::vector<Stats>* stats = nullptr;
    };

    struct ScopeStats {
        std::string name;
        std::vector<Stats> current_stats;
        std::vector<Stats> first_call_stats;
        std::vector<Stats> all_call_stats;
        std::uint64_t call_count = 0;
    };

    static constexpr std::string_view kDefaultRunner = "n.a.";

    static const char* unit_suffix(ProfilingUnit unit);
    static double ns_to_unit(long long ns, ProfilingUnit unit);
    static double ns_to_ms(long long ns);
    static std::string fix(double v, int precision = 3);
    static bool match(const Stats& s, std::string_view r, std::string_view n);
    static bool match(const Stats& s, std::string_view r, std::string_view p, std::string_view n);
    static Stats* find(std::vector<Stats>& stats,
                       std::string_view runner,
                       std::string_view parent,
                       std::string_view name);
    static const Stats* find(const std::vector<Stats>& stats,
                             std::string_view runner,
                             std::string_view parent,
                             std::string_view name);
    static Stats& find_or_create(std::vector<Stats>& stats,
                                 std::string_view runner,
                                 std::string_view parent,
                                 std::string_view name);
    static void add_sample(std::vector<Stats>& stats,
                           std::string_view runner,
                           std::string_view parent,
                           std::string_view name,
                           long long elapsed_ns);
    static bool is_scope_stat(const Stats& stats);
    static bool is_model_stat(const Stats& stats);
    static bool is_direct_child_scope_stat(const Stats& stats, std::string_view scope_name);
    static bool has_direct_child_scope_stats(const std::vector<Stats>& stats, std::string_view scope_name);
    static bool is_summary_contributor(const Stats& stats,
                                       std::string_view scope_name,
                                       bool prefer_subscopes);
    static bool is_nested_detail_stat(const Stats& stats,
                                      const Stats& contributor,
                                      std::string_view scope_name);
    static const char* record_type(const Stats& stats, std::string_view scope_name);
    static const Stats* find_scope_stat(const std::vector<Stats>& stats);
    Stats& find_or_create(std::string_view runner, std::string_view parent, std::string_view name);
    Stats* find(std::string_view runner, std::string_view parent, std::string_view name);
    ScopeStats& find_or_create_scope_stats(std::string_view name);
    ScopeStats* find_scope_stats(std::string_view name);
    const ScopeStats* find_scope_stats(std::string_view name) const;
    void begin_scope_recording(std::string_view name);
    void end_scope_recording();
    void finish_scope(const Scope& scope) noexcept;
    void record_sample(std::string_view runner,
                       std::string_view parent,
                       std::string_view name,
                       long long elapsed_ns);
    std::vector<StatsSection> report_sections() const;

    std::vector<Stats> _stats;
    std::vector<ActiveScope> _active_scopes;
    std::vector<ActiveModel> _active_models;
    std::vector<ActiveTimerAnnotation> _active_timer_annotations;
    std::vector<ScopeStats> _scope_stats;
    std::uint64_t _next_scope_id = 1;
    std::string _active_scope_recording;
    bool _is_recording_scope = false;
    ProfilingOptions _options;
};

}  // namespace compute_flow
