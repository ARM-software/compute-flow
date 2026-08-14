//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "compute_flow/profiler/profiler.h"

#include "timeline_annotations.h"

#include <algorithm>
#include <cmath>
#include <clocale>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <sstream>
#include <utility>

namespace compute_flow {
namespace {

std::string format_percent(double pct) {
    std::ostringstream os;
    os.setf(std::ios::fixed, std::ios::floatfield);
    os << std::setprecision(2) << pct;
    return os.str();
}

std::string csv_escape(std::string_view value) {
    bool needs_quotes = false;
    for (char c : value) {
        if (c == '"' || c == ',' || c == '\n' || c == '\r') {
            needs_quotes = true;
            break;
        }
    }

    if (!needs_quotes) {
        return std::string(value);
    }

    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped += c;
        }
    }
    escaped += "\"";
    return escaped;
}

bool supports_unicode_bars() {
    const auto has_utf8 = [](const char* s) {
        if (s == nullptr) {
            return false;
        }
        const std::string v(s);
        return v.find("UTF-8") != std::string::npos || v.find("utf8") != std::string::npos;
    };

    static std::once_flag locale_init_once;
    std::call_once(locale_init_once, []() {
        // Initialize process locale from environment when available.
        (void)std::setlocale(LC_CTYPE, "");
    });

    if (has_utf8(std::setlocale(LC_CTYPE, nullptr))) {
        return true;
    }

    // Fallback: inspect locale environment variables directly.
    return has_utf8(std::getenv("LC_ALL")) ||
           has_utf8(std::getenv("LC_CTYPE")) ||
           has_utf8(std::getenv("LANG"));
}

std::string bar_for_percent(double pct, std::size_t bar_width, bool unicode_bars) {
    const double clamped = std::max(0.0, std::min(100.0, pct));
    const std::size_t filled = static_cast<std::size_t>(
        std::llround((clamped / 100.0) * static_cast<double>(bar_width)));
    const std::string on = unicode_bars ? "█" : "#";
    const std::string off = unicode_bars ? "░" : "-";

    std::string bar = "[";
    for (std::size_t i = 0; i < bar_width; ++i) {
        bar += (i < filled) ? on : off;
    }
    bar += "]";
    return bar;
}

}  // namespace

Profiler::Scope::Scope(Scope&& other) noexcept {
    *this = std::move(other);
}

Profiler::Scope& Profiler::Scope::operator=(Scope&& other) noexcept {
    if (this != &other) {
        finish();
        _profiler = other._profiler;
        _id = other._id;
        _runner = std::move(other._runner);
        _parent = std::move(other._parent);
        _name = std::move(other._name);
        _start = other._start;
        _records_scope = other._records_scope;
        _timeline_channel = other._timeline_channel;
        other._profiler = nullptr;
        other._id = 0;
        other._records_scope = false;
        other._timeline_channel = 0;
    }
    return *this;
}

Profiler::Scope::~Scope() {
    finish();
}

Profiler::Scope::Scope(Profiler& profiler,
                       std::uint64_t id,
                       std::string runner,
                       std::string parent,
                       std::string name,
                       bool records_scope,
                       uint32_t timeline_channel)
    : _profiler(&profiler)
    , _id(id)
    , _runner(std::move(runner))
    , _parent(std::move(parent))
    , _name(std::move(name))
    , _start(clock::now())
    , _records_scope(records_scope)
    , _timeline_channel(timeline_channel) {}

void Profiler::Scope::finish() noexcept {
    if (_profiler == nullptr) {
        return;
    }
    _profiler->finish_scope(*this);
    _profiler = nullptr;
}

void Profiler::configure(const ProfilingOptions& options) {
    _options = options;
}

bool Profiler::is_timeline_annotations_enabled() const {
    return _options.enable_timeline_annotations && timeline_annotations::is_available();
}

bool Profiler::match(const Stats& s, std::string_view r, std::string_view n) {
    return s.runner == r && s.parent.empty() && s.name == n;
}

bool Profiler::match(const Stats& s, std::string_view r, std::string_view p, std::string_view n) {
    return s.runner == r && s.parent == p && s.name == n;
}

Profiler::Stats& Profiler::find_or_create(std::string_view runner,
                                          std::string_view parent,
                                          std::string_view name) {
    if (auto* p = find(runner, parent, name)) return *p;
    _stats.push_back(Stats{std::string(runner), std::string(parent), std::string(name)});
    return _stats.back();
}

Profiler::Stats* Profiler::find(std::string_view runner,
                                std::string_view parent,
                                std::string_view name) {
    for (auto& s : _stats)
        if (match(s, runner, parent, name)) return &s;
    return nullptr;
}

Profiler::Stats* Profiler::find(std::vector<Stats>& stats,
                                std::string_view runner,
                                std::string_view parent,
                                std::string_view name) {
    for (auto& s : stats)
        if (match(s, runner, parent, name)) return &s;
    return nullptr;
}

const Profiler::Stats* Profiler::find(const std::vector<Stats>& stats,
                                      std::string_view runner,
                                      std::string_view parent,
                                      std::string_view name) {
    for (const auto& s : stats)
        if (match(s, runner, parent, name)) return &s;
    return nullptr;
}

Profiler::Stats& Profiler::find_or_create(std::vector<Stats>& stats,
                                          std::string_view runner,
                                          std::string_view parent,
                                          std::string_view name) {
    if (auto* p = find(stats, runner, parent, name)) return *p;
    stats.push_back(Stats{std::string(runner), std::string(parent), std::string(name)});
    return stats.back();
}

void Profiler::add_sample(std::vector<Stats>& stats,
                          std::string_view runner,
                          std::string_view parent,
                          std::string_view name,
                          long long elapsed_ns) {
    auto& s = find_or_create(stats, runner, parent, name);
    s.total_ns += elapsed_ns;
    s.count += 1;
    if (elapsed_ns < s.min_ns) s.min_ns = elapsed_ns;
    if (elapsed_ns > s.max_ns) s.max_ns = elapsed_ns;
}

bool Profiler::is_scope_stat(const Stats& stats) {
    return stats.runner == kDefaultRunner && stats.parent.empty();
}

bool Profiler::is_model_stat(const Stats& stats) {
    return stats.parent.empty() && !is_scope_stat(stats);
}

bool Profiler::is_direct_child_scope_stat(const Stats& stats, std::string_view scope_name) {
    return stats.runner == kDefaultRunner &&
           stats.parent == scope_name &&
           stats.name.find('/') == std::string::npos;
}

bool Profiler::has_direct_child_scope_stats(const std::vector<Stats>& stats, std::string_view scope_name) {
    for (const auto& s : stats) {
        if (s.count > 0 && is_direct_child_scope_stat(s, scope_name)) {
            return true;
        }
    }
    return false;
}

bool Profiler::is_summary_contributor(const Stats& stats,
                                      std::string_view scope_name,
                                      bool prefer_subscopes) {
    if (stats.count == 0) {
        return false;
    }
    if (prefer_subscopes) {
        return is_direct_child_scope_stat(stats, scope_name);
    }
    return is_model_stat(stats) || is_direct_child_scope_stat(stats, scope_name);
}

bool Profiler::is_nested_detail_stat(const Stats& stats,
                                     const Stats& contributor,
                                     std::string_view scope_name) {
    if (stats.count == 0) {
        return false;
    }

    if (is_direct_child_scope_stat(contributor, scope_name)) {
        if (stats.parent == contributor.name && stats.runner != kDefaultRunner) {
            return true;
        }
        if (stats.runner == contributor.runner && stats.parent == scope_name) {
            const std::string prefix = contributor.name + "/";
            return stats.name.size() > prefix.size() &&
                   stats.name.compare(0, prefix.size(), prefix) == 0;
        }
        return false;
    }

    return stats.runner == contributor.runner && stats.parent == contributor.name;
}

const char* Profiler::record_type(const Stats& stats, std::string_view scope_name) {
    if (is_scope_stat(stats)) {
        return "scope";
    }
    if (stats.runner == kDefaultRunner && stats.parent == scope_name) {
        return "subscope";
    }
    if (is_model_stat(stats)) {
        return "model";
    }
    return "op";
}

const Profiler::Stats* Profiler::find_scope_stat(const std::vector<Stats>& stats) {
    for (const auto& s : stats) {
        if (is_scope_stat(s) && s.count > 0) {
            return &s;
        }
    }
    return nullptr;
}

Profiler::Scope Profiler::measure(std::string_view name) {
    const bool timeline_enabled = is_timeline_annotations_enabled();
    if (!_options.enabled && !timeline_enabled) {
        return {};
    }

    const bool is_top_level = _active_scopes.empty();
    std::string parent;
    std::string scoped_name(name.data(), name.size());
    if (!is_top_level) {
        parent = _active_scopes.front().name;
        scoped_name.clear();
        for (std::size_t i = 1; i < _active_scopes.size(); ++i) {
            if (!scoped_name.empty()) {
                scoped_name += "/";
            }
            scoped_name += _active_scopes[i].name;
        }
        if (!scoped_name.empty()) {
            scoped_name += "/";
        }
        scoped_name.append(name.data(), name.size());
    }

    const bool records_scope = _options.enabled && is_top_level && !_is_recording_scope;
    if (records_scope) {
        begin_scope_recording(name);
    }

    const uint64_t id = _next_scope_id++;
    const uint32_t depth = static_cast<uint32_t>(_active_scopes.size());
    const auto track = is_top_level ? timeline_annotations::Track::Lifecycle
                                    : timeline_annotations::Track::Stage;
    const uint32_t track_depth = is_top_level ? 0 : depth - 1;
    const uint32_t timeline_channel =
        timeline_enabled ? timeline_annotations::begin(track, track_depth, name) : 0;
    _active_scopes.push_back(ActiveScope{id, std::string(name.data(), name.size())});
    return Scope(*this,
                 id,
                 std::string(kDefaultRunner.data(), kDefaultRunner.size()),
                 std::move(parent),
                 std::move(scoped_name),
                 records_scope,
                 timeline_channel);
}

Profiler::Scope Profiler::measure_model(std::string_view runner, std::string_view name) {
    const bool timeline_enabled = is_timeline_annotations_enabled();
    if (!_options.enabled && !timeline_enabled) {
        return {};
    }

    const uint64_t id = _next_scope_id++;
    uint32_t timeline_channel = 0;
    if (timeline_enabled) {
        std::string label(name);
        label += " (";
        label.append(runner.data(), runner.size());
        label += ")";
        timeline_channel = timeline_annotations::begin(
            timeline_annotations::Track::Model,
            static_cast<uint32_t>(_active_models.size() + _active_timer_annotations.size()),
            label);
    }
    _active_models.push_back(ActiveModel{id});
    return Scope(*this,
                 id,
                 std::string(runner),
                 "",
                 std::string(name),
                 false,
                 timeline_channel);
}

void Profiler::start_timer(std::string_view runner, std::string_view name) {
    if (_options.enabled) {
        auto& s = find_or_create(runner, "", name);
        s.stack.emplace_back(clock::now());
    }
    if (is_timeline_annotations_enabled()) {
        std::string label(name);
        label += " (";
        label.append(runner.data(), runner.size());
        label += ")";
        const uint32_t channel = timeline_annotations::begin(
            timeline_annotations::Track::Model,
            static_cast<uint32_t>(_active_models.size() + _active_timer_annotations.size()),
            label);
        _active_timer_annotations.push_back(
            ActiveTimerAnnotation{std::string(runner), std::string(name), channel});
    }
}

void Profiler::end_timer(std::string_view runner, std::string_view name) {
    if (_options.enabled) {
        for (auto& s : _stats) {
            if (match(s, runner, "", name) && !s.stack.empty()) {
                const auto start_tp = s.stack.back();
                s.stack.pop_back();
                const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    clock::now() - start_tp).count();
                record_sample(runner, "", name, ns);
                break;
            }
        }
    }
    for (auto it = _active_timer_annotations.rbegin();
         it != _active_timer_annotations.rend();
         ++it) {
        if (it->runner == runner && it->name == name) {
            timeline_annotations::end(it->channel);
            _active_timer_annotations.erase(std::next(it).base());
            break;
        }
    }
}

void Profiler::cancel_active(std::string_view runner, std::string_view name) {
    if (_options.enabled) {
        for (auto& s : _stats) {
            if (match(s, runner, "", name)) {
                s.stack.clear();
                break;
            }
        }
    }
    for (auto it = _active_timer_annotations.begin(); it != _active_timer_annotations.end();) {
        if (it->runner == runner && it->name == name) {
            timeline_annotations::end(it->channel);
            it = _active_timer_annotations.erase(it);
        } else {
            ++it;
        }
    }
}

void Profiler::add_nested_sample(std::string_view runner,
                                 std::string_view parent,
                                 std::string_view name,
                                 long long elapsed_ns) {
    if (!_options.enabled) {
        return;
    }
    record_sample(runner, parent, name, elapsed_ns);
}

// ---- backward-compatible wrappers (runner="default") ----
void Profiler::start_timer(std::string_view name) {
    start_timer(kDefaultRunner, name);
}

void Profiler::end_timer(std::string_view name) {
    end_timer(kDefaultRunner, name);
}

void Profiler::cancel_active(std::string_view name) {
    cancel_active(kDefaultRunner, name);
}

void Profiler::reset() {
    for (const auto& annotation : _active_timer_annotations) {
        timeline_annotations::end(annotation.channel);
    }
    _stats.clear();
    _active_scopes.clear();
    _active_models.clear();
    _active_timer_annotations.clear();
    _scope_stats.clear();
    _next_scope_id = 1;
    _active_scope_recording.clear();
    _is_recording_scope = false;
}

Profiler::ScopeStats& Profiler::find_or_create_scope_stats(std::string_view name) {
    if (auto* scope = find_scope_stats(name)) {
        return *scope;
    }
    _scope_stats.push_back(ScopeStats{std::string(name)});
    return _scope_stats.back();
}

Profiler::ScopeStats* Profiler::find_scope_stats(std::string_view name) {
    for (auto& scope : _scope_stats) {
        if (scope.name == name) {
            return &scope;
        }
    }
    return nullptr;
}

const Profiler::ScopeStats* Profiler::find_scope_stats(std::string_view name) const {
    for (const auto& scope : _scope_stats) {
        if (scope.name == name) {
            return &scope;
        }
    }
    return nullptr;
}

void Profiler::begin_scope_recording(std::string_view name) {
    if (!_options.enabled) {
        return;
    }
    auto& scope = find_or_create_scope_stats(name);
    scope.current_stats.clear();
    _active_scope_recording.assign(name.data(), name.size());
    _is_recording_scope = true;
}

void Profiler::end_scope_recording() {
    if (!_options.enabled || !_is_recording_scope) {
        return;
    }
    if (auto* scope = find_scope_stats(_active_scope_recording)) {
        scope->call_count += 1;
        if (scope->call_count == 1) {
            scope->first_call_stats = scope->current_stats;
        }
        scope->current_stats.clear();
    }
    _active_scope_recording.clear();
    _is_recording_scope = false;
}

void Profiler::finish_scope(const Scope& scope) noexcept {
    timeline_annotations::end(scope._timeline_channel);
    if (_options.enabled) {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            clock::now() - scope._start).count();
        try {
            record_sample(scope._runner, scope._parent, scope._name, ns);
        } catch (...) {
        }
    }
    _active_scopes.erase(
        std::remove_if(_active_scopes.begin(),
                       _active_scopes.end(),
                       [&scope](const ActiveScope& active) { return active.id == scope._id; }),
        _active_scopes.end());
    _active_models.erase(
        std::remove_if(_active_models.begin(),
                       _active_models.end(),
                       [&scope](const ActiveModel& active) { return active.id == scope._id; }),
        _active_models.end());
    if (scope._records_scope) {
        end_scope_recording();
    }
}

void Profiler::record_sample(std::string_view runner,
                             std::string_view parent,
                             std::string_view name,
                             long long elapsed_ns) {
    add_sample(_stats, runner, parent, name, elapsed_ns);
    if (_is_recording_scope) {
        if (auto* scope = find_scope_stats(_active_scope_recording)) {
            const bool can_attach_to_child_scope = parent.empty() &&
                                                   runner != kDefaultRunner &&
                                                   _active_scopes.size() > 1 &&
                                                   _active_scopes.front().name == _active_scope_recording;
            if (can_attach_to_child_scope) {
                const std::string& child_scope = _active_scopes[1].name;
                add_sample(scope->current_stats, runner, child_scope, name, elapsed_ns);
                add_sample(scope->all_call_stats, runner, child_scope, name, elapsed_ns);
            } else {
                add_sample(scope->current_stats, runner, parent, name, elapsed_ns);
                add_sample(scope->all_call_stats, runner, parent, name, elapsed_ns);
            }
        }
    }
}

std::vector<Profiler::StatsSection> Profiler::report_sections() const {
    std::vector<StatsSection> sections;
    for (const auto& scope : _scope_stats) {
        if (scope.call_count == 0) {
            continue;
        }
        if (scope.call_count == 1) {
            sections.push_back(StatsSection{scope.name, "", scope.call_count, &scope.all_call_stats});
        } else {
            sections.push_back(StatsSection{scope.name, "First Call", scope.call_count, &scope.first_call_stats});
            sections.push_back(StatsSection{scope.name, "All Calls", scope.call_count, &scope.all_call_stats});
        }
    }

    if (sections.empty()) {
        sections.push_back(StatsSection{"", "", 0, &_stats});
    }
    return sections;
}

std::string Profiler::report(std::size_t runner_width, std::size_t name_width) const {
    if (!_options.enabled) {
        return {};
    }

    constexpr int kCallsWidth = 12;
    constexpr int kTotalWidth = 22;
    constexpr int kMetricWidth = 14;

    std::ostringstream os;
    const auto sections = report_sections();
    const std::string unit = unit_suffix(_options.unit);

    for (std::size_t section_index = 0; section_index < sections.size(); ++section_index) {
        const auto& section = sections[section_index];
        const auto& stats = *section.stats;
        const Stats* scope = find_scope_stat(stats);
        const bool prefer_subscopes = !section.scope.empty() &&
                                      has_direct_child_scope_stats(stats, section.scope);

        long long contributor_total_ns = 0;
        for (const auto& s : stats) {
            if (!is_summary_contributor(s, section.scope, prefer_subscopes)) continue;
            contributor_total_ns += s.total_ns;
        }
        const long long scope_overhead_ns = scope != nullptr
                                                ? std::max(0LL, scope->total_ns - contributor_total_ns)
                                                : 0LL;
        const long long summary_total_ns = contributor_total_ns + scope_overhead_ns;
        const long long section_total_ns = scope != nullptr ? scope->total_ns : summary_total_ns;

        if (section_index > 0) {
            os << "\n";
        }
        if (!section.scope.empty()) {
            os << "Scope: " << section.scope << "()";
            if (scope != nullptr) {
                os << " [total " << unit << ": "
                   << fix(ns_to_unit(scope->total_ns, _options.unit)) << "]";
            }
            os << "\n";
        }
        if (!section.table.empty()) {
            os << section.table << "\n";
        }
        if (scope != nullptr) {
            const double scope_total = ns_to_unit(scope->total_ns, _options.unit);
            const double scope_avg = scope_total / static_cast<double>(scope->count);
            os << "Scope stats"
               << " | calls: " << scope->count
               << " | avg " << unit << ": " << fix(scope_avg)
               << " | min " << unit << ": " << fix(ns_to_unit(scope->min_ns, _options.unit))
               << " | max " << unit << ": " << fix(ns_to_unit(scope->max_ns, _options.unit))
               << "\n";
        }

        os << std::left
           << std::setw(static_cast<int>(runner_width)) << "runner"
           << std::setw(static_cast<int>(name_width))   << "contributor"
           << std::right << std::setw(kCallsWidth) << "calls"
           << std::setw(kTotalWidth) << ("total " + unit)
           << std::setw(kMetricWidth) << ("avg " + unit)
           << std::setw(kMetricWidth) << ("min " + unit)
           << std::setw(kMetricWidth) << ("max " + unit) << "\n";

        os << std::string(runner_width + name_width + kCallsWidth + kTotalWidth + kMetricWidth * 3, '-') << "\n";

        for (const auto& s : stats) {
            if (!is_summary_contributor(s, section.scope, prefer_subscopes)) continue;
            const double total = ns_to_unit(s.total_ns, _options.unit);
            const double avg   = total / static_cast<double>(s.count);

            os << std::left
               << std::setw(static_cast<int>(runner_width)) << s.runner
               << std::setw(static_cast<int>(name_width))   << s.name
               << std::right << std::setw(kCallsWidth) << s.count
               << std::setw(kTotalWidth) << fix(total)
               << std::setw(kMetricWidth) << fix(avg)
               << std::setw(kMetricWidth) << fix(ns_to_unit(s.min_ns, _options.unit))
               << std::setw(kMetricWidth) << fix(ns_to_unit(s.max_ns, _options.unit))
               << "\n";

            std::vector<const Stats*> nested_stats;
            nested_stats.reserve(stats.size());
            long long nested_sum_ns = 0;
            for (const auto& nested : stats) {
                if (!is_nested_detail_stat(nested, s, section.scope)) continue;
                nested_stats.push_back(&nested);
                nested_sum_ns += nested.total_ns;
            }
            std::sort(nested_stats.begin(), nested_stats.end(), [](const Stats* a, const Stats* b) {
                return a->total_ns > b->total_ns;
            });

            for (const auto* nested : nested_stats) {
                const double nested_total = ns_to_unit(nested->total_ns, _options.unit);
                const double nested_avg   = nested_total / static_cast<double>(nested->count);
                const double pct_total = s.total_ns > 0
                                             ? (100.0 * static_cast<double>(nested->total_ns) / static_cast<double>(s.total_ns))
                                             : 0.0;
                const std::string nested_total_with_pct = fix(nested_total) + " (" + format_percent(pct_total) + " %)";
                const std::string nested_name = "  " + nested->name;
                os << std::left
                   << std::setw(static_cast<int>(runner_width)) << "-- op"
                   << std::setw(static_cast<int>(name_width))   << nested_name
                   << std::right << std::setw(kCallsWidth) << nested->count
                   << std::setw(kTotalWidth) << nested_total_with_pct
                   << std::setw(kMetricWidth) << fix(nested_avg)
                   << std::setw(kMetricWidth) << fix(ns_to_unit(nested->min_ns, _options.unit))
                   << std::setw(kMetricWidth) << fix(ns_to_unit(nested->max_ns, _options.unit))
                   << "\n";
            }

            const long long overhead_ns = std::max(0LL, s.total_ns - nested_sum_ns);
            const double overhead_total = ns_to_unit(overhead_ns, _options.unit);
            const double overhead_avg = s.count > 0
                                            ? overhead_total / static_cast<double>(s.count)
                                            : 0.0;
            const double overhead_pct = s.total_ns > 0
                                            ? (100.0 * static_cast<double>(overhead_ns) / static_cast<double>(s.total_ns))
                                            : 0.0;
            const std::string overhead_total_with_pct = fix(overhead_total) + " (" + format_percent(overhead_pct) + " %)";
            const std::string overhead_name = "  EXTRA";
            os << std::left
               << std::setw(static_cast<int>(runner_width)) << "-- op"
               << std::setw(static_cast<int>(name_width))   << overhead_name
               << std::right << std::setw(kCallsWidth) << s.count
               << std::setw(kTotalWidth) << overhead_total_with_pct
               << std::setw(kMetricWidth) << fix(overhead_avg)
               << std::setw(kMetricWidth) << ""
               << std::setw(kMetricWidth) << ""
               << "\n";
        }

        if (scope != nullptr) {
            const double overhead_total = ns_to_unit(scope_overhead_ns, _options.unit);
            const double overhead_avg = scope->count > 0
                                            ? overhead_total / static_cast<double>(scope->count)
                                            : 0.0;
            os << std::left
               << std::setw(static_cast<int>(runner_width)) << ""
               << std::setw(static_cast<int>(name_width))   << "Overhead"
               << std::right << std::setw(kCallsWidth) << scope->count
               << std::setw(kTotalWidth) << fix(overhead_total)
               << std::setw(kMetricWidth) << fix(overhead_avg)
               << std::setw(kMetricWidth) << ""
               << std::setw(kMetricWidth) << ""
               << "\n";
        }

        os << std::string(runner_width + name_width + kCallsWidth + kTotalWidth + kMetricWidth * 3, '-') << "\n";
        os << std::left
           << std::setw(static_cast<int>(runner_width)) << ""
           << std::setw(static_cast<int>(name_width))   << (section.scope.empty() ? "Grand Total" : "Scope Total")
           << std::right << std::setw(kCallsWidth) << ""
           << std::setw(kTotalWidth) << fix(ns_to_unit(section_total_ns, _options.unit))
           << std::setw(kMetricWidth) << ""
           << std::setw(kMetricWidth) << ""
           << std::setw(kMetricWidth) << ""
           << "\n";
    }

    return os.str();
}

std::string Profiler::report_csv() const {
    if (!_options.enabled) {
        return {};
    }

    std::ostringstream os;
    const auto sections = report_sections();
    const std::string unit = unit_suffix(_options.unit);
    os << "scope,table,record_type,runner,parent,name,calls,total_" << unit
       << ",avg_" << unit
       << ",min_" << unit
       << ",max_" << unit
       << ",scope_calls\n";

    for (const auto& section : sections) {
        const auto& stats = *section.stats;
        const Stats* scope = find_scope_stat(stats);
        const bool prefer_subscopes = !section.scope.empty() &&
                                      has_direct_child_scope_stats(stats, section.scope);
        long long contributor_total_ns = 0;
        for (const auto& s : stats) {
            if (!is_summary_contributor(s, section.scope, prefer_subscopes)) {
                continue;
            }
            contributor_total_ns += s.total_ns;
        }

        if (scope != nullptr) {
            const double total = ns_to_unit(scope->total_ns, _options.unit);
            const double avg = total / static_cast<double>(scope->count);
            os << csv_escape(section.scope) << ","
               << csv_escape(section.table) << ","
               << "scope,"
               << ",,"
               << csv_escape(scope->name) << ","
               << scope->count << ","
               << fix(total) << ","
               << fix(avg) << ","
               << fix(ns_to_unit(scope->min_ns, _options.unit)) << ","
               << fix(ns_to_unit(scope->max_ns, _options.unit)) << ","
               << section.scope_calls << "\n";
        }

        for (const auto& s : stats) {
            if (s.count == 0) {
                continue;
            }
            if (is_scope_stat(s)) {
                continue;
            }

            const double total = ns_to_unit(s.total_ns, _options.unit);
            const double avg = total / static_cast<double>(s.count);
            os << csv_escape(section.scope) << ","
               << csv_escape(section.table) << ","
               << record_type(s, section.scope) << ","
               << csv_escape(s.runner) << ","
               << csv_escape(s.parent) << ","
               << csv_escape(s.name) << ","
               << s.count << ","
               << fix(total) << ","
               << fix(avg) << ","
               << fix(ns_to_unit(s.min_ns, _options.unit)) << ","
               << fix(ns_to_unit(s.max_ns, _options.unit)) << ","
               << section.scope_calls << "\n";
        }

        if (scope != nullptr) {
            const long long overhead_ns = std::max(0LL, scope->total_ns - contributor_total_ns);
            const double overhead_total = ns_to_unit(overhead_ns, _options.unit);
            const double overhead_avg = scope->count > 0
                                            ? overhead_total / static_cast<double>(scope->count)
                                            : 0.0;
            os << csv_escape(section.scope) << ","
               << csv_escape(section.table) << ","
               << "overhead,"
               << ",,"
               << "Overhead,"
               << scope->count << ","
               << fix(overhead_total) << ","
               << fix(overhead_avg) << ","
               << ",,"
               << section.scope_calls << "\n";
        }
    }

    return os.str();
}

std::string Profiler::report_visual(std::size_t runner_width,
                                    std::size_t name_width,
                                    std::size_t bar_width,
                                    std::size_t top_ops,
                                    bool enable_op_profiling) const {
    if (!_options.enabled) {
        return {};
    }

    constexpr int kCallsWidth  = 12;
    constexpr int kGapWidth    = 1;
    constexpr int kBarPad      = 1;
    constexpr int kPctWidth    = 8; // "100.00%"
    constexpr int kTotalWidth  = 14;
    constexpr int kMetricWidth = 12;

    const bool unicode_bars = supports_unicode_bars();
    const auto sections = report_sections();
    const std::string unit = unit_suffix(_options.unit);
    const int kBarWidth = static_cast<int>(bar_width + 2 + kBarPad); // [....]
    const int total_line_width = static_cast<int>(
        runner_width + name_width + kCallsWidth + kGapWidth + kBarWidth + kPctWidth + kTotalWidth + kMetricWidth * 3);

    std::ostringstream os;

    const auto write_columns = [&]() {
        os << std::left
           << std::setw(static_cast<int>(runner_width)) << "runner"
           << std::setw(static_cast<int>(name_width))   << "contributor"
           << std::right << std::setw(kCallsWidth)      << "calls"
           << std::setw(kGapWidth)                       << ""
           << std::setw(kBarWidth)                       << ""
           << std::setw(kPctWidth)                       << "%"
           << std::setw(kTotalWidth)                     << ("total " + unit)
           << std::setw(kMetricWidth)                    << ("avg " + unit)
           << std::setw(kMetricWidth)                    << ("min " + unit)
           << std::setw(kMetricWidth)                    << ("max " + unit)
           << "\n";
        os << std::string(total_line_width, '-') << "\n";
    };

    const auto write_scope_header = [&](std::string_view name, const Stats* scope) {
        os << "Scope: " << name << "()";
        if (scope != nullptr) {
            os << " [total " << unit << ": "
               << fix(ns_to_unit(scope->total_ns, _options.unit), 2) << "]";
        }
        os << "\n";
        if (scope != nullptr) {
            const double scope_total = ns_to_unit(scope->total_ns, _options.unit);
            const double scope_avg = scope_total / static_cast<double>(scope->count);
            os << "Scope stats"
               << " | calls: " << scope->count
               << " | avg " << unit << ": " << fix(scope_avg, 2)
               << " | min " << unit << ": " << fix(ns_to_unit(scope->min_ns, _options.unit), 2)
               << " | max " << unit << ": " << fix(ns_to_unit(scope->max_ns, _options.unit), 2)
               << "\n";
        }
    };

    const auto write_row = [&](std::string_view runner,
                               const std::string& name,
                               long long calls,
                               long long total_ns,
                               long long min_ns,
                               long long max_ns,
                               long long denominator_ns) {
        const double total = ns_to_unit(total_ns, _options.unit);
        const double avg = calls > 0 ? total / static_cast<double>(calls) : 0.0;
        const double pct = denominator_ns > 0
            ? (100.0 * static_cast<double>(total_ns) / static_cast<double>(denominator_ns))
            : 0.0;
        const std::string pct_str = format_percent(pct) + "%";
        os << std::left
           << std::setw(static_cast<int>(runner_width)) << runner
           << std::setw(static_cast<int>(name_width))   << name
           << std::right << std::setw(kCallsWidth)      << calls
           << std::setw(kGapWidth)                       << ""
           << std::setw(kBarWidth)                       << bar_for_percent(pct, bar_width, unicode_bars)
           << std::right << std::setw(kPctWidth)         << pct_str
           << std::setw(kTotalWidth)                     << fix(total, 2)
           << std::setw(kMetricWidth)                    << fix(avg, 2)
           << std::setw(kMetricWidth)                    << fix(ns_to_unit(min_ns, _options.unit), 2)
           << std::setw(kMetricWidth)                    << fix(ns_to_unit(max_ns, _options.unit), 2)
           << "\n";
    };

    const auto write_empty_metric_row = [&](std::string_view runner,
                                            const std::string& name,
                                            long long total_ns,
                                            long long denominator_ns) {
        const double total = ns_to_unit(total_ns, _options.unit);
        const double pct = denominator_ns > 0
            ? (100.0 * static_cast<double>(total_ns) / static_cast<double>(denominator_ns))
            : 0.0;
        const std::string pct_str = format_percent(pct) + "%";
        os << std::left
           << std::setw(static_cast<int>(runner_width)) << runner
           << std::setw(static_cast<int>(name_width))   << name
           << std::right << std::setw(kCallsWidth)      << ""
           << std::setw(kGapWidth)                       << ""
           << std::setw(kBarWidth)                       << bar_for_percent(pct, bar_width, unicode_bars)
           << std::right << std::setw(kPctWidth)         << pct_str
           << std::setw(kTotalWidth)                     << fix(total, 2)
           << std::setw(kMetricWidth)                    << ""
           << std::setw(kMetricWidth)                    << ""
           << std::setw(kMetricWidth)                    << ""
           << "\n";
    };

    const auto collect_direct_subscopes = [&](const std::vector<Stats>& stats,
                                              std::string_view scope_name) {
        std::vector<const Stats*> rows;
        for (const auto& s : stats) {
            if (s.count > 0 && is_direct_child_scope_stat(s, scope_name)) {
                rows.push_back(&s);
            }
        }
        std::sort(rows.begin(), rows.end(), [](const Stats* a, const Stats* b) {
            return a->total_ns > b->total_ns;
        });
        return rows;
    };

    const auto collect_model_rows = [&](const std::vector<Stats>& stats,
                                        std::string_view parent) {
        std::vector<const Stats*> rows;
        for (const auto& s : stats) {
            if (s.count > 0 && s.runner != kDefaultRunner && s.parent == parent) {
                rows.push_back(&s);
            }
        }
        std::sort(rows.begin(), rows.end(), [](const Stats* a, const Stats* b) {
            return a->total_ns > b->total_ns;
        });
        return rows;
    };

    const auto write_summary_table = [&](std::string_view title,
                                         const std::vector<const Stats*>& rows,
                                         const Stats* scope,
                                         long long total_ns,
                                         std::string_view total_label) {
        os << title << "\n";
        write_columns();

        long long rows_total_ns = 0;
        for (const auto* row : rows) {
            rows_total_ns += row->total_ns;
            write_row(row->runner,
                      row->name,
                      row->count,
                      row->total_ns,
                      row->min_ns,
                      row->max_ns,
                      total_ns);
        }

        if (scope != nullptr) {
            const long long overhead_ns = std::max(0LL, scope->total_ns - rows_total_ns);
            const double overhead = ns_to_unit(overhead_ns, _options.unit);
            const double overhead_avg = scope->count > 0
                                            ? overhead / static_cast<double>(scope->count)
                                            : 0.0;
            const double overhead_pct = total_ns > 0
                ? (100.0 * static_cast<double>(overhead_ns) / static_cast<double>(total_ns))
                : 0.0;
            const std::string overhead_pct_str = format_percent(overhead_pct) + "%";
            os << std::left
               << std::setw(static_cast<int>(runner_width)) << ""
               << std::setw(static_cast<int>(name_width))   << "Overhead"
               << std::right << std::setw(kCallsWidth)      << scope->count
               << std::setw(kGapWidth)                       << ""
               << std::setw(kBarWidth)                       << bar_for_percent(overhead_pct, bar_width, unicode_bars)
               << std::right << std::setw(kPctWidth)         << overhead_pct_str
               << std::setw(kTotalWidth)                     << fix(overhead, 2)
               << std::setw(kMetricWidth)                    << fix(overhead_avg, 2)
               << std::setw(kMetricWidth)                    << ""
               << std::setw(kMetricWidth)                    << ""
               << "\n";
        }

        os << std::string(total_line_width, '-') << "\n";
        os << std::left
           << std::setw(static_cast<int>(runner_width)) << ""
           << std::setw(static_cast<int>(name_width))   << total_label
           << std::right << std::setw(kCallsWidth)      << ""
           << std::setw(kGapWidth)                       << ""
           << std::setw(kBarWidth)                       << ""
           << std::right << std::setw(kPctWidth)         << ""
           << std::setw(kTotalWidth)                     << fix(ns_to_unit(total_ns, _options.unit), 2)
           << std::setw(kMetricWidth)                    << ""
           << std::setw(kMetricWidth)                    << ""
           << std::setw(kMetricWidth)                    << ""
           << "\n";
    };

    const auto write_op_breakdown = [&](const std::vector<Stats>& stats,
                                        const std::vector<const Stats*>& model_rows,
                                        std::string_view section_scope) {
        std::vector<const Stats*> profiled_model_rows;
        for (const auto* model : model_rows) {
            for (const auto& s : stats) {
                if (is_nested_detail_stat(s, *model, section_scope) &&
                    s.name != "Invoke" && s.name != "invoke") {
                    profiled_model_rows.push_back(model);
                    break;
                }
            }
        }
        if (profiled_model_rows.empty()) {
            return;
        }

        os << "Model + Op Breakdown\n";
        write_columns();
        for (const auto* model : profiled_model_rows) {
            os << std::left
               << std::setw(static_cast<int>(runner_width)) << model->runner
               << std::setw(static_cast<int>(name_width))   << model->name
               << std::right << std::setw(kCallsWidth)      << ""
               << std::setw(kGapWidth)                       << ""
               << std::setw(kBarWidth)                       << ""
               << std::right << std::setw(kPctWidth)         << ""
               << std::setw(kTotalWidth)                     << ""
               << std::setw(kMetricWidth)                    << ""
               << std::setw(kMetricWidth)                    << ""
               << std::setw(kMetricWidth)                    << ""
               << "\n";

            std::vector<const Stats*> nested;
            for (const auto& s : stats) {
                if (!is_nested_detail_stat(s, *model, section_scope)) {
                    continue;
                }
                if (s.name == "Invoke" || s.name == "invoke") {
                    continue;
                }
                nested.push_back(&s);
            }
            std::sort(nested.begin(), nested.end(), [](const Stats* a, const Stats* b) {
                return a->total_ns > b->total_ns;
            });

            if (nested.empty()) {
                write_row("", "  Total", model->count, model->total_ns, model->min_ns, model->max_ns, model->total_ns);
                os << "\n";
                continue;
            }

            const std::size_t n_top = std::min(top_ops, nested.size());
            long long shown_ns = 0;
            long long nested_sum_ns = 0;
            for (const auto* op : nested) {
                nested_sum_ns += op->total_ns;
            }
            for (std::size_t i = 0; i < n_top; ++i) {
                const Stats* op = nested[i];
                shown_ns += op->total_ns;
                write_row("", "  " + op->name, op->count, op->total_ns, op->min_ns, op->max_ns, model->total_ns);
            }

            write_empty_metric_row("", "  Extra Ops", std::max(0LL, nested_sum_ns - shown_ns), model->total_ns);
            write_empty_metric_row("", "  Overhead", std::max(0LL, model->total_ns - nested_sum_ns), model->total_ns);
            os << "\n";
        }
        os << std::string(total_line_width, '-') << "\n\n";
    };

    for (std::size_t section_index = 0; section_index < sections.size(); ++section_index) {
        const auto& section = sections[section_index];
        const auto& stats = *section.stats;
        const Stats* scope = find_scope_stat(stats);

        if (section_index > 0) {
            os << "\n";
        }
        if (!section.table.empty()) {
            os << section.table << "\n";
        }

        if (section.scope == "setup") {
            write_scope_header(section.scope, scope);
            const long long setup_total_ns = scope != nullptr ? scope->total_ns : 0;
            write_summary_table("Setup Summary",
                                collect_direct_subscopes(stats, section.scope),
                                scope,
                                setup_total_ns,
                                "Setup Total");
            continue;
        }

        if (section.scope == "run") {
            const Stats* run_impl = find(stats, kDefaultRunner, section.scope, "run_impl");
            if (run_impl != nullptr) {
                write_scope_header("run_impl", run_impl);
                const auto model_rows = collect_model_rows(stats, "run_impl");
                if (enable_op_profiling) {
                    write_op_breakdown(stats, model_rows, section.scope);
                }
                write_summary_table("Model Performance Summary",
                                    model_rows,
                                    run_impl,
                                    run_impl->total_ns,
                                    "Run Total");
                continue;
            }
        }

        if (!section.scope.empty()) {
            write_scope_header(section.scope, scope);
        }
        if (scope != nullptr) {
            const bool prefer_subscopes = !section.scope.empty() &&
                                          has_direct_child_scope_stats(stats, section.scope);
            std::vector<const Stats*> rows;
            for (const auto& s : stats) {
                if (is_summary_contributor(s, section.scope, prefer_subscopes)) {
                    rows.push_back(&s);
                }
            }
            std::sort(rows.begin(), rows.end(), [](const Stats* a, const Stats* b) {
                return a->total_ns > b->total_ns;
            });
            if (section.scope == "run" && enable_op_profiling) {
                write_op_breakdown(stats, rows, section.scope);
            }
            write_summary_table(section.scope.empty() ? "Model Performance Summary" : "Scope Summary",
                                rows,
                                scope,
                                scope->total_ns,
                                section.scope.empty() ? "Grand Total" : "Scope Total");
        } else {
            std::vector<const Stats*> rows;
            long long total_ns = 0;
            for (const auto& s : stats) {
                if (!is_model_stat(s)) {
                    continue;
                }
                rows.push_back(&s);
                total_ns += s.total_ns;
            }
            std::sort(rows.begin(), rows.end(), [](const Stats* a, const Stats* b) {
                return a->total_ns > b->total_ns;
            });
            write_summary_table("Model Performance Summary",
                                rows,
                                nullptr,
                                total_ns,
                                "Grand Total");
        }
    }

    return os.str();
}

void Profiler::display_profile(std::ostream& os, std::size_t runner_width, std::size_t name_width) const {
    os << report(runner_width, name_width);
}

void Profiler::display_profile_visual(std::ostream& os,
                                      std::size_t runner_width,
                                      std::size_t name_width,
                                      std::size_t bar_width,
                                      std::size_t top_ops,
                                      bool enable_op_profiling) const {
    os << report_visual(runner_width, name_width, bar_width, top_ops, enable_op_profiling);
}

void Profiler::write(std::ostream& os) const {
    if (!_options.enabled) {
        return;
    }

    switch (_options.format) {
        case ProfilingFormat::Pretty:
            os << report_visual(16, 50, 30, 5, _options.enable_op_profiling);
            break;
        case ProfilingFormat::Csv:
            os << report_csv();
            break;
    }
}

const char* Profiler::unit_suffix(ProfilingUnit unit) {
    switch (unit) {
        case ProfilingUnit::Seconds:
            return "s";
        case ProfilingUnit::Milliseconds:
            return "ms";
        case ProfilingUnit::Microseconds:
            return "us";
        case ProfilingUnit::Nanoseconds:
            return "ns";
    }
    return "ms";
}

double Profiler::ns_to_unit(long long ns, ProfilingUnit unit) {
    switch (unit) {
        case ProfilingUnit::Seconds:
            return static_cast<double>(ns) / 1'000'000'000.0;
        case ProfilingUnit::Milliseconds:
            return static_cast<double>(ns) / 1'000'000.0;
        case ProfilingUnit::Microseconds:
            return static_cast<double>(ns) / 1'000.0;
        case ProfilingUnit::Nanoseconds:
            return static_cast<double>(ns);
    }
    return ns_to_ms(ns);
}

double Profiler::ns_to_ms(long long ns) {
    return static_cast<double>(ns) / 1'000'000.0;
}

std::string Profiler::fix(double v, int precision) {
    std::ostringstream os;
    os.setf(std::ios::fixed, std::ios::floatfield);
    os << std::setprecision(precision) << v;
    return os.str();
}

}  // namespace compute_flow
