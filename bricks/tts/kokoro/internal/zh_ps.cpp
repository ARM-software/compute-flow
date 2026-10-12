//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "zh_ps.h"

#include <string>
#include <unordered_map>
#include <vector>
#include <cctype>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <regex>
#include <algorithm>
#include <mutex>
#include <locale>
#include <codecvt>

// ---- Jieba ------------------------
#include "cppjieba/Jieba.hpp"
#include "cppjieba/PosTagger.hpp"
// ---- cpp-pinyin ------------------------
#include "cpp-pinyin/Pinyin.h"
#include "cpp-pinyin/G2pglobal.h"

namespace zhps {

static std::string jieba_dict_file = "";
static std::string hmm_model_file = "";
static std::string user_dict_file = "";
static std::string idf_file = "";
static std::string stop_words_file = "";
static std::string pinyin_dict_path = "";

// word: word string
// pos: part-of-speech tag
using WordPos = std::pair<std::string, std::string>;
using SegToken = std::vector<WordPos>;

struct MToken {
    std::string text;
    std::string tag;
    std::string whitespace;
    std::string phonemes;  // empty string if not set

    MToken(const std::string& text, const std::string& tag, const std::string& ws = "")
        : text(text), tag(tag), whitespace(ws), phonemes("") {}
};

// ---------------- transform_an2cn helpers ----------------

static std::string all_num = "0123456789";

static std::map<int, std::string> number_low = {
    {0, "零"}, {1, "一"}, {2, "二"}, {3, "三"}, {4, "四"},
    {5, "五"}, {6, "六"}, {7, "七"}, {8, "八"}, {9, "九"}
};

static std::map<int, std::string> number_up = {
    {0, "零"}, {1, "壹"}, {2, "贰"}, {3, "叁"}, {4, "肆"},
    {5, "伍"}, {6, "陆"}, {7, "柒"}, {8, "捌"}, {9, "玖"}
};

static std::vector<std::string> unit_low_order = {
    "", "十", "百", "千", "万", "十", "百", "千",
    "亿", "十", "百", "千", "万", "十", "百", "千"
};

static std::vector<std::string> unit_up_order = {
    "", "拾", "佰", "仟", "万", "拾", "佰", "仟",
    "亿", "拾", "佰", "仟", "万", "拾", "佰", "仟"
};

static std::string direct_convert(const std::string& inputs) {
    std::string output = "";
    for (char c : inputs) {
        if (c == '.') {
            output += "点";
        } else {
            int digit = c - '0';
            output += number_low[digit];
        }
    }
    return output;
}

static std::string number_to_string(double number_data) {
    std::ostringstream oss;

    // Handle scientific notation
    if (std::abs(number_data) < 1e-10 && number_data != 0.0) {
        oss << std::fixed << std::setprecision(20) << number_data;
    } else {
        oss << number_data;
    }

    std::string str = oss.str();

    // Remove trailing zeros after decimal point
    if (str.find('.') != std::string::npos) {
        str.erase(str.find_last_not_of('0') + 1, std::string::npos);
        if (str.back() == '.') {
            str.pop_back();
        }
    }

    return str;
}

static void check_inputs_is_valid(const std::string& check_data) {
    std::string all_check_keys = all_num + ".-";
    for (char c : check_data) {
        if (all_check_keys.find(c) == std::string::npos) {
            throw std::invalid_argument("this input is not convertable：" + std::string(1, c) + "！");
        }
    }
}

static std::string integer_convert(const std::string& integer_data, const std::string& mode) {
    std::map<int, std::string>* numeral_list;
    std::vector<std::string>* unit_list;

    if (mode == "low") {
        numeral_list = &number_low;
        unit_list = &unit_low_order;
    } else if (mode == "up") {
        numeral_list = &number_up;
        unit_list = &unit_up_order;
    } else if (mode == "direct") {
        return direct_convert(integer_data);
    } else {
        throw std::invalid_argument("error mode: " + mode);
    }

    // Remove leading zeros
    long long int_value = std::stoll(integer_data);
    std::string clean_integer = std::to_string(int_value);

    size_t len_integer_data = clean_integer.length();
    if (len_integer_data > unit_list->size()) {
        throw std::invalid_argument("exceed data range, max support " + std::to_string(unit_list->size()) + " digits");
    }

    std::string output_an = "";
    for (size_t i = 0; i < len_integer_data; i++) {
        int d = clean_integer[i] - '0';
        if (d != 0) {
            output_an += (*numeral_list)[d] + (*unit_list)[len_integer_data - i - 1];
        } else {
            if ((len_integer_data - i - 1) % 4 == 0) {
                output_an += (*numeral_list)[d] + (*unit_list)[len_integer_data - i - 1];
            }

            if (i > 0 && !output_an.empty() && output_an.substr(output_an.length() - 3) != "零") {
                output_an += (*numeral_list)[d];
            }
        }
    }

    // Replace patterns
    auto replace_all = [](std::string& str, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = str.find(from, pos)) != std::string::npos) {
            str.replace(pos, from.length(), to);
            pos += to.length();
        }
    };

    replace_all(output_an, "零零", "零");
    replace_all(output_an, "零万", "万");
    replace_all(output_an, "零亿", "亿");
    replace_all(output_an, "亿万", "亿");

    // Trim leading and trailing zero
    while (!output_an.empty() && output_an.substr(0, 3) == "零") {
        output_an = output_an.substr(3);
    }
    while (!output_an.empty() && output_an.length() >= 3 &&
        output_an.substr(output_an.length() - 3) == "零") {
        output_an = output_an.substr(0, output_an.length() - 3);
    }

    // Handle 10 base case
    if (output_an.length() >= 6 && output_an.substr(0, 6) == "一十") {
        output_an = output_an.substr(3);
    }

    // Handle zero case
    if (output_an.empty()) {
        output_an = "零";
    }

    return output_an;
}

static std::string decimal_convert(const std::string& decimal_data, const std::string& mode) {
    if (decimal_data.empty()) {
        return "";
    }

    std::string data = decimal_data;
    if (data.length() > 16) {
        data = data.substr(0, 16);
    }

    std::string output_an = "点";
    std::map<int, std::string>* numeral_list;

    if (mode == "low") {
        numeral_list = &number_low;
    } else if (mode == "up") {
        numeral_list = &number_up;
    } else {
        throw std::invalid_argument("error mode: " + mode);
    }

    for (char c : data) {
        int digit = c - '0';
        output_an += (*numeral_list)[digit];
    }

    return output_an;
}

static std::string an2cn(const std::string& inputs, const std::string& mode) {
    if (inputs.empty()) {
        throw std::invalid_argument("input data is empty!");
    }

    std::vector<std::string> mode_list = {"low", "up", "rmb", "direct"};
    if (std::find(mode_list.begin(), mode_list.end(), mode) == mode_list.end()) {
        throw std::invalid_argument("mode only supports low, up, rmb, direct!");
    }

    std::string processed_inputs = inputs;

    // Check validity
    check_inputs_is_valid(processed_inputs);

    // Handle sign
    std::string sign = "";
    if (!processed_inputs.empty() && processed_inputs[0] == '-') {
        sign = "负";
        processed_inputs = processed_inputs.substr(1);
    }

    std::string output;

    if (mode == "direct") {
        output = direct_convert(processed_inputs);
    } else {
        // Split integer and decimal parts
        size_t dot_pos = processed_inputs.find('.');

        if (dot_pos == std::string::npos) {
            // No decimal part
            std::string integer_data = processed_inputs;
            if (mode == "rmb") {
                output = integer_convert(integer_data, "up") + "元整";
            } else {
                output = integer_convert(integer_data, mode);
            }
        } else {
            // Has decimal part
            std::string integer_data = processed_inputs.substr(0, dot_pos);
            std::string decimal_data = processed_inputs.substr(dot_pos + 1);

            if (mode == "rmb") { // price format
                // RMB mode - complex logic, simplified here
                std::string int_data = integer_convert(integer_data, "up");
                std::string dec_data = decimal_convert(decimal_data, "up");
                output = int_data + "元" + dec_data;
            } else {
                output = integer_convert(integer_data, mode) +
                        decimal_convert(decimal_data, mode);
            }
        }
    }

    return sign + output;
}

static std::string sub_util(const std::string& inputs, const std::string& method, const std::string& sub_mode) {
    try {

        if (inputs.empty()) {
            return inputs;
        }

        if (method == "an2cn") {
            if (sub_mode == "date") {
                // For dates: years use direct, months/days use low
                std::string result = inputs;

                // Convert year (before 年)
                std::regex year_pattern(R"((\d+)(?=年))");
                std::string temp_result;
                auto it = std::sregex_iterator(result.begin(), result.end(), year_pattern);
                auto end = std::sregex_iterator();
                size_t last_pos = 0;

                for (; it != end; ++it) {
                    std::smatch match = *it;
                    temp_result.append(result, last_pos, match.position() - last_pos);
                    temp_result.append(an2cn(match.str(), "direct"));
                    last_pos = match.position() + match.length();
                }
                temp_result.append(result, last_pos, result.length() - last_pos);
                result = temp_result;

                // Convert other numbers (months, days)
                std::regex num_pattern(R"(\d+)");
                temp_result.clear();
                it = std::sregex_iterator(result.begin(), result.end(), num_pattern);
                last_pos = 0;

                for (; it != end; ++it) {
                    std::smatch match = *it;
                    temp_result.append(result, last_pos, match.position() - last_pos);
                    temp_result.append(an2cn(match.str(), "low"));
                    last_pos = match.position() + match.length();
                }
                temp_result.append(result, last_pos, result.length() - last_pos);
                result = temp_result;

                return result;
            } else if (sub_mode == "fraction") {
                // Convert fraction: numerator/denominator -> denominator分之numerator
                std::regex num_pattern(R"(\d+)");

                std::vector<std::string> parts;
                std::sregex_iterator iter(inputs.begin(), inputs.end(), num_pattern);
                std::sregex_iterator end;

                for (; iter != end; ++iter) {
                    parts.push_back(an2cn(iter->str(), "low"));
                }

                if (parts.size() == 2) {
                    return parts[1] + "分之" + parts[0];
                }
                return inputs;
            } else if (sub_mode == "celsius") {
                // Remove degree C and convert number
                std::string num = inputs.substr(0, inputs.length() - 3); // Remove degree C
                return an2cn(num, "low") + "摄氏度";
            } else if (sub_mode == "percent") {
                // Convert percent: remove % and add 百分之
                std::string num = inputs.substr(0, inputs.length() - 1); // Remove %
                return "百分之" + an2cn(num, "low");
            } else if (sub_mode == "number") {
                return an2cn(inputs, "low");
            }
        }

        return inputs;
    // TODO: cn2an implementation
    } catch (const std::exception& e) {
        // On error, return original input
        return inputs;
    }
}

// ---------------- map_punctuation helpers ----------------
static std::string trim(const std::string &s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
        ++start;

    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1])))
        --end;

    return s.substr(start, end - start);
}

static void replace_all(std::string &s, const std::string &from, const std::string &to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.length(), to);
        pos += to.length();
    }
}

// ---------------- Segmen_cut helpers ----------------
struct JiebaPossegContext {
    cppjieba::Jieba jieba;

    JiebaPossegContext()
        : jieba(jieba_dict_file, hmm_model_file, user_dict_file, idf_file, stop_words_file) {}
};

static JiebaPossegContext& get_jieba_posseg_context() {
    static JiebaPossegContext* ctx = nullptr;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        ctx = new JiebaPossegContext();
    });
    return *ctx;
}

// ---------------- Segmen_cut helpers ----------------
static Pinyin::Pinyin& get_g2p() {
    static std::once_flag init_flag;
    static std::unique_ptr<Pinyin::Pinyin> g2p;

    std::call_once(init_flag, []() {
        Pinyin::setDictionaryPath(pinyin_dict_path);
        g2p = std::make_unique<Pinyin::Pinyin>();
    });

    return *g2p;
}

// ---------------- sandhi_pre_merge_for_modify helpers ----------------

static std::string BU = "不";
static std::string YI = "一";
static const std::unordered_set<std::string> X_ENG = {"x", "eng"};

// Check if all finals have tone 3
static bool all_tone_three(const std::vector<std::string>& finals) {
    return std::all_of(finals.begin(), finals.end(),
        [](const std::string& x) {
            return !x.empty() && x.back() == '3';
        });
}

// Check if a word is reduplication (e.g., AA pattern)
static bool is_reduplication(const std::string& word) {
    // Count UTF-8 characters
    size_t char_count = 0;
    std::vector<std::string> chars;

    for (size_t i = 0; i < word.size(); ) {
        size_t char_len = 1;
        unsigned char c = word[i];

        // UTF-8 character length detection
        if ((c & 0x80) == 0) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        }

        chars.push_back(word.substr(i, char_len));
        i += char_len;
        char_count++;
    }

    return char_count == 2 && chars[0] == chars[1];
}

// Merge "bu" and the word behind it
// if don't merge, "bu" sometimes appears alone according to jieba, which may occur sandhi error
static SegToken merge_bu(const SegToken& seg) {
    SegToken new_seg;

    for (size_t i = 0; i < seg.size(); i++) {
        std::string word = seg[i].first;
        std::string pos = seg[i].second;

        if (X_ENG.find(pos) == X_ENG.end()) {
            std::string last_word;
            if (i > 0) {
                last_word = seg[i - 1].first;
            }
            if (last_word == BU) {
                word = last_word + word;
            }
        }

        std::string next_pos;
        if (i + 1 < seg.size()) {
            next_pos = seg[i + 1].second;
        }

        if (word != BU || next_pos.empty() || X_ENG.find(next_pos) != X_ENG.end()) {
            new_seg.push_back({word, pos});
        }
    }

    return new_seg;
}

// function 1: merge "一" and reduplication words in it's left and right, e.g. "听","一","听" ->"听一听"
// function 2: merge single "一" and the word behind it
// if don't merge, "一" sometimes appears alone according to jieba, which may occur sandhi error
// e.g.
// input seg: [('听', 'v'), ('一', 'm'), ('听', 'v')]
// output seg: [['听一听', 'v']]
static SegToken merge_yi(const SegToken& seg) {
    SegToken new_seg;
    bool skip_next = false;

    // function 1
    for (size_t i = 0; i < seg.size(); i++) {
        if (skip_next) {
            skip_next = false;
            continue;
        }

        std::string word = seg[i].first;
        std::string pos = seg[i].second;

        if (i >= 1 && word == YI && i + 1 < seg.size() &&
            seg[i - 1].first == seg[i + 1].first &&
            seg[i - 1].second == "v" &&
            X_ENG.find(seg[i + 1].second) == X_ENG.end()) {

            new_seg.back().first = new_seg.back().first + YI + seg[i + 1].first;
            skip_next = true;
        } else {
            new_seg.push_back({word, pos});
        }
    }

    SegToken seg2 = new_seg;
    new_seg.clear();

    // function 2
    for (size_t i = 0; i < seg2.size(); i++) {
        std::string word = seg2[i].first;
        std::string pos = seg2[i].second;

        if (!new_seg.empty() && new_seg.back().first == YI &&
            X_ENG.find(pos) == X_ENG.end()) {
            new_seg.back().first = new_seg.back().first + word;
        } else {
            new_seg.push_back({word, pos});
        }
    }

    return new_seg;
}

// Merge er suffix
static SegToken merge_er(const SegToken& seg) {
    SegToken new_seg;

    for (size_t i = 0; i < seg.size(); i++) {
        std::string word = seg[i].first;
        std::string pos = seg[i].second;

        if (i >= 1 && word == "儿" &&
            X_ENG.find(new_seg.back().second) == X_ENG.end()) {
            new_seg.back().first = new_seg.back().first + word;
        } else {
            new_seg.push_back({word, pos});
        }
    }

    return new_seg;
}

// Merge reduplication words
static SegToken merge_reduplication(const  SegToken& seg) {
    SegToken new_seg;

    for (size_t i = 0; i < seg.size(); i++) {
        std::string word = seg[i].first;
        std::string pos = seg[i].second;

        if (!new_seg.empty() && word == new_seg.back().first &&
            X_ENG.find(pos) == X_ENG.end()) {
            new_seg.back().first = new_seg.back().first + word;
        } else {
            new_seg.push_back({word, pos});
        }
    }

    return new_seg;
}

// the first and the second words are all_tone_three
static SegToken merge_continuous_three_tones(const SegToken& seg) {
    SegToken new_seg;
    std::vector<std::vector<std::string>> sub_finals_list;

    for (const auto& [word, pos] : seg) {
        if (X_ENG.find(pos) != X_ENG.end()) {
            sub_finals_list.push_back({"0"});
            continue;
        }

        // std::vector<std::string> orig_finals = lazy_pinyin(word); TODO
        auto& g2p = get_g2p();
        std::vector<std::string> orig_finals = g2p.hanziToPinyin(word, Pinyin::ManTone::Style::FINALS_TONE3).toStdVector();

        // after pypinyin==0.44.0, '嗯' need to be n2
        for (size_t i = 0; i < word.size(); ) {
            size_t char_len = 1;
            unsigned char c = word[i];

            if ((c & 0x80) == 0) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;

            std::string ch = word.substr(i, char_len);
            if (ch == "嗯") {
                size_t idx = i / 3; // Rough approximation for character index
                if (idx < orig_finals.size()) {
                    orig_finals[idx] = "n2";
                }
            }
            i += char_len;
        }

        sub_finals_list.push_back(orig_finals);
    }

    std::vector<bool> merge_last(seg.size(), false);

    for (size_t i = 0; i < seg.size(); i++) {
        std::string word = seg[i].first;
        std::string pos = seg[i].second;

        if (X_ENG.find(pos) == X_ENG.end() && i >= 1 &&
            all_tone_three(sub_finals_list[i - 1]) &&
            all_tone_three(sub_finals_list[i]) &&
            !merge_last[i - 1]) {

            // if the last word is reduplication, not merge
            size_t prev_len = 0, curr_len = 0;

            // Count UTF-8 characters in previous word
            for (size_t j = 0; j < seg[i - 1].first.size(); ) {
                unsigned char c = seg[i - 1].first[j];
                size_t char_len = 1;
                if ((c & 0x80) == 0) char_len = 1;
                else if ((c & 0xE0) == 0xC0) char_len = 2;
                else if ((c & 0xF0) == 0xE0) char_len = 3;
                else if ((c & 0xF8) == 0xF0) char_len = 4;
                j += char_len;
                prev_len++;
            }

            // Count UTF-8 characters in current word
            for (size_t j = 0; j < word.size(); ) {
                unsigned char c = word[j];
                size_t char_len = 1;
                if ((c & 0x80) == 0) char_len = 1;
                else if ((c & 0xE0) == 0xC0) char_len = 2;
                else if ((c & 0xF0) == 0xE0) char_len = 3;
                else if ((c & 0xF8) == 0xF0) char_len = 4;
                j += char_len;
                curr_len++;
            }

            if (!is_reduplication(seg[i - 1].first) &&
                prev_len + curr_len <= 3) {
                new_seg.back().first = new_seg.back().first + word;
                merge_last[i] = true;
            } else {
                new_seg.push_back({word, pos});
            }
        } else {
            new_seg.push_back({word, pos});
        }
    }

    return new_seg;
}

// the last char of first word and the first char of second word is tone_three
static SegToken merge_continuous_three_tones_2(const SegToken& seg) {
    SegToken new_seg;
    std::vector<std::vector<std::string>> sub_finals_list;

    for (const auto& [word, pos] : seg) {
        if (X_ENG.find(pos) != X_ENG.end()) {
            sub_finals_list.push_back({"0"});
            continue;
        }

        auto& g2p = get_g2p();
        std::vector<std::string> orig_finals = g2p.hanziToPinyin(word, Pinyin::ManTone::Style::FINALS_TONE3).toStdVector();

        // Handle '嗯' character
        for (size_t i = 0; i < word.size(); ) {
            size_t char_len = 1;
            unsigned char c = word[i];

            if ((c & 0x80) == 0) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;

            std::string ch = word.substr(i, char_len);
            if (ch == "嗯") {
                size_t idx = i / 3;
                if (idx < orig_finals.size()) {
                    orig_finals[idx] = "n2";
                }
            }
            i += char_len;
        }

        sub_finals_list.push_back(orig_finals);
    }

    std::vector<bool> merge_last(seg.size(), false);

    for (size_t i = 0; i < seg.size(); i++) {
        std::string word = seg[i].first;
        std::string pos = seg[i].second;

        if (X_ENG.find(pos) == X_ENG.end() && i >= 1 &&
            !sub_finals_list[i - 1].empty() &&
            !sub_finals_list[i].empty() &&
            !sub_finals_list[i - 1].back().empty() &&
            !sub_finals_list[i][0].empty() &&
            sub_finals_list[i - 1].back().back() == '3' &&
            sub_finals_list[i][0].back() == '3' &&
            !merge_last[i - 1]) {

            // Count UTF-8 characters
            size_t prev_len = 0, curr_len = 0;

            for (size_t j = 0; j < seg[i - 1].first.size(); ) {
                unsigned char c = seg[i - 1].first[j];
                size_t char_len = 1;
                if ((c & 0x80) == 0) char_len = 1;
                else if ((c & 0xE0) == 0xC0) char_len = 2;
                else if ((c & 0xF0) == 0xE0) char_len = 3;
                else if ((c & 0xF8) == 0xF0) char_len = 4;
                j += char_len;
                prev_len++;
            }

            for (size_t j = 0; j < word.size(); ) {
                unsigned char c = word[j];
                size_t char_len = 1;
                if ((c & 0x80) == 0) char_len = 1;
                else if ((c & 0xE0) == 0xC0) char_len = 2;
                else if ((c & 0xF0) == 0xE0) char_len = 3;
                else if ((c & 0xF8) == 0xF0) char_len = 4;
                j += char_len;
                curr_len++;
            }

            if (!is_reduplication(seg[i - 1].first) &&
                prev_len + curr_len <= 3) {
                new_seg.back().first = new_seg.back().first + word;
                merge_last[i] = true;
            } else {
                new_seg.push_back({word, pos});
            }
        } else {
            new_seg.push_back({word, pos});
        }
    }

    return new_seg;
}

// ---------------- sandhi_modified_tones helpers ----------------

static std::string PUNC = "、：，；。？！""''':,;.?!";

// Initialize must_neural_tone_words
static const std::unordered_set<std::string> must_neural_tone_words = {
    "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头", "馄饨", "风筝",
    "难为", "队伍", "阔气", "闺女", "门道", "锄头", "铺盖", "铃铛", "铁匠", "钥匙", "里脊",
    "里头", "部分", "那么", "道士", "造化", "迷糊", "连累", "这么", "这个", "运气", "过去",
    "软和", "转悠", "踏实", "跳蚤", "跟头", "趔趄", "财主", "豆腐", "讲究", "记性", "记号",
    "认识", "规矩", "见识", "裁缝", "补丁", "衣裳", "衣服", "衙门", "街坊", "行李", "行当",
    "蛤蟆", "蘑菇", "薄荷", "葫芦", "葡萄", "萝卜", "荸荠", "苗条", "苗头", "苍蝇", "芝麻",
    "舒服", "舒坦", "舌头", "自在", "膏药", "脾气", "脑袋", "脊梁", "能耐", "胳膊", "胭脂",
    "胡萝", "胡琴", "胡同", "聪明", "耽误", "耽搁", "耷拉", "耳朵", "老爷", "老实", "老婆",
    "戏弄", "将军", "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火", "累赘", "糨糊", "糊涂",
    "精神", "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语", "笑话", "窟窿",
    "窝囊", "窗户", "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗", "砚台",
    "码头", "石榴", "石头", "石匠", "知识", "眼睛", "眯缝", "眨巴", "眉毛", "相声", "盘算",
    "白净", "痢疾", "痛快", "疟疾", "疙瘩", "疏忽", "畜生", "生意", "甘蔗", "琵琶", "琢磨",
    "琉璃", "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务", "牲口", "牙碜", "牌楼", "爽快",
    "爱人", "热闹", "烧饼", "烟筒", "烂糊", "点心", "炊帚", "灯笼", "火候", "漂亮", "滑溜",
    "溜达", "温和", "清楚", "消息", "浪头", "活泼", "比方", "正经", "欺负", "模糊", "槟榔",
    "棺材", "棒槌", "棉花", "核桃", "栅栏", "柴火", "架势", "枕头", "枇杷", "机灵", "本事",
    "木头", "木匠", "朋友", "月饼", "月亮", "暖和", "明白", "时候", "新鲜", "故事", "收拾",
    "收成", "提防", "挖苦", "挑剔", "指甲", "指头", "拾掇", "拳头", "拨弄", "招牌", "招呼",
    "抬举", "护士", "折腾", "扫帚", "打量", "打算", "打扮", "打听", "打发", "扎实", "扁担",
    "戒指", "懒得", "意识", "意思", "悟性", "怪物", "思量", "怎么", "念头", "念叨", "别人",
    "快活", "忙活", "志气", "心思", "得罪", "张罗", "弟兄", "开通", "应酬", "庄稼", "干事",
    "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌", "差事", "工夫", "岁数", "屁股",
    "尾巴", "少爷", "小气", "小伙", "将就", "对头", "对付", "寡妇", "家伙", "客气", "实在",
    "官司", "学问", "字号", "嫁妆", "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘", "姐夫",
    "妯娌", "妥当", "妖精", "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫",
    "多少", "多么", "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴", "嘱咐", "嘟囔",
    "嘀咕", "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴", "哈欠", "哆嗦", "咳嗽", "和尚",
    "告诉", "告示", "含糊", "吓唬", "后头", "名字", "名堂", "合同", "吆喝", "叫唤", "口袋",
    "厚道", "厉害", "千斤", "包袱", "包涵", "匀称", "勤快", "动静", "动弹", "功夫", "力气",
    "前头", "刺猬", "刺激", "别扭", "利落", "利索", "利害", "分析", "出息", "凑合", "凉快",
    "冷战", "冤枉", "冒失", "养活", "关系", "先生", "兄弟", "便宜", "使唤", "佩服", "作坊",
    "体面", "位置", "似的", "伙计", "休息", "什么", "人家", "亲戚", "亲家", "交情", "云彩",
    "事情", "买卖", "主意", "丫头", "丧气", "两口", "东西", "东家", "世故", "不由", "下水",
    "下巴", "上头", "上司", "丈夫", "丈人", "一辈", "那个", "菩萨", "父亲", "母亲", "咕噜",
    "邋遢", "费用", "冤家", "甜头", "介绍", "荒唐", "大人", "泥鳅", "幸福", "熟悉", "计划",
    "扑腾", "蜡烛", "姥爷", "照顾", "喉咙", "吉他", "弄堂", "蚂蚱", "凤凰", "拖沓", "寒碜",
    "糟蹋", "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记"
};

// Initialize must_not_neural_tone_words
static const std::unordered_set<std::string> must_not_neural_tone_words = {
    "男子", "女子", "分子", "原子", "量子", "莲子", "石子", "瓜子", "电子", "人人", "虎虎",
    "幺幺", "干嘛", "学子", "哈哈", "数数", "袅袅", "局地", "以下", "娃哈哈", "花花草草", "留得",
    "耕地", "想想", "熙熙", "攘攘", "卵子", "死死", "冉冉", "恳恳", "佼佼", "吵吵", "打打",
    "考考", "整整", "莘莘", "落地", "算子", "家家户户", "青青"
};

static std::vector<std::string> split_word(const std::string& word){

    std::vector<std::string> word_list;
    auto& ctx = get_jieba_posseg_context();
    ctx.jieba.CutForSearch(word, word_list);

    // Sort by length (shortest first)
    std::sort(word_list.begin(), word_list.end(),
            [](const std::string& a, const std::string& b) {
                return a.length() < b.length();
            });

    if (word_list.empty()) {
        return {word};
    }

    // Find the first subword and determine split position
    std::string first_subword = word_list[0];
    size_t first_begin_idx = word.find(first_subword);

    std::vector<std::string> new_word_list;
    if (first_begin_idx == 0) {
        // First subword is at the beginning
        std::string second_subword = word.substr(first_subword.length());
        new_word_list.push_back(first_subword);
        if (!second_subword.empty()) {
            new_word_list.push_back(second_subword);
        }
    } else {
        // First subword is not at the beginning
        std::string second_subword = word.substr(0, word.length() - first_subword.length());
        if (!second_subword.empty()) {
            new_word_list.push_back(second_subword);
        }
        new_word_list.push_back(first_subword);
    }

    return new_word_list;
}

// "不" tone sandhi
static std::vector<std::string> bu_sandhi(const std::string& word, const std::vector<std::string>& finals) {
    std::vector<std::string> result = finals;

    // Count number of characters in the word
    size_t char_count = 0;
    for (size_t i = 0; i < word.size(); ) {
        unsigned char c = word[i];
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;
        i += char_len;
        char_count++;
    }

    // e.g. 看不懂
    if (char_count == 3 && result.size() == 3) {
        // Check if middle character is "不"
        size_t char_idx = 0;
        for (size_t i = 0; i < word.size() && char_idx < 2; ) {
            unsigned char c = word[i];
            size_t char_len = 1;
            if ((c & 0x80) == 0) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;

            if (char_idx == 1) {
                std::string current_char = word.substr(i, char_len);
                if (current_char == BU) {
                    // Change tone to 5
                    if (!result[1].empty()) {
                        result[1] = result[1].substr(0, result[1].length() - 1) + "5";
                    }
                    return result;  // Done, don't process further
                }
                break;
            }

            i += char_len;
            char_idx++;
        }
    }

    // For all other cases, check "不" before tone4
    size_t char_idx = 0;
    for (size_t i = 0; i < word.size() && char_idx < result.size(); ) {
        unsigned char c = word[i];
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        std::string current_char = word.substr(i, char_len);

        // "不" before tone4 should be bu2, e.g. 不怕
        if (current_char == BU && char_idx + 1 < result.size()) {
            if (!result[char_idx + 1].empty() && result[char_idx + 1].back() == '4') {
                if (!result[char_idx].empty()) {
                    result[char_idx] = result[char_idx].substr(0, result[char_idx].length() - 1) + "2";
                }
            }
        }

        i += char_len;
        char_idx++;
    }

    return result;
}

// "一" tone sandhi
static std::vector<std::string> yi_sandhi(const std::string& word, const std::vector<std::string>& finals) {
    std::vector<std::string> result = finals;

    // "一" in number sequences, e.g. 一零零, 二一零
    bool all_numeric = true;
    for (size_t i = 0; i < word.size(); ) {
        size_t char_len = 1;
        unsigned char c = word[i];

        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        std::string current_char = word.substr(i, char_len);
        if (current_char != YI && !std::isdigit(static_cast<unsigned char>(word[i]))) {
            // Check if it's a Chinese digit
            if (current_char != "零" && current_char != "〇" &&
                current_char != "一" && current_char != "二" && current_char != "三" &&
                current_char != "四" && current_char != "五" && current_char != "六" &&
                current_char != "七" && current_char != "八" && current_char != "九") {
                all_numeric = false;
                break;
            }
        }
        i += char_len;
    }

    if (word.find(YI) != std::string::npos && all_numeric) {
        return result;
    }

    // Process each character
    size_t char_idx = 0;
    std::vector<std::string> chars;

    for (size_t i = 0; i < word.size(); ) {
        size_t char_len = 1;
        unsigned char c = word[i];

        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        chars.push_back(word.substr(i, char_len));
        i += char_len;
    }

    // "一" between reduplication words should be yi5, e.g. 看一看
    if (chars.size() == 3 && chars[1] == YI && chars[0] == chars[2]) {
        if (result.size() > 1 && !result[1].empty()) {
            result[1] = result[1].substr(0, result[1].length() - 1) + "5";
        }
    }
    // when "一" is ordinal word, it should be yi1
    else if (word.find("第一") == 0) {
        if (result.size() > 1 && !result[1].empty()) {
            result[1] = result[1].substr(0, result[1].length() - 1) + "1";
        }
    }
    else {
        for (size_t i = 0; i < chars.size() && i < result.size(); i++) {
            if (chars[i] == YI && i + 1 < result.size()) {
                // "一" before tone4 should be yi2, e.g. 一段
                if (!result[i + 1].empty() &&
                    (result[i + 1].back() == '4' || result[i + 1].back() == '5')) {
                    if (!result[i].empty()) {
                        result[i] = result[i].substr(0, result[i].length() - 1) + "2";
                    }
                }
                // "一" before non-tone4 should be yi4, e.g. 一天
                else {
                    // "一" 后面如果是标点，还读一声
                    if (i + 1 < chars.size() &&
                        PUNC.find(chars[i + 1]) == std::string::npos) {
                        if (!result[i].empty()) {
                            result[i] = result[i].substr(0, result[i].length() - 1) + "4";
                        }
                    }
                }
            }
        }
    }

    return result;
}

// Neural tone sandhi
static std::vector<std::string> neural_sandhi(const std::string& word, const std::string& pos, const std::vector<std::string>& finals) {

    if (must_not_neural_tone_words.find(word) != must_not_neural_tone_words.end()) {
        return finals;
    }

    std::vector<std::string> result = finals;

    // Parse characters
    std::vector<std::string> chars;
    for (size_t i = 0; i < word.size(); ) {
        size_t char_len = 1;
        unsigned char c = word[i];

        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        chars.push_back(word.substr(i, char_len));
        i += char_len;
    }

    // reduplication words for n. and v. e.g. 奶奶, 试试, 旺旺
    for (size_t j = 0; j < chars.size() && j < result.size(); j++) {
        if (j >= 1 && chars[j] == chars[j - 1] &&
            !pos.empty() && (pos[0] == 'n' || pos[0] == 'v' || pos[0] == 'a')) {
            if (!result[j].empty()) {
                result[j] = result[j].substr(0, result[j].length() - 1) + "5";
            }
        }
    }

    // Sentence-final particles
    if (!chars.empty() && !result.empty()) {
        std::string last_char = chars.back();
        if (last_char == "吧" || last_char == "呢" || last_char == "啊" ||
            last_char == "呐" || last_char == "噻" || last_char == "嘛" ||
            last_char == "吖" || last_char == "嗨" || last_char == "哦" ||
            last_char == "哒" || last_char == "滴" || last_char == "哩" ||
            last_char == "哟" || last_char == "喽" || last_char == "啰" ||
            last_char == "耶" || last_char == "喔" || last_char == "诶") {
            if (!result.back().empty()) {
                result.back() = result.back().substr(0, result.back().length() - 1) + "5";
            }
        }
        else if (last_char == "的" || last_char == "地" || last_char == "得") {
            if (!result.back().empty()) {
                result.back() = result.back().substr(0, result.back().length() - 1) + "5";
            }
        }
        // e.g. 走了, 看着, 去过
        else if (chars.size() == 1 &&
                (last_char == "了" || last_char == "着" || last_char == "过") &&
                (pos == "ul" || pos == "uz" || pos == "ug")) {
            if (!result.back().empty()) {
                result.back() = result.back().substr(0, result.back().length() - 1) + "5";
            }
        }
        else if (chars.size() > 1 &&
                (last_char == "们" || last_char == "子") &&
                (pos == "r" || pos == "n")) {
            if (!result.back().empty()) {
                result.back() = result.back().substr(0, result.back().length() - 1) + "5";
            }
        }
        // e.g. 桌上, 地下
        else if (chars.size() > 1 &&
                (last_char == "上" || last_char == "下") &&
                (pos == "s" || pos == "l" || pos == "f")) {
            if (!result.back().empty()) {
                result.back() = result.back().substr(0, result.back().length() - 1) + "5";
            }
        }
        // e.g. 上来, 下去
        else if (chars.size() > 1 &&
                (last_char == "来" || last_char == "去") &&
                chars.size() >= 2) {
            std::string second_last = chars[chars.size() - 2];
            if (second_last == "上" || second_last == "下" || second_last == "进" ||
                second_last == "出" || second_last == "回" || second_last == "过" ||
                second_last == "起" || second_last == "开") {
                if (!result.back().empty()) {
                    result.back() = result.back().substr(0, result.back().length() - 1) + "5";
                }
            }
        }
    }

    // Handle "个" as measure word
    size_t ge_idx = std::string::npos;
    for (size_t i = 0; i < chars.size(); i++) {
        if (chars[i] == "个") {
            ge_idx = i;
            break;
        }
    }

    if (ge_idx != std::string::npos && ge_idx < result.size()) {
        if ((ge_idx >= 1 && !chars[ge_idx - 1].empty() &&
            (std::isdigit(static_cast<unsigned char>(chars[ge_idx - 1][0])) ||
            chars[ge_idx - 1] == "几" || chars[ge_idx - 1] == "有" ||
            chars[ge_idx - 1] == "两" || chars[ge_idx - 1] == "半" ||
            chars[ge_idx - 1] == "多" || chars[ge_idx - 1] == "各" ||
            chars[ge_idx - 1] == "整" || chars[ge_idx - 1] == "每" ||
            chars[ge_idx - 1] == "做" || chars[ge_idx - 1] == "是")) ||
            word == "个") {
            if (!result[ge_idx].empty()) {
                result[ge_idx] = result[ge_idx].substr(0, result[ge_idx].length() - 1) + "5";
            }
        }
    }

    // Check must_neural_tone_words
    if (must_neural_tone_words.find(word) != must_neural_tone_words.end()) {
        if (!result.empty() && !result.back().empty()) {
            result.back() = result.back().substr(0, result.back().length() - 1) + "5";
        }
    } else if (chars.size() >= 2) {
        std::string last_two = chars[chars.size() - 2] + chars[chars.size() - 1];
        if (must_neural_tone_words.find(last_two) != must_neural_tone_words.end()) {
            if (!result.empty() && !result.back().empty()) {
                result.back() = result.back().substr(0, result.back().length() - 1) + "5";
            }
        }
    }

    // Handle word splitting for conventional neural tone
    std::vector<std::string> word_list = split_word(word);
    if (word_list.size() > 1 && result.size() > 0) {
        std::vector<std::vector<std::string>> finals_list;
        size_t pos_idx = 0;

        for (const auto& w : word_list) {
            // Count characters in this subword
            size_t char_count = 0;
            for (size_t i = 0; i < w.size(); ) {
                unsigned char c = w[i];
                size_t char_len = 1;
                if ((c & 0x80) == 0) char_len = 1;
                else if ((c & 0xE0) == 0xC0) char_len = 2;
                else if ((c & 0xF0) == 0xE0) char_len = 3;
                else if ((c & 0xF8) == 0xF0) char_len = 4;
                i += char_len;
                char_count++;
            }

            std::vector<std::string> sub_finals;
            for (size_t i = 0; i < char_count && pos_idx + i < result.size(); i++) {
                sub_finals.push_back(result[pos_idx + i]);
            }
            finals_list.push_back(sub_finals);
            pos_idx += char_count;
        }

        for (size_t i = 0; i < word_list.size() && i < finals_list.size(); i++) {
            if (must_neural_tone_words.find(word_list[i]) != must_neural_tone_words.end()) {
                if (!finals_list[i].empty() && !finals_list[i].back().empty()) {
                    finals_list[i].back() = finals_list[i].back().substr(0, finals_list[i].back().length() - 1) + "5";
                }
            } else if (word_list[i].length() >= 2) {
                // Get last 2 characters
                std::vector<std::string> w_chars;
                for (size_t j = 0; j < word_list[i].size(); ) {
                    size_t char_len = 1;
                    unsigned char c = word_list[i][j];
                    if ((c & 0x80) == 0) char_len = 1;
                    else if ((c & 0xE0) == 0xC0) char_len = 2;
                    else if ((c & 0xF0) == 0xE0) char_len = 3;
                    else if ((c & 0xF8) == 0xF0) char_len = 4;
                    w_chars.push_back(word_list[i].substr(j, char_len));
                    j += char_len;
                }

                if (w_chars.size() >= 2) {
                    std::string last_two = w_chars[w_chars.size() - 2] + w_chars[w_chars.size() - 1];
                    if (must_neural_tone_words.find(last_two) != must_neural_tone_words.end()) {
                        if (!finals_list[i].empty() && !finals_list[i].back().empty()) {
                            finals_list[i].back() = finals_list[i].back().substr(0, finals_list[i].back().length() - 1) + "5";
                        }
                    }
                }
            }
        }

        // Merge finals_list back
        result.clear();
        for (const auto& sub : finals_list) {
            result.insert(result.end(), sub.begin(), sub.end());
        }
    }

    return result;
}

// Tone 3 sandhi
static std::vector<std::string> three_sandhi(const std::string& word,
                                                    const std::vector<std::string>& finals) {
    std::vector<std::string> result = finals;

    if (result.size() == 2 && all_tone_three(result)) {
        if (!result[0].empty()) {
            result[0] = result[0].substr(0, result[0].length() - 1) + "2";
        }
    }
    else if (result.size() == 3) {
        std::vector<std::string> word_list = split_word(word);

        if (all_tone_three(result)) {
            // disyllabic + monosyllabic, e.g. 蒙古/包
            if (!word_list.empty() && word_list[0].length() >= 2) {
                size_t first_char_count = 0;
                for (size_t i = 0; i < word_list[0].size(); ) {
                    unsigned char c = word_list[0][i];
                    size_t char_len = 1;
                    if ((c & 0x80) == 0) char_len = 1;
                    else if ((c & 0xE0) == 0xC0) char_len = 2;
                    else if ((c & 0xF0) == 0xE0) char_len = 3;
                    else if ((c & 0xF8) == 0xF0) char_len = 4;
                    i += char_len;
                    first_char_count++;
                }

                if (first_char_count == 2) {
                    if (!result[0].empty()) {
                        result[0] = result[0].substr(0, result[0].length() - 1) + "2";
                    }
                    if (!result[1].empty()) {
                        result[1] = result[1].substr(0, result[1].length() - 1) + "2";
                    }
                }
                // monosyllabic + disyllabic, e.g. 纸/老虎
                else if (first_char_count == 1) {
                    if (!result[1].empty()) {
                        result[1] = result[1].substr(0, result[1].length() - 1) + "2";
                    }
                }
            }
        }
        else {
            // Split finals based on word_list lengths
            std::vector<std::vector<std::string>> finals_list;
            if (!word_list.empty()) {
                size_t pos_idx = 0;

                for (const auto& w : word_list) {
                    size_t char_count = 0;
                    for (size_t i = 0; i < w.size(); ) {
                        unsigned char c = w[i];
                        size_t char_len = 1;
                        if ((c & 0x80) == 0) char_len = 1;
                        else if ((c & 0xE0) == 0xC0) char_len = 2;
                        else if ((c & 0xF0) == 0xE0) char_len = 3;
                        else if ((c & 0xF8) == 0xF0) char_len = 4;
                        i += char_len;
                        char_count++;
                    }

                    std::vector<std::string> sub_finals;
                    for (size_t i = 0; i < char_count && pos_idx + i < result.size(); i++) {
                        sub_finals.push_back(result[pos_idx + i]);
                    }
                    finals_list.push_back(sub_finals);
                    pos_idx += char_count;
                }

                if (finals_list.size() == 2) {
                    for (size_t i = 0; i < finals_list.size(); i++) {
                        auto& sub = finals_list[i];
                        // e.g. 所有/人
                        if (all_tone_three(sub) && sub.size() == 2) {
                            if (!sub[0].empty()) {
                                sub[0] = sub[0].substr(0, sub[0].length() - 1) + "2";
                            }
                        }
                        // e.g. 好/喜欢
                        else if (i == 1 && !all_tone_three(sub) &&
                                !sub.empty() && !sub[0].empty() && sub[0].back() == '3' &&
                                !finals_list[0].empty() && !finals_list[0].back().empty() &&
                                finals_list[0].back().back() == '3') {
                            finals_list[0].back() = finals_list[0].back().substr(0, finals_list[0].back().length() - 1) + "2";
                        }
                    }

                    result.clear();
                    for (const auto& sub : finals_list) {
                        result.insert(result.end(), sub.begin(), sub.end());
                    }
                }
            }
        }
    }
    // split idiom into two words whose length is 2
    else if (result.size() == 4) {
        std::vector<std::vector<std::string>> finals_list = {
            {result[0], result[1]},
            {result[2], result[3]}
        };

        result.clear();
        for (auto& sub : finals_list) {
            if (all_tone_three(sub)) {
                if (!sub[0].empty()) {
                    sub[0] = sub[0].substr(0, sub[0].length() - 1) + "2";
                }
            }
            result.insert(result.end(), sub.begin(), sub.end());
        }
    }

    return result;
}

// ---------------- process_zh_tokens_to_phonemes helpers ----------------

// merge_erhua helpers for
// Get UTF-32 character at index in UTF-8 string
static char32_t get_char_at_index(const std::string& str, size_t index) {
    size_t byte_pos = 0;
    size_t char_idx = 0;

    while (byte_pos < str.length() && char_idx < index) {
        unsigned char c = str[byte_pos];
        if ((c & 0x80) == 0) { byte_pos += 1; }
        else if ((c & 0xE0) == 0xC0) { byte_pos += 2; }
        else if ((c & 0xF0) == 0xE0) { byte_pos += 3; }
        else if ((c & 0xF8) == 0xF0) { byte_pos += 4; }
        else { byte_pos += 1; }
        char_idx++;
    }

    if (byte_pos >= str.length() || char_idx != index) {
        return 0;
    }

    // Decode UTF-8 character at byte_pos
    unsigned char c = str[byte_pos];
    char32_t result = 0;

    if ((c & 0x80) == 0) {
        result = c;
    } else if ((c & 0xE0) == 0xC0) {
        result = ((c & 0x1F) << 6) | (str[byte_pos + 1] & 0x3F);
    } else if ((c & 0xF0) == 0xE0) {
        result = ((c & 0x0F) << 12) |
                ((str[byte_pos + 1] & 0x3F) << 6) |
                (str[byte_pos + 2] & 0x3F);
    } else if ((c & 0xF8) == 0xF0) {
        result = ((c & 0x07) << 18) |
                ((str[byte_pos + 1] & 0x3F) << 12) |
                ((str[byte_pos + 2] & 0x3F) << 6) |
                (str[byte_pos + 3] & 0x3F);
    }

    return result;
}

// Get last N characters of a UTF-8 string
static std::string get_last_n_chars(const std::string& str, size_t n) {
    if (n == 0) return "";

    // Count UTF-8 characters from the end
    size_t byte_pos = str.length();
    size_t char_count = 0;

    while (byte_pos > 0 && char_count < n) {
        byte_pos--;
        // If this is the start of a UTF-8 character
        if ((str[byte_pos] & 0xC0) != 0x80) {
            char_count++;
        }
    }

    return str.substr(byte_pos);
}

// Get UTF-8 string length in characters (not bytes)
static size_t utf8_length(const std::string& str) {
    size_t count = 0;
    for (size_t i = 0; i < str.length(); ) {
        unsigned char c = str[i];
        if ((c & 0x80) == 0) { i += 1; }
        else if ((c & 0xE0) == 0xC0) { i += 2; }
        else if ((c & 0xF0) == 0xE0) { i += 3; }
        else if ((c & 0xF8) == 0xF0) { i += 4; }
        else { i += 1; } // Invalid UTF-8
        count++;
    }
    return count;
}

static std::pair<std::vector<std::string>, std::vector<std::string>> merge_erhua(
    std::vector<std::string> initials,
    std::vector<std::string> finals,
    const std::string& word,
    const std::string& pos,
    const std::unordered_set<std::string>& must_erhua,
    const std::unordered_set<std::string>& not_erhua
) {
    /**
     * Do erhua.
     */

    // fix er1
    for (size_t i = 0; i < finals.size(); ++i) {
        const std::string& phn = finals[i];
        if (i == finals.size() - 1 &&
            get_char_at_index(word, i) == U'儿' &&
            phn == "er1") {
            finals[i] = "er2";
        }
    }

    // 发音 (pronunciation check)
    if (must_erhua.find(word) == must_erhua.end() &&
        (not_erhua.find(word) != not_erhua.end() ||
        pos == "a" || pos == "j" || pos == "nr")) {
        return {initials, finals};
    }

    // "……" 等情况直接返回 (return directly for cases like "……")
    size_t word_len = utf8_length(word);
    if (finals.size() != word_len) {
        return {initials, finals};
    }

    // assert len(finals) == len(word)
    // In production code, you might want to throw an exception or log an error

    // 不发音 (silent erhua)
    std::vector<std::string> new_initials;
    std::vector<std::string> new_finals;

    for (size_t i = 0; i < finals.size(); ++i) {
        const std::string& phn = finals[i];

        if (i == finals.size() - 1 &&
            get_char_at_index(word, i) == U'儿' &&
            (phn == "er2" || phn == "er5") &&
            not_erhua.find(get_last_n_chars(word, 2)) == not_erhua.end() &&
            !new_finals.empty()) {
            // Insert 'R' before the last character of the last final
            std::string& last_final = new_finals.back();
            if (!last_final.empty()) {
                last_final = last_final.substr(0, last_final.length() - 1) +
                            "R" +
                            last_final.substr(last_final.length() - 1);
            }
        } else {
            new_initials.push_back(initials[i]);
            new_finals.push_back(phn);
        }
    }

    return {new_initials, new_finals};
}

static bool is_chinese_char_range(const std::string& word) {
    if (word.empty()) return false;

    char32_t min_char = 0xFFFFFFFF;
    char32_t max_char = 0;

    size_t byte_pos = 0;
    while (byte_pos < word.length()) {
        unsigned char c = word[byte_pos];
        char32_t ch = 0;

        if ((c & 0x80) == 0) {
            ch = c;
            byte_pos += 1;
        } else if ((c & 0xE0) == 0xC0 && byte_pos + 1 < word.length()) {
            ch = ((c & 0x1F) << 6) | (word[byte_pos + 1] & 0x3F);
            byte_pos += 2;
        } else if ((c & 0xF0) == 0xE0 && byte_pos + 2 < word.length()) {
            ch = ((c & 0x0F) << 12) |
                ((word[byte_pos + 1] & 0x3F) << 6) |
                (word[byte_pos + 2] & 0x3F);
            byte_pos += 3;
        } else if ((c & 0xF8) == 0xF0 && byte_pos + 3 < word.length()) {
            ch = ((c & 0x07) << 18) |
                ((word[byte_pos + 1] & 0x3F) << 12) |
                ((word[byte_pos + 2] & 0x3F) << 6) |
                (word[byte_pos + 3] & 0x3F);
            byte_pos += 4;
        } else {
            // Invalid UTF-8, skip
            byte_pos += 1;
            continue;
        }

        if (ch < min_char) min_char = ch;
        if (ch > max_char) max_char = ch;
    }

    if (min_char == 0xFFFFFFFF) return false;

    // Check if all characters are in the Chinese range (U+4E00 to U+9FFF)
    return (min_char >= 0x4E00 && max_char <= 0x9FFF);
}

static const std::unordered_set<std::string> punc_set = {
    "、", "：", "，", "；", "。", "？", "！", """, """, "'", "'",
    ":", ",", ";", ".", "?", "!"
};

// Chinese phoneme to bopomofo mapping
static std::string get_zh_map(const std::string& phoneme, const std::string& unk) {
    static const std::unordered_map<std::string, std::string> zh_map = {
        {"a", "ㄚ"}, {"ai", "ㄞ"}, {"an", "ㄢ"}, {"ang", "ㄤ"}, {"ao", "ㄠ"},
        {"b", "ㄅ"}, {"c", "ㄘ"}, {"ch", "ㄔ"}, {"d", "ㄉ"},
        {"e", "ㄜ"}, {"ei", "ㄟ"}, {"en", "ㄣ"}, {"eng", "ㄥ"}, {"er", "ㄦ"},
        {"f", "ㄈ"}, {"g", "ㄍ"}, {"h", "ㄏ"},
        {"i", "ㄧ"}, {"ia", "ㄧㄚ"}, {"ian", "ㄧㄢ"}, {"iang", "ㄧㄤ"},
        {"iao", "ㄧㄠ"}, {"ie", "ㄧㄝ"}, {"in", "ㄧㄣ"}, {"ing", "ㄧㄥ"},
        {"iong", "ㄩㄥ"}, {"iu", "ㄧㄡ"},
        {"j", "ㄐ"}, {"k", "ㄎ"}, {"l", "ㄌ"}, {"m", "ㄇ"}, {"n", "ㄋ"},
        {"o", "ㄛ"}, {"ong", "ㄨㄥ"}, {"ou", "ㄡ"},
        {"p", "ㄆ"}, {"q", "ㄑ"}, {"r", "ㄖ"}, {"s", "ㄙ"}, {"sh", "ㄕ"},
        {"t", "ㄊ"}, {"u", "ㄨ"}, {"ua", "ㄨㄚ"}, {"uai", "ㄨㄞ"},
        {"uan", "ㄨㄢ"}, {"uang", "ㄨㄤ"}, {"ue", "ㄩㄝ"}, {"ui", "ㄨㄟ"},
        {"un", "ㄨㄣ"}, {"uo", "ㄨㄛ"}, {"v", "ㄩ"}, {"van", "ㄩㄢ"},
        {"ve", "ㄩㄝ"}, {"vn", "ㄩㄣ"},
        {"w", "ㄨ"}, {"x", "ㄒ"}, {"y", "ㄧ"}, {"z", "ㄗ"}, {"zh", "ㄓ"},
        {"ii", "ㄭ"}, {"iii", "ㄭ"},
        {"iou", "ㄧㄡ"}, {"uei", "ㄨㄟ"}, {"uen", "ㄨㄣ"},
        {"1", "1"}, {"2", "2"}, {"3", "3"}, {"4", "4"}, {"5", "5"}
    };

    auto it = zh_map.find(phoneme);
    if (it != zh_map.end()) {
        return it->second;
    }
    return unk;
}

static std::string transform_an2cn(const std::string &text) {

    std::string output = text;
    std::string method = "an2cn";

    // Process dates
    {
        std::regex date_pattern(R"((\d{2,4}年)?(\d{1,2}月)?(\d{1,2}日)?)");
        std::string result;
        auto it = std::sregex_iterator(output.begin(), output.end(), date_pattern);
        auto end = std::sregex_iterator();
        size_t last_pos = 0;

        for (; it != end; ++it) {
            std::smatch match = *it;
            if (!match.str().empty() && match.str().find_first_of("0123456789") != std::string::npos) {
                result.append(output, last_pos, match.position() - last_pos);
                result.append(sub_util(match.str(), method, "date"));
                last_pos = match.position() + match.length();
            }
        }
        result.append(output, last_pos, output.length() - last_pos);
        output = result;
    }

    // Process fractions
    {
        std::regex frac_pattern(R"(\d+/\d+)");
        std::string result;
        auto it = std::sregex_iterator(output.begin(), output.end(), frac_pattern);
        auto end = std::sregex_iterator();
        size_t last_pos = 0;

        for (; it != end; ++it) {
            std::smatch match = *it;
            result.append(output, last_pos, match.position() - last_pos);
            result.append(sub_util(match.str(), method, "fraction"));
            last_pos = match.position() + match.length();
        }
        result.append(output, last_pos, output.length() - last_pos);
        output = result;
    }

    // Process percentages
    {
        std::regex percent_pattern(R"(-?(\d+\.)?\d+%)");
        std::string result;
        auto it = std::sregex_iterator(output.begin(), output.end(), percent_pattern);
        auto end = std::sregex_iterator();
        size_t last_pos = 0;

        for (; it != end; ++it) {
            std::smatch match = *it;
            result.append(output, last_pos, match.position() - last_pos);
            result.append(sub_util(match.str(), method, "percent"));
            last_pos = match.position() + match.length();
        }
        result.append(output, last_pos, output.length() - last_pos);
        output = result;
    }

    // Process celsius
    {
        std::regex celsius_pattern(R"(\d+℃)");
        std::string result;
        auto it = std::sregex_iterator(output.begin(), output.end(), celsius_pattern);
        auto end = std::sregex_iterator();
        size_t last_pos = 0;

        for (; it != end; ++it) {
            std::smatch match = *it;
            result.append(output, last_pos, match.position() - last_pos);
            result.append(sub_util(match.str(), method, "celsius"));
            last_pos = match.position() + match.length();
        }
        result.append(output, last_pos, output.length() - last_pos);
        output = result;
    }

    // Process regular numbers
    {
        std::regex number_pattern(R"(-?(\d+\.)?\d+)");
        std::string result;
        auto it = std::sregex_iterator(output.begin(), output.end(), number_pattern);
        auto end = std::sregex_iterator();
        size_t last_pos = 0;

        for (; it != end; ++it) {
            std::smatch match = *it;
            result.append(output, last_pos, match.position() - last_pos);
            result.append(sub_util(match.str(), method, "number"));
            last_pos = match.position() + match.length();
        }
        result.append(output, last_pos, output.length() - last_pos);
        output = result;
    }

    return output;
}

static std::string map_punctuation(const std::string &text) {
    std::string out = text;

    replace_all(out, "、", ", ");
    replace_all(out, "，", ", ");

    replace_all(out, "。", ". ");
    replace_all(out, "．", ". ");
    replace_all(out, "！", "! ");
    replace_all(out, "：", ": ");
    replace_all(out, "；", "; ");
    replace_all(out, "？", "? ");
    replace_all(out, "«", " “");
    replace_all(out, "»", "” ");

    replace_all(out, "《", " “");
    replace_all(out, "》", "” ");
    replace_all(out, "「", " “");
    replace_all(out, "」", "” ");

    replace_all(out, "【", " “");
    replace_all(out, "】", "” ");
    replace_all(out, "（", " (");
    replace_all(out, "）", ") ");

    return trim(out);
}

static SegToken segment_cut(const std::string& text){
    auto& ctx = get_jieba_posseg_context();
    std::vector<std::pair<std::string, std::string>> raw;
    ctx.jieba.Tag(text, raw);

    SegToken result;
    result.reserve(raw.size());
    for (auto& p : raw) {
        result.push_back(std::move(p));
    }
    return result;
}

static SegToken sandhi_pre_merge_for_modify(const SegToken& seg) {
    SegToken result = seg;
    result = merge_bu(result);
    result = merge_yi(result);
    result = merge_reduplication(result);
    result = merge_continuous_three_tones(result);
    result = merge_continuous_three_tones_2(result);
    result = merge_er(result);
    return result;
}

static std::pair<std::vector<std::string>, std::vector<std::string>> get_initials_finals(const std::string& word) {
    // const auto applicationDirPath = std::filesystem::current_path() / "pinyin_dict";
    // Pinyin::setDictionaryPath(applicationDirPath);
    // std::cout << "Dictionary Path: " << applicationDirPath << std::endl;

    // auto g2p = std::make_unique<Pinyin::Pinyin>();
    auto& g2p = get_g2p();

    std::vector<std::string> initials = g2p.hanziToPinyin(word, Pinyin::ManTone::Style::INITIALS).toStdVector();
    std::vector<std::string> finals   = g2p.hanziToPinyin(word, Pinyin::ManTone::Style::FINALS_TONE3).toStdVector();

    // Find all indices where the character is "嗯" and set corresponding finals to "n2"
    size_t char_index = 0;
    for (size_t i = 0; i < word.size(); ) {
        unsigned char c = word[i];

        // Determine UTF-8 character length
        size_t char_len;
        if ((c & 0x80) == 0) {
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
        } else {
            char_len = 1; // Invalid UTF-8, skip single byte
        }

        // Check if current character is "嗯" and update finals directly
        if (word.compare(i, char_len, "嗯") == 0 && char_index < finals.size()) {
            finals[char_index] = "n2";
        }

        i += char_len;
        char_index++;
    }

    // Process initials and finals to handle special 'i' pronunciations
    for (size_t i = 0; i < initials.size() && i < finals.size(); i++) {
        const std::string& c = initials[i];
        std::string& v = finals[i];  // Non-const reference to modify in place

        // Check if final matches pattern 'i' followed by a digit (i0, i1, i2, i3, i4, i5)
        if (v.length() >= 2 && v[0] == 'i' && std::isdigit(v[1])) {
            if (c == "z" || c == "c" || c == "s") {
                // zi, ci, si -> zii, cii, sii (apical vowel)
                v.replace(0, 1, "ii");
            } else if (c == "zh" || c == "ch" || c == "sh" || c == "r") {
                // zhi, chi, shi, ri -> zhiii, chiii, shiii, riii (retroflex vowel)
                v.replace(0, 1, "iii");
            }
        }
    }

    return {initials, finals};
}

static std::vector<std::string> sandhi_modified_tone(const std::string& word, const std::string& pos, std::vector<std::string>& finals) {
    std::vector<std::string> result = finals;
    result = bu_sandhi(word, result);
    result = yi_sandhi(word, result);
    result = neural_sandhi(word, pos, result);
    result = three_sandhi(word, result);
    return result;
}

void set_paths(const std::string& jieba_dict_file_in,
               const std::string& hmm_model_file_in,
               const std::string& user_dict_file_in,
               const std::string& idf_file_in,
               const std::string& stop_words_file_in,
               const std::string& pinyin_dict_path_in) {
    jieba_dict_file  = jieba_dict_file_in;
    hmm_model_file   = hmm_model_file_in;
    user_dict_file   = user_dict_file_in;
    idf_file         = idf_file_in;
    stop_words_file  = stop_words_file_in;
    pinyin_dict_path = pinyin_dict_path_in;
}

std::string process_zh_tokens_to_phonemes(const std::string& prompt, bool with_erhua, std::string& unk_symbol) {

    static const std::unordered_set<std::string> must_erhua = {};  // Add specific words if needed
    static const std::unordered_set<std::string> not_erhua = {};   // Add specific words if needed

    std::string text = transform_an2cn(prompt);
    std::string mapped_text = map_punctuation(text);
    SegToken seg_cut = segment_cut(mapped_text);
    SegToken seg_modified = sandhi_pre_merge_for_modify(seg_cut);

    std::vector<MToken> tokens;

    for (const auto& [word, pos_orig] : seg_modified) {
        std::string pos = pos_orig;

        // If pos is 'x' and word contains Chinese characters in range U+4E00 to U+9FFF
        if (pos == "x" && is_chinese_char_range(word)) {
            pos = "X";
        }
        // If pos is not 'x' but word is in punctuation set
        else if (pos != "x" && punc_set.find(word) != punc_set.end()) {
            pos = "x";
        }

        MToken tk(word, pos);

        // Handle punctuation ('x') and English ('eng') tags
        if (pos == "x" || pos == "eng") {
            // If word is not whitespace
            if (!std::all_of(word.begin(), word.end(), [](unsigned char c) { return std::isspace(c); })) {
                // If it's punctuation in the punc set
                if (pos == "x" && punc_set.find(word) != punc_set.end()) {
                    tk.phonemes = word;
                }
                tokens.push_back(tk);
            }
            // If word is whitespace and tokens is not empty
            else if (!tokens.empty()) {
                tokens.back().whitespace += word;
            }
            continue;
        }
        // If tokens is not empty and last token is not 'x' or 'eng' and has no whitespace
        else if (!tokens.empty() &&
                    tokens.back().tag != "x" &&
                    tokens.back().tag != "eng" &&
                    tokens.back().whitespace.empty()) {
            tokens.back().whitespace = "/";
        }

        auto [sub_initials, sub_finals] = get_initials_finals(word);
        auto modified_finals = sandhi_modified_tone(word, pos, sub_finals);

        if (with_erhua) {
            auto [new_initials, new_finals] = merge_erhua(sub_initials, sub_finals, word, pos, must_erhua, not_erhua);
            sub_initials = new_initials;
            sub_finals = new_finals;
        }

        // Build phones list
        std::vector<std::string> phones;
        for (size_t i = 0; i < sub_initials.size() && i < sub_finals.size(); ++i) {
            const std::string& c = sub_initials[i];
            const std::string& v = sub_finals[i];

            // NOTE: post process for pypinyin outputs
            // we discriminate i, ii and iii
            if (!c.empty()) {
                phones.push_back(c);
            }
            // replace punctuation by ` `
            // if c and c in self.punc:
            //     phones.append(c)
            if (!v.empty() && (punc_set.find(v) == punc_set.end() || v != c)) {
                phones.push_back(v);
            }
        }

        // Join with '_'
        std::string phones_str;
        for (size_t i = 0; i < phones.size(); ++i) {
            if (i > 0) phones_str += "_";
            phones_str += phones[i];
        }

        // Replace '_eR' with '_er'
        size_t pos_er = phones_str.find("_eR");
        while (pos_er != std::string::npos) {
            phones_str.replace(pos_er, 3, "_er");
            pos_er = phones_str.find("_eR", pos_er + 3);
        }

        // Replace 'R' with '_R' (but not if already preceded by '_')
        size_t pos_r = 0;
        while ((pos_r = phones_str.find("R", pos_r)) != std::string::npos) {
            if (pos_r == 0 || phones_str[pos_r - 1] != '_') {
                phones_str.replace(pos_r, 1, "_R");
                pos_r += 2;
            } else {
                pos_r += 1;
            }
        }

        // Add '_' before digits using regex
        std::regex digit_regex(R"((?=\d))");
        phones_str = std::regex_replace(phones_str, digit_regex, "_");

        // Split by '_'
        std::vector<std::string> phone_parts;
        std::stringstream ss(phones_str);
        std::string part;
        while (std::getline(ss, part, '_')) {
            if (!part.empty()) {
                phone_parts.push_back(part);
            }
        }

        // Map using ZH_MAP
        std::string result_phonemes;
        for (const auto& p : phone_parts) {
            result_phonemes += get_zh_map(p, unk_symbol);
        }

        tk.phonemes = result_phonemes;
        tokens.push_back(tk);
    }

    std::string result;
    for (const auto& tk : tokens) {
        result += (tk.phonemes.empty() ? unk_symbol : tk.phonemes) + tk.whitespace;
    }

    return result;
}

}  // namespace zhps
