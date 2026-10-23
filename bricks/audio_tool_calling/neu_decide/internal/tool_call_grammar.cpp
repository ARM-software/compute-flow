//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "tool_call_grammar.h"

#include <algorithm>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace compute_flow::bricks::audio_tool_calling::neu_decide {
namespace {

enum class ParseStatus { Incomplete, Invalid };

// Checks generated token prefixes, not asset files: incomplete prefixes may continue, invalid ones are rejected.
// Incomplete prefixes are allowed during decoding; NeuDecide returns [] if the token limit is reached.
// A per-parse object shares the input, tool definitions, and cursor across the recursive helpers.
class PrefixParser {
public:
    PrefixParser(std::string_view text, const std::vector<Tool>& tools) : _text(text), _tools(tools) {}

    PrefixResult parse() {
        PrefixResult result;

        try {
            expect('[');
            whitespace();

            if (peek() != ']') {
                while (true) {
                    // Validate syntax without converting argument numbers to a bounded C++ type.
                    call();
                    whitespace();

                    if (peek() == ']') {
                        break;
                    }

                    expect(',');
                }
            }

            expect(']');
            result.valid    = _position == _text.size();
            result.complete = result.valid;
        } catch (ParseStatus status) {
            result.valid = status == ParseStatus::Incomplete;
        }

        return result;
    }

private:
    char peek() const {
        if (_position == _text.size()) {
            throw ParseStatus::Incomplete;
        }

        return _text[_position];
    }

    void whitespace() {
        while (_position < _text.size() &&
               std::string_view(" \n\r\t").find(_text[_position]) != std::string_view::npos) {
            ++_position;
        }
    }

    void expect(char value) {
        whitespace();

        if (peek() != value) {
            throw ParseStatus::Invalid;
        }

        ++_position;
    }

    void literal(std::string_view value) {
        whitespace();

        for (const char character : value) {
            if (peek() != character) {
                throw ParseStatus::Invalid;
            }

            ++_position;
        }
    }

    int64_t choice(const std::vector<std::string>& values) {
        whitespace();

        bool incomplete      = false;
        const auto remaining = _text.substr(_position);

        for (uint64_t index = 0; index < values.size(); ++index) {
            for (const bool ascii : {true, false}) {
                const std::string encoded = nlohmann::ordered_json(values[index]).dump(-1, ' ', ascii);
                const auto num_compare    = std::min(remaining.size(), encoded.size());

                if (remaining.substr(0, num_compare) != std::string_view(encoded).substr(0, num_compare)) {
                    continue;
                }

                if (remaining.size() < encoded.size()) {
                    incomplete = true;
                } else {
                    _position += encoded.size();
                    return static_cast<int64_t>(index);
                }
            }
        }

        throw incomplete ? ParseStatus::Incomplete : ParseStatus::Invalid;
    }

    void call() {
        if (_tools.empty()) {
            throw ParseStatus::Invalid;
        }

        expect('{');
        literal("\"name\"");
        expect(':');
        whitespace();

        std::vector<std::string> names;

        for (const auto& tool : _tools) {
            names.push_back(tool.name);
        }

        const auto tool_index = choice(names);
        const auto& tool      = _tools[tool_index];

        expect(',');
        literal("\"arguments\"");
        expect(':');
        expect('{');
        whitespace();

        if (peek() != '}') {
            uint64_t num_arguments = 0;

            while (true) {
                if (num_arguments >= tool.arguments.size()) {
                    throw ParseStatus::Invalid;
                }

                choice(tool.arguments);
                expect(':');
                value(0);

                ++num_arguments;

                whitespace();

                if (peek() == '}') {
                    break;
                }

                expect(',');
            }
        }

        expect('}');
        expect('}');
    }

    void string() {
        whitespace();

        expect('"');

        while (true) {
            const auto character = static_cast<unsigned char>(peek());
            ++_position;

            if (character == '"') {
                return;
            }

            // Match the upstream llguidance string rule, including its exclusion of DEL.
            if (character < 0x20 || character == 0x7f) {
                throw ParseStatus::Invalid;
            }

            if (character >= 0x80) {
                const int32_t num_bytes = character >= 0xc2 && character <= 0xdf   ? 2
                                          : character >= 0xe0 && character <= 0xef ? 3
                                          : character >= 0xf0 && character <= 0xf4 ? 4
                                                                                   : 0;

                if (num_bytes == 0) {
                    throw ParseStatus::Invalid;
                }

                for (int32_t index = 1; index < num_bytes; ++index) {
                    const auto continuation = static_cast<unsigned char>(peek());

                    if (continuation < 0x80 || continuation > 0xbf ||
                        (index == 1 &&
                         ((character == 0xe0 && continuation < 0xa0) || (character == 0xed && continuation >= 0xa0) ||
                          (character == 0xf0 && continuation < 0x90) || (character == 0xf4 && continuation >= 0x90)))) {
                        throw ParseStatus::Invalid;
                    }

                    ++_position;
                }
            }

            if (character == '\\') {
                const char escaped = peek();
                ++_position;

                if (escaped == 'u') {
                    const auto code = unicode_escape();

                    if (code >= 0xd800 && code <= 0xdbff) {
                        // A high surrogate must be followed by a low surrogate.
                        for (const char expected : std::string_view("\\u")) {
                            if (peek() != expected) {
                                throw ParseStatus::Invalid;
                            }

                            ++_position;
                        }

                        const auto low = unicode_escape(true);

                        if (low < 0xdc00 || low > 0xdfff) {
                            throw ParseStatus::Invalid;
                        }
                    } else if (code >= 0xdc00 && code <= 0xdfff) {
                        throw ParseStatus::Invalid;
                    }
                } else if (std::string_view("\"\\bfnrt").find(escaped) == std::string_view::npos) {
                    throw ParseStatus::Invalid;
                }
            }
        }
    }

    uint32_t unicode_escape(bool low_surrogate = false) {
        uint32_t code = 0;

        for (int32_t digit = 0; digit < 4; ++digit) {
            const char character = peek();
            const auto index     = std::string_view("0123456789abcdefABCDEF").find(character);

            if (index == std::string_view::npos) {
                throw ParseStatus::Invalid;
            }

            code = code * 16 + static_cast<uint32_t>(index >= 16 ? index - 6 : index);
            ++_position;

            const auto remaining_bits = 4 * (3 - digit);
            const auto first          = code << remaining_bits;
            const auto last           = first | ((uint32_t{1} << remaining_bits) - 1);

            if ((low_surrogate && (last < 0xdc00 || first > 0xdfff)) ||
                (!low_surrogate && first >= 0xdc00 && last <= 0xdfff)) {
                throw ParseStatus::Invalid;
            }
        }

        return code;
    }

    void digits() {
        if (peek() < '0' || peek() > '9') {
            throw ParseStatus::Invalid;
        }

        while (_position < _text.size() && _text[_position] >= '0' && _text[_position] <= '9') {
            ++_position;
        }
    }

    void number() {
        if (peek() == '-') {
            ++_position;
        }

        if (peek() == '0') {
            ++_position;
        } else {
            digits();
        }

        if (peek() == '.') {
            ++_position;

            digits();
        }

        if (peek() == 'e' || peek() == 'E') {
            ++_position;

            if (peek() == '+' || peek() == '-') {
                ++_position;
            }

            digits();
        }
    }

    void value(int32_t depth) {
        if (depth > 128) {
            throw ParseStatus::Invalid;
        }

        whitespace();

        switch (peek()) {
            case '"':
                string();
                return;
            case 't':
                literal("true");
                return;
            case 'f':
                literal("false");
                return;
            case 'n':
                literal("null");
                return;
            case '{':
            case '[': {
                const bool object = peek() == '{';
                const char close  = object ? '}' : ']';
                ++_position;

                whitespace();

                if (peek() != close) {
                    while (true) {
                        if (object) {
                            string();
                            expect(':');
                        }

                        value(depth + 1);
                        whitespace();

                        if (peek() == close) {
                            break;
                        }

                        expect(',');
                    }
                }

                expect(close);
                return;
            }
            default:
                number();
                return;
        }
    }

    std::string_view _text;
    const std::vector<Tool>& _tools;
    uint64_t _position = 0;
};

}  // namespace

PrefixResult match_tool_calls(std::string_view text, const std::vector<Tool>& tools) {
    return PrefixParser(text, tools).parse();
}

}  // namespace compute_flow::bricks::audio_tool_calling::neu_decide
