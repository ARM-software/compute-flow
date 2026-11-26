//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "compute_flow/cli_examples/internal/wav.h"

namespace {

using Bytes = std::vector<uint8_t>;

// Build little-endian fixtures directly so the tests do not depend on an external audio writer.
void append_u16(Bytes& bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

void append_u32(Bytes& bytes, uint32_t value) {
    append_u16(bytes, static_cast<uint16_t>(value));
    append_u16(bytes, static_cast<uint16_t>(value >> 16));
}

Bytes make_extension(uint16_t subformat, uint16_t valid_bits) {
    Bytes extension;

    append_u16(extension, 22);  // cbSize: extension bytes after this field.
    append_u16(extension, valid_bits);
    append_u32(extension, 4);  // Front-center mono channel.
    append_u32(extension, subformat);
    extension.insert(extension.end(), {0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71});
    return extension;
}

Bytes make_wav(uint16_t format, uint16_t bits, const Bytes& samples, const Bytes& extension = {}) {
    const uint32_t format_size = 16 + static_cast<uint32_t>(extension.size());
    const uint32_t data_size   = static_cast<uint32_t>(samples.size());
    Bytes wav                  = {'R', 'I', 'F', 'F'};

    // RIFF size includes WAVE, both chunk headers and padding, but excludes its own eight-byte header.
    append_u32(wav, 20 + format_size + (format_size & 1U) + data_size + (data_size & 1U));
    wav.insert(wav.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    append_u32(wav, format_size);
    append_u16(wav, format);
    append_u16(wav, 1);
    append_u32(wav, 16000);
    append_u32(wav, 16000 * (bits / 8));
    append_u16(wav, bits / 8);
    append_u16(wav, bits);
    wav.insert(wav.end(), extension.begin(), extension.end());
    if (format_size & 1U) {
        wav.push_back(0);
    }
    wav.insert(wav.end(), {'d', 'a', 't', 'a'});
    append_u32(wav, data_size);
    wav.insert(wav.end(), samples.begin(), samples.end());
    if (data_size & 1U) {
        wav.push_back(0);
    }
    return wav;
}

/** @brief Run WAV decoder cases using a shared temporary file and collect failures. */
class WavTests {
public:
    /**
     * @brief Set the temporary file used by the test cases.
     * @param[in] path File overwritten by check() and removed on destruction.
     */
    explicit WavTests(const std::string& path) : _path(path) {}

    /** @brief Remove the temporary WAV file without throwing on cleanup errors. */
    ~WavTests() {
        std::error_code error;
        std::filesystem::remove(_path, error);
    }

    /**
     * @brief Write a WAV fixture and verify its decoding or rejection.
     *
     * Accepted fixtures must decode to the five expected finite samples at 16 kHz.
     * Rejected fixtures must throw std::runtime_error. Mismatches are recorded for result().
     *
     * @param[in] name Case name included in failure diagnostics.
     * @param[in] wav Complete RIFF/WAVE file bytes, including any deliberate corruption.
     * @param[in] should_accept Whether the decoder is expected to accept the fixture.
     * @throws std::runtime_error If the fixture cannot be written to the temporary file.
     */
    void check(const std::string& name, const Bytes& wav, bool should_accept = true) {
        {
            std::ofstream stream(_path, std::ios::binary);
            if (!stream.write(reinterpret_cast<const char*>(wav.data()), static_cast<std::streamsize>(wav.size()))) {
                throw std::runtime_error("Cannot write test WAV: " + _path);
            }
        }

        std::vector<float> audio;
        int64_t sample_rate = 0;
        bool accepted       = false;

        try {
            sample_rate = compute_flow::bricks::internal::read_mono_wav(_path, audio);
            accepted    = true;
        } catch (const std::runtime_error&) {
        }

        if (accepted != should_accept) {
            fail(name, should_accept ? "unexpected rejection" : "unexpected acceptance");
            return;
        }
        if (!accepted) {
            return;
        }

        // Every accepted fixture represents the same waveform, regardless of its sample encoding.
        const std::vector<float> expected = {0.0F, 0.5F, -0.5F, -1.0F, 0.25F};

        if (sample_rate != 16000 || audio.size() != expected.size()) {
            fail(name, "incorrect sample rate or sample count");
            return;
        }
        for (uint32_t index = 0; index < expected.size(); ++index) {
            if (!std::isfinite(audio[index]) || std::abs(audio[index] - expected[index]) > 1e-6F) {
                fail(name, "incorrect sample at index " + std::to_string(index));
                return;
            }
        }
    }

    /**
     * @brief Return the aggregate status for the test runner.
     * @return 0 if every case passed, or 1 if any case failed.
     */
    int32_t result() const { return _num_failures == 0 ? 0 : 1; }

private:
    void fail(const std::string& name, const std::string& reason) {
        std::cerr << name << ": " << reason << '\n';
        ++_num_failures;
    }

    std::string _path;
    int32_t _num_failures = 0;
};

}  // namespace

int32_t main(int32_t argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: wav_tests <temporary-wav-path>\n";
        return 1;
    }

    WavTests tests(argv[1]);
    Bytes pcm16;
    Bytes pcm32;
    Bytes float32;

    // Independently encoded samples: zero, positive/negative half-scale, minimum, quarter-scale.
    for (const uint16_t sample : {0x0000, 0x4000, 0xc000, 0x8000, 0x2000}) {
        append_u16(pcm16, sample);
        append_u32(pcm32, static_cast<uint32_t>(sample) << 16);
    }
    for (const uint32_t sample : {0x00000000U, 0x3f000000U, 0xbf000000U, 0xbf800000U, 0x3e800000U}) {
        append_u32(float32, sample);
    }

    // Equivalent legacy and extensible headers must produce the same decoded samples.
    tests.check("PCM16", make_wav(1, 16, pcm16));
    tests.check("PCM32", make_wav(1, 32, pcm32));
    tests.check("float32", make_wav(3, 32, float32));
    tests.check("float32 with WAVEFORMATEX header", make_wav(3, 32, float32, {0, 0}));
    tests.check("extensible PCM16", make_wav(0xfffe, 16, pcm16, make_extension(1, 16)));
    tests.check("extensible PCM32", make_wav(0xfffe, 32, pcm32, make_extension(1, 32)));
    tests.check("extensible float32", make_wav(0xfffe, 32, float32, make_extension(3, 32)));
    tests.check("24 valid bits in PCM32", make_wav(0xfffe, 32, pcm32, make_extension(1, 24)));

    // Extra fmt bytes and RIFF padding must leave the reader aligned with the data chunk.
    Bytes extension = make_extension(1, 32);
    extension.insert(extension.end(), {0x12, 0x34});
    tests.check("fmt chunk with trailing bytes", make_wav(0xfffe, 32, pcm32, extension));
    extension[0] = 24;
    tests.check("extended fmt chunk", make_wav(0xfffe, 32, pcm32, extension));
    extension[0] = 25;
    extension.push_back(0x56);
    tests.check("odd-sized fmt chunk", make_wav(0xfffe, 32, pcm32, extension));

    // Reject incomplete extensions and size declarations that would extend beyond the fmt chunk.
    tests.check("missing extension", make_wav(0xfffe, 32, pcm32), false);
    extension = make_extension(1, 32);
    extension.pop_back();
    tests.check("short fmt chunk", make_wav(0xfffe, 32, pcm32, extension), false);
    extension    = make_extension(1, 32);
    extension[0] = 20;
    tests.check("short cbSize", make_wav(0xfffe, 32, pcm32, extension), false);
    extension[0] = 24;
    tests.check("cbSize exceeds fmt chunk", make_wav(0xfffe, 32, pcm32, extension), false);

    tests.check("unsupported subformat", make_wav(0xfffe, 32, pcm32, make_extension(6, 32)), false);
    // Keep the known format prefixes while changing other GUID bytes to exercise full-GUID matching.
    extension     = make_extension(1, 32);
    extension[10] = 1;
    tests.check("GUID with matching first two bytes", make_wav(0xfffe, 32, pcm32, extension), false);
    extension = make_extension(3, 32);
    extension.back() ^= 1;
    tests.check("GUID with matching format tag", make_wav(0xfffe, 32, float32, extension), false);
    tests.check("too many valid bits", make_wav(0xfffe, 32, pcm32, make_extension(1, 33)), false);

    // Packed 24-bit PCM remains unsupported, unlike 24 valid bits within a 32-bit container.
    tests.check("unsupported PCM24", make_wav(1, 24, Bytes(15)), false);
    tests.check("unsupported extensible PCM24", make_wav(0xfffe, 24, Bytes(15), make_extension(1, 24)), false);

    // Keep declared chunk sizes intact while truncating the actual file to exercise short reads.
    Bytes truncated = make_wav(0xfffe, 32, pcm32, make_extension(1, 32));
    truncated.resize(55);
    tests.check("truncated extension", truncated, false);
    truncated = make_wav(0xfffe, 32, pcm32, make_extension(1, 32));
    truncated.pop_back();
    tests.check("truncated samples", truncated, false);

    return tests.result();
}
