//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <vector>

namespace zhps {

    /**
     * @brief Configure file paths for Chinese text processing resources.
     *
     * Sets the locations of all external dictionaries and models required by the
     * Chinese tokenizer, HMM segmenter, IDF weighting, stop-word filtering,
     * and pinyin conversion components. This function must be called before
     * any text processing is performed.
     *
     * @param jieba_dict_file_in Path to the main Jieba dictionary file.
     * @param hmm_model_file_in Path to the HMM file used for word segmentation.
     * @param user_dict_file_in Path to the user-defined dictionary file.
     * @param idf_file_in       Path to the IDF dictionary file used for keyword extraction or weighting.
     * @param stop_words_file_in Path to the stop-words list file.
     * @param pinyin_dict_path_in Path to the directory or file containing pinyin mapping data.
     */
    void set_paths(const std::string& jieba_dict_file_in,
               const std::string& hmm_model_file_in,
               const std::string& user_dict_file_in,
               const std::string& idf_file_in,
               const std::string& stop_words_file_in,
               const std::string& pinyin_dict_path_in);

    /**
     * @brief Convert a Chinese text prompt into a phoneme sequence.
     *
     * Tokenizes the input Chinese text, resolves pronunciation using pinyin
     * dictionaries, and converts the result into a phoneme-level representation.
     * Optionally applies *erhua* (儿化) rules during phoneme generation.
     *
     * Unknown or out-of-vocabulary tokens are replaced with a configurable
     * fallback symbol.
     *
     * @param prompt      Input Chinese text to be processed.
     * @param with_erhua  If true, apply erhua (儿化) pronunciation rules where applicable.
     * @param unk_symbol  Symbol used to represent unknown or unsupported tokens.
     *
     * @return A string containing the resulting phoneme sequence.
     */
    std::string process_zh_tokens_to_phonemes(const std::string& prompt, bool with_erhua, std::string& unk_symbol);

}  // namespace zhps
