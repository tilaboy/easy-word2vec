'''build vocab from tokenized corpors'''
import numpy as np
from collections import defaultdict

from .. import LOGGER


class VocabBuilder:
    """
    TODO:
        - save the vocab given the output vocab filename
        - prun the vacab if does not fit max_vocab_size
    """

    def __init__(self, min_freq, subsample_rate=None):
        self.min_freq = min_freq
        self.subsample_rate = subsample_rate
        self.token_to_id = dict()
        self.token_arr = list()
        self.freq_arr = list()
        self.prob_arr = list()
        self.vacab_size = 0
        self.min_reduce = 1

    def _prepare_token_freq_table(self, tokenized_corpus):
        LOGGER.info("Build token frequency table")
        nr_tokens, min_reduce = 0, 1
        token_freq_table = defaultdict(int)
        id_sentence = 0
        for sentence in tokenized_corpus:
            if id_sentence % 10000 == 0:
                LOGGER.info(
                    "PROGRESS: #%i tokenized_corpus, %i words, keeping %i word types",
                    id_sentence, nr_tokens, len(token_freq_table)
                )
            for token in sentence:
                token_freq_table[token] += 1
            nr_tokens += len(sentence)
            id_sentence += 1

            # TODO:
            # if self.max_vocab_size and len(token_freq_table) > self.max_vocab_size:
            #     reduce_vocab(token_table, min_reduce)
            #     min_reduce += 1

        nr_tokenized_corpus = id_sentence + 1 if tokenized_corpus else 0
        LOGGER.info("in total %s tokens from %s tokenized_corpus", nr_tokens, nr_tokenized_corpus)
        return token_freq_table

    def _filter_on_freq(self, token_freq_table, min_freq=None):
        """
        prepare the vacab table:

        - drop tokens with frequency < min_freq
        - sort it to descending order
        """
        if min_freq is None:
            min_freq = self.min_freq
        nr_kept_tokens, kept_vocab = 0, dict()

        for token, freq in token_freq_table.items():
            if freq >= min_freq:
                kept_vocab[token] = freq
                nr_kept_tokens += freq

        kept_vocab_size = len(kept_vocab)
        full_vocab_size = len(token_freq_table)
        kept_vocab_pct = kept_vocab_size * 100 / max(full_vocab_size, 1)
        LOGGER.info("kept %s uniq tokens from %s all uniq tokens, or %s%% kept",
                    kept_vocab_size, full_vocab_size, round(kept_vocab_pct, 2))

        nr_total_tokens = sum(token_freq_table.values())
        kept_token_pct = nr_kept_tokens * 100 / max(nr_total_tokens, 1)
        LOGGER.info("kept %s tokens from %s tokens, or %s%% kept",
                    nr_kept_tokens, nr_total_tokens, round(kept_token_pct, 2))

        return kept_vocab_size, kept_vocab

    def _compute_subsampling_prob(self, subsample_rate=None):
        nr_total_tokens = sum(self.freq_arr)
        nr_uniq_tokens = len(self.token_arr)
        if subsample_rate is None:
            subsample_rate = self.subsample_rate
        if not subsample_rate:
            # no words downsampled
            threshold_count = nr_total_tokens
        elif subsample_rate < 1.0:
            # set parameter as proportion of total
            # compitible to word2vec implementation
            threshold_count = subsample_rate * nr_total_tokens
        else:
            # sample >= 1: downsample all words with higher count than sample
            # compitible to gensim implementation
            threshold_count = int(subsample_rate * (3 + np.sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        self.prob_arr = list()
        for token_id in range(nr_uniq_tokens):
            token = self.token_arr[token_id]
            freq = self.freq_arr[token_id]
            token_probability = (np.sqrt(freq / threshold_count) + 1) * (threshold_count / freq)
            if token_probability < 1.0:
                downsample_unique += 1
                downsample_total += token_probability * freq
            else:
                token_probability = 1.0
                downsample_total += freq
            self.prob_arr.append(token_probability)



    def _sort_by_descending_freq(self, vocab_freq):
        if not vocab_freq:
            return vocab_freq
        return sorted(vocab_freq, key=lambda ele:vocab_freq[ele], reverse=True)


    def build_vocab(self, corpus):
        """
        Build vocabulary from a sequence of token list.

        Params:
        -------
            tokenized_corpus: a list of token list
            [[token_1, token_2, ...], [token_i, token_i+1], ...]

        """
        token_freq_table = self._prepare_token_freq_table(corpus)
        self.vocab_size, vocab_table = self._filter_on_freq(token_freq_table)
        self.token_arr = self._sort_by_descending_freq(vocab_table)
        self.freq_arr = [vocab_table[token] for token in self.token_arr]
        vocab_table = None
        self.token_to_id = {
            token: token_id
            for token_id, token in
            enumerate(token_arr)
        }
        # compute probabilities from frequencies
        self._compute_subsampling_prob()

        # no Hierarchical Softmax for now
        # if self.hs:
        #    # add info about each word's Huffman encoding
        #    self.create_binary_tree()
