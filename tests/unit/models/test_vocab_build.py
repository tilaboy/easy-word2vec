from unittest import TestCase
import os
import tempfile
import pickle
import shutil
from easy_word2vec.models.vocab_builder import VocabBuilder

class TestCaseVocabBuilder(TestCase):

    @classmethod
    def setUpClass(self):
        self.corpus = [
            ['foo', 'bar', 'foo', 'abc', 'def'],
            ['new', 'old', 'abc', 'how', 'yes', 'no'],
            ['no', 'foo', 'sir', 'foo'],
            ['abc', 'foo', 'old', 'sir', 'old', 'new', 'bar', 'foo', 'zoo'],
            ['old', 'foo', 'how', 'foo'],
            ['foo', 'bar', 'zoo', 'foo'],
            ['bye', 'bye', 'foo'],
            ['foo', 'old', 'foo']
        ]
        self.vb = VocabBuilder(min_freq=2, subsample_rate=None)
        self.test_dir = tempfile.mkdtemp()
        self.full_token_path = os.path.join(self.test_dir, 'full_token_freq.txt')
        self.kept_token_path = os.path.join(self.test_dir, 'kept_token_freq.txt')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.test_dir)

    def test_01_prepare_token_freq_table(self):
        token_freq_table = self.vb._prepare_token_freq_table(self.corpus)
        self.assertEqual(len(token_freq_table), 12)
        with open(self.full_token_path, 'wb') as p_fh:
            pickle.dump(token_freq_table, p_fh)


    def test_02_filter_on_freq(self):
        with open(self.full_token_path, 'rb') as p_fh:
            token_freq_table = pickle.load(p_fh)
        vocab_size, vocab_table = self.vb._filter_on_freq(token_freq_table, 1)
        self.assertEqual(vocab_size, 12)
        vocab_size, vocab_table = self.vb._filter_on_freq(token_freq_table, 2)
        self.assertEqual(vocab_size, 10)
        with open(self.kept_token_path, 'wb') as p_fh:
            pickle.dump(vocab_table, p_fh)
        vocab_size, vocab_table = self.vb._filter_on_freq(token_freq_table, 3)
        self.assertEqual(vocab_size, 4)

    def test_03_descending_freq_sort(self):
        with open(self.kept_token_path, 'rb') as p_fh:
            vocab_table = pickle.load(p_fh)

        token_arr = self.vb._sort_by_descending_freq(vocab_table)
        self.assertEqual(
            token_arr,
            ['foo', 'old', 'bar', 'abc', 'new', 'how', 'no', 'sir', 'zoo', 'bye']
        )

    def test_04_compute_subsampling_prob(self):
        with open(self.kept_token_path, 'rb') as p_fh:
            orig_vocab_table = pickle.load(p_fh)

        vocab_table = dict(orig_vocab_table)
        self.vb.token_arr = self.vb._sort_by_descending_freq(vocab_table)
        self.vb.freq_arr = [vocab_table[token] for token in self.vb.token_arr]
        self.vb._compute_subsampling_prob()
        self.assertAlmostEqual(self.vb.prob_arr[0], 1.0, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[1], 1.0, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[2], 1.0, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[4], 1.0, places=2)

        vocab_table = dict(orig_vocab_table)
        self.vb._compute_subsampling_prob(0.01)
        self.assertAlmostEqual(self.vb.prob_arr[0], 0.19, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[1], 0.34, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[2], 0.47, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[4], 0.60, places=2)

        vocab_table = dict(orig_vocab_table)
        self.vb._compute_subsampling_prob(0.001)
        self.assertAlmostEqual(self.vb.prob_arr[0], 0.06, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[1], 0.09, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[2], 0.12, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[4], 0.15, places=2)

        vocab_table = dict(orig_vocab_table)
        self.vb._compute_subsampling_prob(2)
        self.assertAlmostEqual(self.vb.prob_arr[0], 1.0, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[1], 1.0, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[2], 1.0, places=2)
        self.assertAlmostEqual(self.vb.prob_arr[4], 1.0, places=2)
