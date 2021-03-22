from unittest import TestCase
import os
import tempfile
import pickle
import shutil
from easy_word2vec.utils.data_utils import prepare_corpus

class TestCasePrepareCorpus(TestCase):
    def test_prepare_corpus(self):
        input_data = '''foo bar  foo  abc    def
  new old abc how yes   no
 no foo sir foo
abc foo old sir old new bar foo zoo
old foo how foo
foo bar zoo foo
bye bye foo
foo old foo'''
        expected_corpus = [
            ['foo', 'bar', 'foo', 'abc', 'def'],
            ['new', 'old', 'abc', 'how', 'yes', 'no'],
            ['no', 'foo', 'sir', 'foo'],
            ['abc', 'foo', 'old', 'sir', 'old', 'new', 'bar', 'foo', 'zoo'],
            ['old', 'foo', 'how', 'foo'],
            ['foo', 'bar', 'zoo', 'foo'],
            ['bye', 'bye', 'foo'],
            ['foo', 'old', 'foo']
        ]

        test_dir = tempfile.mkdtemp()
        input_file = os.path.join(test_dir, 'test.txt')
        with open(input_file, 'w', encoding='utf-8') as input_fh:
            input_fh.write(input_data)
        loaded_corpus = list(prepare_corpus(input_file))
        self.assertEqual(loaded_corpus, expected_corpus)

        shutil.rmtree(test_dir)
