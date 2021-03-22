from easy_tokenizer.tokenizer import Tokenizer


def _iter_data_line(data_file):
    with open(data_file, 'r', encoding='utf-8') as data_fh:
        for line in data_fh:
            yield line.rstrip()


def prepare_corpus(train_data):
    tokenizer = Tokenizer()
    for line in _iter_data_line(train_data):
        yield tokenizer.tokenize(line)
