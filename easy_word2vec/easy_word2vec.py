'''easy_word2vec training, adapt the same parameter as word2vec'''
from argparse import ArgumentParser
from . import LOGGER

from .models.vocab_builder import VocabBuilder
from .models.word2vec_trainer import Word2VecTrainer
from .utils.negative_sampling import build_cumulative_table
from .utils.data_utils import prepare_corpus, save_word2vec_format


def get_args():
    '''get arguments'''
    parser = ArgumentParser(description='''easy_word2vec training,
adapt the same parameter name as original google word2vec c code''')
    parser.add_argument(
        "-train",
        help="Use text data from file TRAIN to train the model",
        required=True
    )
    parser.add_argument(
        "-output",
        help="Use file OUTPUT to save the resulting word vectors",
        default='word2vec.output'
    )
    parser.add_argument(
        "-window",
        help="Set max skip length WINDOW between words; default is 5",
        type=int,
        default=5
    )
    parser.add_argument(
        "-size",
        help="Set size of word vectors; default is 100",
        type=int,
        default=150
    )
    parser.add_argument(
        "-sample",
        help="Set threshold for occurrence of words. "
             "Those that appear with higher frequency "
             "in the training data will be randomly down-sampled;"
             " default is 1e-3, useful range is (0, 1e-5)",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        "-hs",
        help="Use Hierarchical Softmax; default is 0 (not used)",
        type=int,
        default=0,
        choices=[0, 1]
    )
    parser.add_argument(
        "-negative",
        help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)",
        type=int,
        default=5
    )
    parser.add_argument(
        "-threads",
        help="Use THREADS threads (default 4)",
        type=int,
        default=4
    )
    parser.add_argument(
        "-iter",
        help="Run more training iterations (default 5)",
        type=int,
        default=5
    )
    parser.add_argument(
        "-min_count",
        help="This will discard words that appear less than min_count times;"
             " default is 5",
        type=int,
        default=5
    )
    parser.add_argument(
        "-cbow",
        help="Use the continuous bag of words model;"
             " default is 1 (use 0 for skip-gram model)",
        type=int,
        default=1,
        choices=[0, 1]
    )
    parser.add_argument(
        "-alpha",
        help="Set the starting learning rate; "
             "default is 0.025 for skip-gram and 0.05 for CBOW",
        type=float,
        default=0.025
    )
    parser.add_argument(
        "-binary",
        help="Save the resulting vectors in binary mode; default is 0 (off)",
        type=int,
        default=0,
        choices=[0, 1]
    )
    parser.add_argument(
        "-accuracy",
        help="Use questions from file ACCURACY to evaluate the model"
    )

    return parser.parse_args()


# Example: ./word2vec.py -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 \
# -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        print("""Examples:
./easy-word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n
""")
        sys.exit(1)

    np.seterr(all='raise')  # don't ignore numpy errors

    skipgram = 1 if args.cbow == 0 else 0

    tokenized_corpus = prepare_corpus(args.train)
    vocab_obj = VocabBuilder(tokenized_corpus)
    vocab_obj.build_vocab(args.min_count, args.sample)
    if args.negative:
        build_cumulative_table(vocab_obj)

    trainer = Word2VecTrainer(
        corpus, embedding_size=args.size, min_freq=args.min_freq, nr_threads=args.threads,
        window=args.window, sample=args.sample, sg=skipgram, hs=args.hs,
        negative=args.negative, cbow_mean=1, epochs=args.iter,
    )
    trainer.prepare_weights()

    trainer.train(
        corpus_iterable=corpus_iterable, corpus_file=corpus_file, total_examples=self.corpus_count,
        nr_tokens=self.corpus_nr_tokens, epochs=self.epochs, start_learning_rate=self.learning_rate,
        end_learning_rate=self.min_learning_rate, compute_loss=self.compute_loss, callbacks=callbacks)

    # save_word2vec_format(args.output, binary=args.binary)

    # if args.eval_file:
    #     evaluate_embedding(args.eval_file)

    logger.info("finished running %s", program)
