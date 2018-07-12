import argparse

import jsonlines
from nltk import word_tokenize
from collections import Counter


def collect_words(path, lower):
    word_tf = Counter()
    word_df = Counter()
    with jsonlines.open(path, 'r') as reader:
        for obj in reader:
            for key in ['sentence1', 'sentence2']:
                sentence = obj[key]
                if lower:
                    sentence = sentence.lower()
                words = word_tokenize(sentence)
                for word in words:
                    word_tf[word] = word_tf.get(word, 0) + 1
                for word in set(words):
                    word_df[word] = word_df.get(word, 0) + 1
    return word_tf, word_df


def save_vocab(word_tf, word_df, path):
    # word_tf and word_df are both Counter
    with open(path, 'w', encoding='utf-8') as f:
        for word, tf in word_tf.most_common():
            # <word> <tf> <df>\n
            f.write('%s %d %d\n' % (word, tf, word_df[word]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help="training data")
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    word_tf = Counter()
    word_df = Counter()
    _word_tf, _word_df = collect_words(path=args.data_path, lower=args.lower)
    word_tf = word_tf + _word_tf
    word_df = word_df + _word_df
    save_vocab(word_tf=word_tf, word_df=word_df, path=args.out)


if __name__ == '__main__':
    main()
