import argparse
import pickle
import os
import jsonlines
import numpy as np
from nltk import word_tokenize
from collections import Counter

from dataLoader import SNLIDataset

def collect_words(path, lower):
    # collect words from train set
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--vocab-size', type=int, default=50000)
    parser.add_argument('--max-length', type=int, default=200)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    print("Build vocab...")
    word_tf = Counter()
    for f in ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl']:
        _word_tf, _ = collect_words(os.path.join(args.data, f), args.lower)
        word_tf = word_tf + _word_tf
    word_wtoi = {'<unk>':0, '<pad>':1}
    word_tf = dict(Counter(word_tf).most_common(args.vocab_size))
    for w in word_tf:
        word_wtoi[w] = len(word_wtoi)
    word_tf['<unk>'] = 5000000
    word_tf['<pad>'] = 5000000 # just give a very large occurrence number
    label_wtoi = {'neutral': 0, 'entailment': 1, 'contradiction': 2}

    vocab = {
        'word_token_to_idx': word_wtoi,
        'word_token_to_freq': word_tf,
        'label_token_to_idx': label_wtoi,
            }

    print("Build dataset")
    train_reader = SNLIDataset(
        data_path=os.path.join(args.data, 'snli_1.0_train.jsonl'), vocab=vocab, max_length=args.max_length, lower=args.lower)
    valid_reader = SNLIDataset(
        data_path=os.path.join(args.data, 'snli_1.0_dev.jsonl'), vocab=vocab, max_length=args.max_length, lower=args.lower)
    test_reader = SNLIDataset(
        data_path=os.path.join(args.data, 'snli_1.0_test.jsonl'), vocab=vocab, max_length=args.max_length, lower=args.lower)

    with open(args.out, 'wb') as f:
        pickle.dump(train_reader, f)
        pickle.dump(valid_reader, f)
        pickle.dump(test_reader, f)


if __name__ == '__main__':
    main()
