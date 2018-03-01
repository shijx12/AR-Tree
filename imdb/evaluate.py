import argparse

import numpy as np
import torch
from torchtext import data, datasets

from sst.model import SSTModel, SSTAttModel
from utils.helper import wrap_with_variable, unwrap_scalar_variable


def evaluate(args):
    text_field = data.Field(lower=args.lower, include_lengths=True,
                            batch_first=True)
    label_field = data.Field(sequential=False)

    dataset_splits = datasets.IMDB.splits(
        root=args.datadir, text_field=text_field, label_field=label_field)
    test_dataset = dataset_splits[2]

    text_field.build_vocab(*dataset_splits)
    label_field.build_vocab(*dataset_splits)
    text_field.vocab.id_to_word = lambda i: text_field.vocab.itos[i]
    text_field.vocab.id_to_tf = lambda i: text_field.freqs[i]

    print(f'Number of classes: {len(label_field.vocab)}')

    _, test_loader = data.BucketIterator.splits(
        datasets=dataset_splits, batch_size=args.batch_size, device=args.gpu)

    num_classes = len(label_field.vocab)
    if args.model_type == 'binary':
        model = SSTModel(num_classes=num_classes, num_words=len(text_field.vocab),
                    word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                    clf_hidden_dim=args.clf_hidden_dim,
                    clf_num_layers=args.clf_num_layers,
                    use_leaf_rnn=args.leaf_rnn,
                    bidirectional=args.bidirectional,
                    use_batchnorm=args.batchnorm,
                    dropout_prob=args.dropout,
                    weighted_by_interval_length=args.weighted,
                    weighted_base=args.weighted_base,
                    weighted_update=args.weighted_update,
                    cell_type=args.cell_type)
    else:
        model = SSTAttModel(vocab=text_field.vocab,
                      num_classes=num_classes, num_words=len(text_field.vocab),
                      word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                      clf_hidden_dim=args.clf_hidden_dim,
                      clf_num_layers=args.clf_num_layers,
                      use_leaf_rnn=args.leaf_rnn,
                      use_batchnorm=args.batchnorm,
                      dropout_prob=args.dropout,
                      bidirectional=args.bidirectional,
                      cell_type=args.cell_type,
                      att_type=args.att_type,
                      sample_num=args.sample_num)
    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.word_embedding.weight.size())
    print(f'# of parameters: {num_params}')
    print(f'# of word embedding parameters: {num_embedding_params}')
    print(f'# of parameters (excluding word embeddings): '
          f'{num_params - num_embedding_params}')
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    if args.gpu > -1:
        model.cuda(args.gpu)
    num_correct = 0
    num_data = len(test_dataset)
    for batch in test_loader:
        words, length = batch.text
        label = batch.label
        length = wrap_with_variable(length, volatile=True, gpu=args.gpu)
        logits, supplements = model(words=words, length=length, display=True)
        label_pred = logits.max(1)[1]
        num_correct_batch = torch.eq(label, label_pred).long().sum()
        num_correct_batch = unwrap_scalar_variable(num_correct_batch)
        num_correct += num_correct_batch
    print(f'# data: {num_data}')
    print(f'# correct: {num_correct}')
    print(f'Accuracy: {num_correct / num_data:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='/data/share/aclImdb/')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--cell_type', default='treelstm', choices=['treelstm', 'simple', 'P2K', 'Tri'])
    parser.add_argument('--model_type', default='binary', choices=['binary', 'att'])
    parser.add_argument('--att_type', default='corpus', choices=['corpus', 'rank0', 'rank1'], help='Used only when model_type==att')
    parser.add_argument('--sample_num', default=1, type=int)

    
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--hidden-dim', default=300, type=int)
    parser.add_argument('--clf-hidden-dim', default=1024, type=int)
    parser.add_argument('--clf-num-layers', default=2, type=int)
    parser.add_argument('--leaf-rnn', default=True, action='store_true')
    parser.add_argument('--batchnorm', default=True, action='store_true')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--bidirectional', default=False, action='store_true')

    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--weighted_base', type=float, default=2)
    parser.add_argument('--weighted_update', default=False, action='store_true')

    parser.add_argument('--fine-grained', default=False, action='store_true')
    parser.add_argument('--lower', default=False, action='store_true')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
