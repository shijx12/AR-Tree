import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from collections import Counter, defaultdict
from snli.model import SNLIModel
from snli.utils.dataset import SNLIDataset
from utils.helper import wrap_with_variable, unwrap_scalar_variable, parse_tree_avg_depth, parse_tree
from IPython import embed


def evaluate(args):
    with open(args.data, 'rb') as f:
        test_dataset: SNLIDataset = pickle.load(f)
    word_vocab = test_dataset.word_vocab
    label_vocab = test_dataset.label_vocab

    if args.use_important_words and len(args.important_words) > 0:
        logging.info('Set _id_tf of important_words to 0')
        logging.info('words: %s' % ','.join(args.important_words))
        for word in args.important_words:
            word_vocab._id_tf[word_vocab.word_to_id(word)] = 0

    model = SNLIModel(
        typ='RL-SA',
        vocab=word_vocab,
        num_classes=len(label_vocab), num_words=len(word_vocab),
        word_dim=args.word_dim, hidden_dim=args.hidden_dim,
        clf_hidden_dim=args.clf_hidden_dim,
        clf_num_layers=args.clf_num_layers,
        use_leaf_rnn=args.leaf_rnn,
        use_batchnorm=args.batchnorm,
        dropout_prob=args.dropout,
        bidirectional=args.bidirectional,
        cell_type=args.cell_type,
        att_type=args.att_type,
        sample_num=1,
        rich_state=args.rich_state,
        rank_init='normal',
        rank_input=args.rank_input,
        rank_detach=False,
        rank_tanh=args.rank_tanh)
    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.word_embedding.weight.size())
    print(f'# of parameters: {num_params}')
    print(f'# of word embedding parameters: {num_embedding_params}')
    print(f'# of parameters (excluding word embeddings): '
          f'{num_params - num_embedding_params}')
    state_dict = torch.load(args.ckpt)
    del state_dict['encoder.comp_query']
    model.load_state_dict(state_dict)
    model.eval()
    if args.gpu > -1:
        model.cuda(args.gpu)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=test_dataset.collate)
    num_correct = depth_cumsum = word_cumsum = 0
    num_data = len(test_dataset)

    scores = {}
    for w, i in word_vocab._vocab_dict.items():
        i = Variable(torch.Tensor((i,))).long()
        if args.gpu > -1:
            i = i.cuda()
        scores[w] = unwrap_scalar_variable(
                model.encoder.calc_score(model.word_embedding(i)).squeeze()
                )
    sort_scores = list(filter(lambda w: word_vocab.id_to_tf(word_vocab.word_to_id(w)) > 100, list(map(lambda _: _[0], Counter(scores).most_common()))))
    print()
    print('; '.join(sort_scores[:300]))

#    for batch in test_data_loader:
#        pre = wrap_with_variable(batch['pre'], volatile=True, gpu=args.gpu)
#        hyp = wrap_with_variable(batch['hyp'], volatile=True, gpu=args.gpu)
#        pre_length = wrap_with_variable(batch['pre_length'], volatile=True,
#                                        gpu=args.gpu)
#        hyp_length = wrap_with_variable(batch['hyp_length'], volatile=True,
#                                        gpu=args.gpu)
#        label = wrap_with_variable(batch['label'], volatile=True, gpu=args.gpu)
#
#
#        logits, supplements = model(pre=pre, pre_length=pre_length, hyp=hyp, hyp_length=hyp_length, display=True)
#        label_pred = logits.max(1)[1]
#        num_correct += unwrap_scalar_variable(torch.eq(label, label_pred).long().sum())
#        for pt, ht in zip(supplements['pre_tree'], supplements['hyp_tree']):
#            print(pt)
#            print(ht)
#
#    depth = depth_cumsum / word_cumsum if word_cumsum != 0 else -1
#    print(f'model: {args.ckpt}')
#    print(f'# data: {num_data}')
#    print(f'# correct: {num_correct}')
#    print(f'Accuracy: {num_correct / num_data:.4f}. Average depth: {depth:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/snli_test.pickle')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--cell_type', default='TriPad', choices=['treelstm', 'Nary', 'TriPad'])
    parser.add_argument('--att_type', default='rank1', choices=['corpus', 'rank0', 'rank1', 'rank2'], help='Used only when model_type==att')
    parser.add_argument('--rich-state', default=False, action='store_true')
    parser.add_argument('--rank_input', default='word', choices=['word', 'h'])
    parser.add_argument('--rank_tanh', action='store_true')


    parser.add_argument('--use_important_words', default=False, action='store_true')
    parser.add_argument('--important_words', default=['no','No','NO','not','Not','NOT','isn\'t','aren\'t','hasn\'t','haven\'t','can\'t'])


    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--hidden-dim', default=300, type=int)
    parser.add_argument('--clf-hidden-dim', default=1024, type=int)
    parser.add_argument('--clf-num-layers', default=1, type=int)
    parser.add_argument('--leaf-rnn', default=True, action='store_true')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=True, action='store_true')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--fix-word-embedding', default=True, action='store_true')
    parser.add_argument('--batch-size', default=32, type=int)

    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--weighted_base', type=float, default=2)
    parser.add_argument('--weighted_update', default=False, action='store_true')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
