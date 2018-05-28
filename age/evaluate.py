import argparse

import numpy as np
import torch
from age.dataset import AGE2
from sst.model import SSTModel
from utils.helper import wrap_with_variable, unwrap_scalar_variable
from collections import Counter, defaultdict
from torch.autograd import Variable
from IPython import embed


def evaluate(args):
    data = AGE2(datapath=args.data, batch_size=args.batch_size)
    num_classes = 5
    num_words = data.num_words
    model = SSTModel(
        typ='RL-SA',
        vocab=data,
        num_classes=num_classes, num_words=num_words,
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
    num_correct = 0
    num_data = data.test_size

    data._id_tf = {}
    for s in data.train_set+data.dev_set+data.test_set:
        for i in s[0]:
            data._id_tf[i] = data._id_tf.get(i, 0) + 1

    scores = {}
    for w, i in data._word_to_id.items():
        i = Variable(torch.Tensor((i,))).long()
        if args.gpu > -1:
            i = i.cuda()
        scores[w] = unwrap_scalar_variable(
                model.encoder.calc_score(model.word_embedding(i)).squeeze()
                )
    sort_scores = list(filter(lambda w: data._id_tf.get(data.word_to_id(w), 0) > 100 and len(w) >= 5,list(map(lambda _: _[0], Counter(scores).most_common()))))
    print()
    print('; '.join(sort_scores[:500]))

#    for batch in data.test_minibatch_generator():
#        words, length, label = batch
#
#        length = wrap_with_variable(length, volatile=True, gpu=args.gpu)
#        words = wrap_with_variable(words, volatile=True, gpu=args.gpu)
#        label = wrap_with_variable(label, volatile=True, gpu=args.gpu)
#        logits, supplements = model(words=words, length=length, display=True)
#        label_pred = logits.max(1)[1]
#        num_correct_batch = unwrap_scalar_variable(torch.eq(label, label_pred).long().sum())
#        num_correct += num_correct_batch
#        for t in supplements['trees']:
#            print(t)
#    print(f'# data: {num_data}')
#    print(f'# correct: {num_correct}')
#    print(f'Accuracy: {num_correct / num_data:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/age2.pickle')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--cell_type', default='TriPad', choices=['treelstm', 'Nary', 'TriPad'])
    parser.add_argument('--att_type', default='rank1', choices=['corpus', 'rank0', 'rank1', 'rank2'], help='Used only when model_type==att')
    parser.add_argument('--rich-state', default=False, action='store_true')
    parser.add_argument('--rank_input', default='word', choices=['word', 'h'])
    parser.add_argument('--rank_tanh', action='store_true')

    
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--hidden-dim', default=300, type=int)
    parser.add_argument('--clf-hidden-dim', default=2000, type=int)
    parser.add_argument('--clf-num-layers', default=2, type=int)
    parser.add_argument('--leaf-rnn', default=True, action='store_true')
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--bidirectional', default=True, action='store_true')


    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--weighted_base', type=float, default=2)
    parser.add_argument('--weighted_update', default=False, action='store_true')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
