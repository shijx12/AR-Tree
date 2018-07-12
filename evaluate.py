import argparse
import numpy as np
import torch
from model.SingleModel import SingleModel
from model.PairModel import PairModel
from age.dataLoader import AGE2
from sst.dataLoader import SST
from snli.dataLoader import SNLI

def eval_iter(batch, model):
    model.train(False)
    model_arg, label = batch
    logits, supplements = model(**model_arg)
    label_pred = logits.max(1)[1]
    num_correct = torch.eq(label, label_pred).long().sum().item()
    return num_correct, supplements 

def evaluate(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    args.device = device

    ################################  data  ###################################
    if args.data_type == 'sst2':
        args.fine_grained = False
        data = SST(args) # some extra info will be appended into args
    elif args.data_type == 'sst5':
        args.fine_grained = True
        data = SST(args)
    elif args.data_type == 'age':
        data = AGE2(args)
    elif args.data_type == 'snli':
        data = SNLI(args) 

    num_train_batches = data.num_train_batches # number of batches per epoch
    ################################  model  ###################################
    if args.data_type == 'snli':
        Model = PairModel
    else:
        Model = SingleModel
    model = Model(**vars(args))

    num_params = sum(np.prod(p.size()) for p in model.parameters())
    num_embedding_params = np.prod(model.word_embedding.weight.size())
    print(f'# of parameters: {num_params}')
    print(f'# of word embedding parameters: {num_embedding_params}')
    print(f'# of parameters (excluding word embeddings): '
          f'{num_params - num_embedding_params}')

    # load ckpt
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    correct_num = 0
    for test_batch in data.test_minibatch_generator():
        correct, supplements = eval_iter(test_batch, model)
        correct_sum += correct
        for t in supplements['tree']:
            print(t)
    print(f'Accuracy: {correct_num / data.num_test:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data-type', required=True, choices=['sst2', 'sst5', 'age'])
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--model-type', required=True, choices=['Choi', 'RL-SA', 'tfidf', 'STG-SA', 'SSA'])

    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--cell-type', default='TriPad', choices=['Nary', 'TriPad'])
    parser.add_argument('--leaf-rnn-type', default='bilstm', choices=['no', 'bilstm'])
    parser.add_argument('--rank-input', default='h', choices=['w', 'h'], help='whether feed word embedding or hidden state of bilstm into score function')
    parser.add_argument('--word-dim', default=300, type=int)

    parser.add_argument('--hidden-dim', type=int, help='dimension of final sentence embedding')
    parser.add_argument('--clf-hidden-dim', type=int)
    parser.add_argument('--clf-num-layers', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--use-batchnorm', action='store_true')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
