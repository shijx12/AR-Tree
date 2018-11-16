import argparse
import numpy as np
import torch
from model.SingleModel import SingleModel
from model.PairModel import PairModel
from age.dataLoader import AGE2
from sst.dataLoader import SST
from snli.dataLoader import SNLI
from ete3 import Tree

def eval_iter(batch, model):
    model.eval()
    model_arg, label = batch
    logits, supplements = model(**model_arg)
    label_pred = logits.max(1)[1]
    num_correct = torch.eq(label, label_pred).long().sum().item()
    return num_correct, supplements 



def legal(s):
    return s.replace(',', '<comma>')

def postOrder(root):
    def recursion(node):
        if node is None:
            return '-'
        left = recursion(node.left)
        right = recursion(node.right)
        if node.left is None and node.right is None:
            return legal(node.word) # leaf node
        else:
            return '(%s,%s)%s' % (left, right, legal(node.word))
    return recursion(root)+';'

def visualizeTree(postOrderStr):
    t = Tree(postOrderStr, format=8)
    t_ascii = t.get_ascii(show_internal=True)
    print(t_ascii)


def recoverSentence(ids, length, vocab):
    ids = ids[0].tolist()
    length = length[0].item() # convert tensor to int
    sentence = list(map(lambda i: vocab.id_to_word[i], ids))
    sentence = ' '.join(sentence[:length])
    return sentence
    
    


def main(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    args.device = device
    args.batch_size = 128 if args.mode == 'val' else 1 # batch_size=1 for visualize
    # load model parameters from checkpoint
    loaded = torch.load(args.ckpt, map_location={'cuda:0':'cpu'})
    model_kwargs = loaded['model_kwargs']
    for k, v in model_kwargs.items():
        setattr(args, k, v)

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
    model.load_state_dict(loaded['model'])
    model.eval()
    model = model.to(device)

    if args.mode == 'val': # validate
        print('validate on test set......')
        correct_num = 0
        for test_batch in data.test_minibatch_generator():
            correct, supplements = eval_iter(test_batch, model)
            correct_sum += correct
            for t in supplements['tree']:
                print(t)
        print(f'Accuracy: {correct_num / data.num_test:.4f}')
    elif args.mode == 'vis': # visualize
        print('visualize learned tree structures.......')
        cnt = 0
        for test_batch in data.test_minibatch_generator():
            cnt += 1
            model_arg, label = test_batch
            logits, supplements = model(**model_arg)
            if args.data_type =='snli':
                visualizeTree(postOrder(supplements['pre_tree'][0]))
                visualizeTree(postOrder(supplements['hyp_tree'][0]))
                print(recoverSentence(model_arg['pre'], model_arg['pre_length'], args.vocab))
                print(recoverSentence(model_arg['hyp'], model_arg['hyp_length'], args.vocab))
            else:
                visualizeTree(postOrder(supplements['tree'][0]))
                print(recoverSentence(model_arg['words'], model_arg['length'], args.vocab))
            print('='*50)
            if cnt > 10:
                break





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--mode', choices=['vis', 'val'], help='visualize or validate')
    parser.add_argument('--glove', default='glove.840B.300d', help='used only by torchtext')
    args = parser.parse_args()
    main(args)

