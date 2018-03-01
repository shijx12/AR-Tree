import argparse
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from snli.model import SNLIBinaryModel, SNLIAttModel
from snli.utils.dataset import SNLIDataset
from utils.helper import wrap_with_variable, unwrap_scalar_variable, parse_tree_avg_depth, parse_tree


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

    if args.model_type == 'binary':
        model = SNLIBinaryModel(num_classes=len(label_vocab), num_words=len(word_vocab),
                      word_dim=args.word_dim, hidden_dim=args.hidden_dim,
                      clf_hidden_dim=args.clf_hidden_dim,
                      clf_num_layers=args.clf_num_layers,
                      use_leaf_rnn=args.leaf_rnn,
                      intra_attention=args.intra_attention,
                      use_batchnorm=args.batchnorm,
                      dropout_prob=args.dropout,
                      bidirectional=args.bidirectional,
                      weighted_by_interval_length=args.weighted,
                      weighted_base=args.weighted_base,
                      weighted_update=args.weighted_update,
                      cell_type=args.cell_type)
    else:
        model = SNLIAttModel(vocab=word_vocab,
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
                      rich_state=args.rich_state)
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
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=test_dataset.collate)
    num_correct = depth_cumsum = word_cumsum = 0
    num_data = len(test_dataset)
    for batch in test_data_loader:
        pre = wrap_with_variable(batch['pre'], volatile=True, gpu=args.gpu)
        hyp = wrap_with_variable(batch['hyp'], volatile=True, gpu=args.gpu)
        pre_length = wrap_with_variable(batch['pre_length'], volatile=True,
                                        gpu=args.gpu)
        hyp_length = wrap_with_variable(batch['hyp_length'], volatile=True,
                                        gpu=args.gpu)
        label = wrap_with_variable(batch['label'], volatile=True, gpu=args.gpu)

        ####################################
        # model_type: binary
        if args.model_type == 'binary':
            logits, supplements = model(pre=pre, pre_length=pre_length, hyp=hyp, hyp_length=hyp_length)
            pre_select_masks, hyp_select_masks = supplements['pre_select_masks'], supplements['hyp_select_masks']
            label_pred = logits.max(1)[1]
            num_correct += unwrap_scalar_variable(torch.eq(label, label_pred).long().sum())
            depth_cumsum_1, word_cumsum_1 = parse_tree_avg_depth(pre_length, pre_select_masks)
            depth_cumsum_2, word_cumsum_2 = parse_tree_avg_depth(hyp_length, hyp_select_masks)
            depth_cumsum += depth_cumsum_1 + depth_cumsum_2
            word_cumsum += word_cumsum_1 + word_cumsum_2
            # sample parse tree to print
            '''trees = parse_tree(word_vocab, batch['pre'], pre_length, pre_select_masks)
            for i in range(1):
                print(trees[i])
            trees = parse_tree(word_vocab, batch['hyp'], hyp_length, hyp_select_masks)
            for i in range(1):
                print(trees[i])
            print(' ----------------------------------- ')'''
        ######################################
        # model_type: att
        else:
            ########################
            # att_type: corpus
            if args.att_type == 'corpus':
                logits, pre_tree, hyp_tree = model(pre=pre, pre_length=pre_length, hyp=hyp, hyp_length=hyp_length, display=True)
                label_pred = logits.max(1)[1]
                num_correct += unwrap_scalar_variable(torch.eq(label, label_pred).long().sum())
            #######################
            # att_type: rank
            else:
                logits, supplements = model(pre=pre, pre_length=pre_length, hyp=hyp, hyp_length=hyp_length, display=True)
                label_pred = logits.max(1)[1]
                num_correct += unwrap_scalar_variable(torch.eq(label, label_pred).long().sum())
        #######################################

    depth = depth_cumsum / word_cumsum if word_cumsum != 0 else -1
    print(f'model: {args.ckpt}')
    print(f'# data: {num_data}')
    print(f'# correct: {num_correct}')
    print(f'Accuracy: {num_correct / num_data:.4f}. Average depth: {depth:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/snli_test.pickle')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--cell_type', default='treelstm', choices=['treelstm', 'simple', 'P2K', 'Tri', 'TriPad'])
    parser.add_argument('--model_type', default='binary', choices=['binary', 'att'])
    parser.add_argument('--att_type', default='corpus', choices=['corpus', 'rank0', 'rank1', 'rank2'], help='Used only when model_type==att')


    parser.add_argument('--use_important_words', default=False, action='store_true')
    parser.add_argument('--important_words', default=['no','No','NO','not','Not','NOT','isn\'t','aren\'t','hasn\'t','haven\'t','can\'t'])


    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--hidden-dim', default=300, type=int)
    parser.add_argument('--clf-hidden-dim', default=1024, type=int)
    parser.add_argument('--clf-num-layers', default=2, type=int)
    parser.add_argument('--leaf-rnn', default=True, action='store_true')
    parser.add_argument('--bidirectional', default=False, action='store_true')
    parser.add_argument('--intra-attention', default=False, action='store_true')
    parser.add_argument('--batchnorm', default=True, action='store_true')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--fix-word-embedding', default=True, action='store_true')
    parser.add_argument('--rich-state', default=False, action='store_true')
    parser.add_argument('--batch-size', default=100, type=int)

    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--weighted_base', type=float, default=2)
    parser.add_argument('--weighted_update', default=False, action='store_true')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
