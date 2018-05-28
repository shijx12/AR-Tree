import argparse
import pickle
import torch
from torch.autograd import Variable
from snli.model import SNLIModel
from snli.utils.dataset import SNLIDataset
from sst.model import SSTModel
from age.dataset import AGE2
from torchtext import data, datasets
from nltk import word_tokenize
from IPython import embed


def snli_parser(args):
    with open(args.snli_data, 'rb') as f:
        test_dataset: SNLIDataset = pickle.load(f)
    word_vocab = test_dataset.word_vocab

    model = SNLIModel(
        typ='RL-SA',
        vocab=word_vocab,
        num_classes=3, num_words=len(word_vocab),
        word_dim=300, hidden_dim=300,
        clf_hidden_dim=1024,
        clf_num_layers=1,
        use_leaf_rnn=True,
        use_batchnorm=True,
        dropout_prob=0.1,
        bidirectional=False,
        cell_type='TriPad',
        att_type='rank1',
        sample_num=1,
        rich_state=False,
        rank_init='normal',
        rank_input='word',
        rank_detach=True,
        rank_tanh=False
    )

    state_dict = torch.load(args.snli_ckpt)
    del state_dict['encoder.comp_query']
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    def parse(sent):
        obj = {
                'sentence1': sent,
                'sentence2': 'oh',
                'gold_label': 'neutral'
                }
        pre, hyp, pre_length, hyp_length, label = list(map(
                lambda x: Variable(torch.LongTensor([x]).cuda(), volatile=True),
                test_dataset._convert_obj(obj)
            ))
        logits, supplements = model(pre=pre, pre_length=pre_length, hyp=hyp, hyp_length=hyp_length)
        return supplements['pre_tree'][0]
    return parse


def sst_parser(args):
    text_field = data.Field(lower=False, include_lengths=True, batch_first=True)
    label_field = data.Field(sequential=False)

    filter_pred = lambda ex: ex.label != 'neutral'
    dataset_splits = datasets.SST.splits(
        root=args.sst_data, text_field=text_field, label_field=label_field,
        fine_grained=False, train_subtrees=True, filter_pred=filter_pred)
    test_dataset = dataset_splits[2]
    text_field.build_vocab(*dataset_splits)
    label_field.build_vocab(*dataset_splits)
    text_field.vocab.id_to_word = lambda i: text_field.vocab.itos[i]
    text_field.vocab.id_to_tf = lambda i: text_field.freqs[i]
    model = SSTModel(
        typ='RL-SA',
        vocab=text_field.vocab,
        num_classes=3, num_words=len(text_field.vocab),
        word_dim=300, hidden_dim=300,
        clf_hidden_dim=300,
        clf_num_layers=1,
        use_leaf_rnn=True,
        use_batchnorm=False,
        dropout_prob=0.5,
        bidirectional=False,
        cell_type='TriPad',
        att_type='rank1',
        sample_num=1,
        rich_state=False,
        rank_init='normal',
        rank_input='word',
        rank_detach=False,
        rank_tanh=False
        )

    state_dict = torch.load(args.sst_ckpt)
    del state_dict['encoder.comp_query']
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    def parse(sent):
        sent = word_tokenize(sent)
        sent = [text_field.vocab.stoi[w] for w in sent]
        sent, l = list(map(
                lambda x: Variable(torch.LongTensor([x]).cuda(), volatile=True),
                [sent, len(sent)]
            ))
        logits, supplements = model(words=sent, length=l)
        return supplements['tree'][0]
    return parse


def age_parser(args):
    data = AGE2(datapath=args.age_data)
    model = SSTModel(
        typ='RL-SA',
        vocab=data,
        num_classes=5, num_words=data.num_words,
        word_dim=300, hidden_dim=300,
        clf_hidden_dim=2000,
        clf_num_layers=2,
        use_leaf_rnn=True,
        use_batchnorm=False,
        dropout_prob=0.3,
        bidirectional=True,
        cell_type='TriPad',
        att_type='rank1',
        sample_num=1,
        rich_state=False,
        rank_init='normal',
        rank_input='word',
        rank_detach=False,
        rank_tanh=False
        )

    state_dict = torch.load(args.age_ckpt)
    del state_dict['encoder.comp_query']
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    def parse(sent):
        sent = word_tokenize(sent)
        sent = [data.word_to_id(w) for w in sent]
        sent, l = list(map(
                lambda x: Variable(torch.LongTensor([x]).cuda(), volatile=True),
                [sent, len(sent)]
            ))
        logits, supplements = model(words=sent, length=l)
        return supplements['tree'][0]
    
    return parse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snli-data', default='./data/snli_test.pickle')
    parser.add_argument('--snli-ckpt', default='/data/sjx/SA-Tree-Exp/snli/rlsa/model-8.20-0.8613-0.8548.pkl')
    parser.add_argument('--sst-data', default='/data/share/stanfordSentimentTreebank/')
    parser.add_argument('--sst-ckpt', default='/data/sjx/SA-Tree-Exp/sst/rlsa/model-8.78-0.8899-0.9039.pkl')
    parser.add_argument('--age-data', default='./data/age2.pickle')
    parser.add_argument('--age-ckpt', default='/data/sjx/SA-Tree-Exp/age2/rlsa/model-2.00-0.8025-0.8085.pkl')
    args = parser.parse_args()
    
    model_name = ['snli', 'sst', 'age']
    parser_class = [snli_parser, sst_parser, age_parser]
    parsers = []
    for n, p in zip(model_name, parser_class):
        print('Construct parser for %s...' % n)
        parsers.append(p(args))
    
    def parse(sent):
        for n, p in zip(model_name, parsers):
            print('%s:' % n)
            print(p(sent))
    
    embed()


if __name__ == '__main__':
    main()
