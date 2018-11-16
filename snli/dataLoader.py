from torch.utils.data import DataLoader
import jsonlines
import torch
from nltk import word_tokenize
from torch.utils.data import Dataset
import pickle
import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def invert_dict(d):
    return { v:k for k,v in d.items() }

class SNLIDataset(Dataset):

    def __init__(self, data_path, vocab, max_length, lower):
        vocab['word_idx_to_token'] = invert_dict(vocab['word_token_to_idx'])
        vocab['label_idx_to_token'] = invert_dict(vocab['label_token_to_idx'])
        self.vocab = vocab

        self.lower = lower
        self._max_length = max_length
        self._data = []
        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                converted = self._convert_obj(obj)
                if converted:
                    self._data.append(converted)

    def _convert_obj(self, obj):
        pre_sentence = obj['sentence1']
        hyp_sentence = obj['sentence2']
        if self.lower:
            pre_sentence = pre_sentence.lower()
            hyp_sentence = hyp_sentence.lower()
        pre_words = word_tokenize(pre_sentence)
        hyp_words = word_tokenize(hyp_sentence)
        pre = [self.vocab['word_token_to_idx'].get(w, 0) for w in pre_words]
        hyp = [self.vocab['word_token_to_idx'].get(w, 0) for w in hyp_words]
        pre_length = len(pre)
        hyp_length = len(hyp)
        label = obj['gold_label']
        if len(pre) > self._max_length or len(hyp) > self._max_length:
            return None
        if label == '-':
            return None
        label = self.vocab['label_token_to_idx'][label]
        return pre, hyp, pre_length, hyp_length, label

    def _pad_sentence(self, data):
        max_length = max(len(d) for d in data)
        padded = [d + [self.vocab['word_token_to_idx']['<pad>']] * (max_length - len(d))
                  for d in data]
        return padded

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def collate(self, batch):
        (pre_batch, hyp_batch,
         pre_length_batch, hyp_length_batch, label_batch) = list(zip(*batch))
        pre_batch = self._pad_sentence(pre_batch)
        hyp_batch = self._pad_sentence(hyp_batch)
        pre = torch.LongTensor(pre_batch)
        hyp = torch.LongTensor(hyp_batch)
        pre_length = torch.LongTensor(pre_length_batch)
        hyp_length = torch.LongTensor(hyp_length_batch)
        label = torch.LongTensor(label_batch)
        return {'pre': pre, 'hyp': hyp,
                'pre_length': pre_length, 'hyp_length': hyp_length,
                'label': label}


class SNLI(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.device = args.device

        tic = time.time()
        with open(args.data_path, 'rb') as f:
            train_dataset = pickle.load(f)
            valid_dataset = pickle.load(f)
            test_dataset = pickle.load(f)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0,
                                  collate_fn=train_dataset.collate,
                                  pin_memory=True)
        self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0,
                                  collate_fn=valid_dataset.collate,
                                  pin_memory=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0,
                                  collate_fn=test_dataset.collate,
                                  pin_memory=True)

        self.vocab = train_dataset.vocab
        self.word_to_id = self.vocab['word_token_to_idx']
        self.id_to_word = self.vocab['word_idx_to_token']
        #### load glove
        if hasattr(args, 'glove_path'):
            print("load glove from %s" % (args.glove_path))
            glove = pickle.load(open(args.glove_path, 'rb'))
            dim = len(glove['the'])
            num_words = len(self.vocab['word_token_to_idx'])
            self.weight = np.zeros((num_words, dim), dtype=np.float32)
            for i in range(num_words):
                w = self.id_to_word[i]
                if w in glove:
                    self.weight[i] = glove[w]
            self.weight[1] = 0 # <pad>
            self.weight = torch.FloatTensor(self.weight)
        else:
            self.weight = None
            print("no glove")

        ####### required items
        self.num_train_batches = len(self.train_loader)
        self.num_valid = len(valid_dataset)
        self.num_test = len(test_dataset)
        args.num_classes = 3
        args.num_words = len(self.vocab['word_token_to_idx'])
        args.vocab = self
        ########
        print('It takes %.2f sec to load datafile. train/dev/test: %d/%d/%d.' % (time.time() - tic, len(train_dataset), len(valid_dataset), len(test_dataset)))

    
    def generator(self, loader):
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(self.device)
            label = batch.pop('label')
            yield batch, label

    def train_minibatch_generator(self):
        return self.generator(self.train_loader) 

    def dev_minibatch_generator(self):
        return self.generator(self.valid_loader) 

    def test_minibatch_generator(self):
        return self.generator(self.test_loader) 
        
