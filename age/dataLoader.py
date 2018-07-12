import pickle
import numpy
import torch
import random
import math

class AGE2(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.device = args.device
        
        data_file = open(args.data_path, 'rb')
        self.train_set, self.dev_set, self.test_set = pickle.load(data_file, encoding='latin1')
        self.weight = pickle.load(data_file, encoding='latin1').astype('float32')
        self.word_to_id = pickle.load(data_file, encoding='latin1')     # key: word, value: number
        self.id_to_word = pickle.load(data_file, encoding='latin1')     # key: number, value: word
        data_file.close()

        self.train_size = len(self.train_set)
        self.dev_size = len(self.dev_set)
        self.test_size = len(self.test_set)
        self.train_ptr = 0
        self.dev_ptr = 0
        self.test_ptr = 0

        ####### required items
        self.num_train_batches = math.ceil(self.train_size / self.batch_size)
        self.num_valid = self.dev_size
        self.num_test = self.test_size
        self.weight = torch.FloatTensor(self.weight)
        args.num_classes = 5
        args.num_words = len(self.word_to_id)
        args.vocab = self
        #######

    def wrap_numpy_to_longtensor(self, *args):
        res = []
        for arg in args:
            arg = torch.LongTensor(arg).to(self.device)
            res.append(arg)
        return res

    def wrap_to_model_arg(self, words, length): # should match the kwargs of model.forward
        return {
                'words': words,
                'length': length
                }


    def train_minibatch_generator(self):
        random.shuffle(self.train_set)
        while self.train_ptr <= self.train_size - self.batch_size:
            self.train_ptr += self.batch_size
            minibatch = self.train_set[self.train_ptr - self.batch_size : self.train_ptr]
            longest_hypo = numpy.max(list(map(lambda x: len(x[0]), minibatch)), axis=0)
            hypos = numpy.zeros((self.batch_size, longest_hypo), dtype='int32')
            truth = numpy.zeros((self.batch_size,), dtype='int32')
            length = numpy.zeros((self.batch_size,), dtype='int32')
            for i, (h, t) in enumerate(minibatch):
                hypos[i, :len(h)] = h
                length[i] = len(h)
                truth[i] = t
            words, length, label = self.wrap_numpy_to_longtensor(hypos, length, truth)
            model_arg = self.wrap_to_model_arg(words, length) 
            yield model_arg, label



    # NOTE: for dev and test, all data should be fetched regardless of batch_size!
    def dev_minibatch_generator(self, ):
        while self.dev_ptr < self.dev_size:
            batch_size = min(self.batch_size, self.dev_size - self.dev_ptr)
            self.dev_ptr += batch_size
            minibatch = self.dev_set[self.dev_ptr - batch_size : self.dev_ptr]
            longest_hypo = numpy.max(list(map(lambda x: len(x[0]), minibatch)), axis=0)
            hypos = numpy.zeros((batch_size, longest_hypo), dtype='int32')
            truth = numpy.zeros((batch_size,), dtype='int32')
            length = numpy.zeros((batch_size,), dtype='int32')
            for i, (h, t) in enumerate(minibatch):
                hypos[i, :len(h)] = h
                length[i] = len(h)
                truth[i] = t
            words, length, label = self.wrap_numpy_to_longtensor(hypos, length, truth)
            model_arg = self.wrap_to_model_arg(words, length) 
            yield model_arg, label

    def test_minibatch_generator(self, ):
        while self.test_ptr < self.test_size:
            batch_size = min(self.batch_size, self.test_size - self.test_ptr)
            self.test_ptr += batch_size
            minibatch = self.test_set[self.test_ptr - batch_size : self.test_ptr]
            longest_hypo = numpy.max(list(map(lambda x: len(x[0]), minibatch)), axis=0)
            hypos = numpy.zeros((batch_size, longest_hypo), dtype='int32')
            truth = numpy.zeros((batch_size,), dtype='int32')
            length = numpy.zeros((batch_size,), dtype='int32')
            for i, (h, t) in enumerate(minibatch):
                hypos[i, :len(h)] = h
                length[i] = len(h)
                truth[i] = t
            words, length, label = self.wrap_numpy_to_longtensor(hypos, length, truth)
            model_arg = self.wrap_to_model_arg(words, length) 
            yield model_arg, label

