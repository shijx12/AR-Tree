import pickle
import numpy
import torch
import random

class AGE2(object):
    def __init__(self, datapath, batch_size=50):
        self.batch_size = batch_size
        self.datapath = datapath
        
        data_file = open(self.datapath, 'rb')
        pickle.load(data_file)
        pickle.load(data_file)
        self.train_set, self.dev_set, self.test_set = pickle.load(data_file)
        self.weight = pickle.load(data_file).astype('float32')
        self.weight = torch.FloatTensor(self.weight)

        self.word2embed = pickle.load(data_file)   # key: word, value: embedding
        _word_to_id = pickle.load(data_file)     # key: word, value: number
        _id_to_word = pickle.load(data_file)     # key: number, value: word
        self._word_to_id = _word_to_id

        self.word_to_id = lambda _: _word_to_id[_]
        self.id_to_word = lambda _: _id_to_word[_]
        self.id_to_tf = lambda _: 0
        self.num_words = len(_word_to_id)
        data_file.close()

        self.train_size = len(self.train_set)
        self.dev_size = len(self.dev_set)
        self.test_size = len(self.test_set)
        self.train_ptr = 0
        self.dev_ptr = 0
        self.test_ptr = 0

    def wrap_numpy_to_longtensor(self, *args):
        res = []
        for arg in args:
            arg = torch.LongTensor(arg)
            res.append(arg)
        return res


    def train_minibatch_generator(self):
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
            hypos, length, truth = self.wrap_numpy_to_longtensor(hypos, length, truth)
            
            yield hypos, length, truth
        else:
            self.train_ptr = 0
            random.shuffle(self.train_set)
            raise StopIteration


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
            hypos, length, truth = self.wrap_numpy_to_longtensor(hypos, length, truth)
            
            yield hypos, length, truth
        else:
            self.dev_ptr = 0
            raise StopIteration

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
            hypos, length, truth = self.wrap_numpy_to_longtensor(hypos, length, truth)
            
            yield hypos, length, truth
        else:
            self.test_ptr = 0
            raise StopIteration

