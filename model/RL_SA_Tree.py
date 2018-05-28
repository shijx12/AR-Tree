import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.nn.functional import log_softmax, softmax
from utils.helper import unwrap_scalar_variable
from collections import defaultdict

from . import basic
from .basic import NaryLSTMLayer, TriPadLSTMLayer
import numpy as np
import random


class RlSaTree(nn.Module):

    def __init__(self, **kwargs):
        super(RlSaTree, self).__init__()
        self.vocab = kwargs['vocab']
        hidden_dim = self.hidden_dim = kwargs['hidden_dim'] 
        self.use_leaf_rnn = kwargs['use_leaf_rnn'] 
        self.bidirectional = kwargs['bidirectional'] 
        self.cell_type = kwargs['cell_type'] 
        self.sample_num = kwargs['sample_num'] 
        self.rich_state = kwargs['rich_state'] 
        self.rank_init = kwargs['rank_init'] 
        self.rank_input = kwargs['rank_input'] 
        self.rank_detach = kwargs['rank_detach'] 
        # only used in __init__
        word_dim = kwargs['word_dim']
        assert self.vocab.id_to_word


        ComposeCell = None
        if self.cell_type == 'Nary':
            ComposeCell = NaryLSTMLayer
        elif self.cell_type == 'TriPad':
            ComposeCell = TriPadLSTMLayer

        assert not (self.bidirectional and not self.use_leaf_rnn)

        if self.use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
            if self.bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(
                    input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_dim)
        real_hidden_dim = 2*hidden_dim if self.bidirectional else hidden_dim
        self.treelstm_layer = ComposeCell(real_hidden_dim)
        
        if self.rank_input == 'word':
            rank_dim = word_dim
        elif self.rank_input == 'h':
            rank_dim = real_hidden_dim
        if self.rich_state:
            rank_dim += 1 # word_embedding | pos

        self.rank = nn.Sequential(
                nn.Linear(in_features=rank_dim, out_features=128, bias=False),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=1, bias=False),
            )
        self.identity = nn.Dropout(0)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_leaf_rnn:
            init.kaiming_normal(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal(self.leaf_rnn_cell.weight_hh.data)
            init.constant(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            if self.bidirectional:
                init.kaiming_normal(self.leaf_rnn_cell_bw.weight_ih.data)
                init.orthogonal(self.leaf_rnn_cell_bw.weight_hh.data)
                init.constant(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
                init.constant(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            init.kaiming_normal(self.word_linear.weight.data)
            init.constant(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()
        for layer in self.rank:
            if type(layer)==nn.Linear:
                if self.rank_init == 'normal':
                    init.normal(layer.weight.data, mean=0, std=0.01)
                elif self.rank_init == 'kaiming':
                    init.kaiming_normal(layer.weight.data)
                else:
                    raise Exception('unsupported rank init')


    def calc_score(self, x):
        # x: word embeddings (batch_size, rank_dim)
        if self.rank_detach:
            x = self.identity(x).detach() # no gradient conveyed to word embedding
        if self.rich_state:
            l = x.size(0) # length
            pos_feat = [[abs(l-1-i-i) / l] for i in range(l)] # (length, 1)
            pos_feat = Variable(torch.from_numpy(np.asarray(pos_feat)).float().cuda())
            x_y = torch.cat((x, pos_feat), dim=1)
            s = self.rank(x_y)
        else:
            s = self.rank(x)
        return s


    def greedy_build(self, sentence, embedding, hs, cs, start, end, collector):
        """
        Args:
            hs: (length, 1, hidden_dim)
            cs: (length, 1, hidden_dim)
            start: int
            end: int
            collector: dict
        Output:
            h, c: (1, hidden_dim), embedding of sentence[start:end]
            all probabilities 
        """
        if end == start:
            return None, ''
        elif end == start+1:
            word = self.vocab.id_to_word(sentence[start])
            return (hs[start], cs[start]), f'({word})'
        
        scores = self.calc_score(embedding[start:end])
        pos = start + unwrap_scalar_variable(torch.max(scores, dim=0)[1])
        word = self.vocab.id_to_word(sentence[pos])
        collector[word].append((end - start) * log_softmax(scores, dim=0)[pos-start])

        left_state, left_word = self.greedy_build(sentence, embedding, hs, cs, start, pos, collector)
        right_state, right_word = self.greedy_build(sentence, embedding, hs, cs, pos+1, end, collector)
        output_state = self.treelstm_layer(left_state, right_state, (hs[pos], cs[pos]))
        word = self.vocab.id_to_word(sentence[pos])
        return output_state, f'({left_word}{word}{right_word})'


    def sample(self, sentence, embedding, hs, cs, start, end, collector):
        """
        To sample a tree structure for REINFORCE.
        Output:
            h, c
            all probabilities of selected word
        """
        if end == start:
            return None, ''
        elif end == start+1:
            word = self.vocab.id_to_word(sentence[start])
            return (hs[start], cs[start]), f'({word})'

        scores = self.calc_score(embedding[start:end])
        probs = softmax(scores, dim=0)
        cum = 0
        p = random.random()
        pos = end - 1
        for i in range(start, end):
            cum = cum + probs[i-start]
            if unwrap_scalar_variable(cum) >= p:
                pos = i
                break
        word = self.vocab.id_to_word(sentence[pos])
        collector[word].append((end - start) * torch.log(1e-9 + probs[pos-start]))  # collect log-probability of pos-th word

        left_state, left_word = self.sample(sentence, embedding, hs, cs, start, pos, collector)
        right_state, right_word = self.sample(sentence, embedding, hs, cs, pos+1, end, collector)
        output_state = self.treelstm_layer(left_state, right_state, (hs[pos], cs[pos]))
        word = self.vocab.id_to_word(sentence[pos])
        return output_state, f'({left_word}{word}{right_word})'


    def forward(self, sentence_embedding, sentence_word, length):
        """
        Args:
            sentence_embedding: (batch_size, max_length, word_dim). word embedding
            sentence_word: (batch_size, max_length). word id
            length: (batch_size, ). sentence length
        """
        batch_size, max_length, _ = sentence_embedding.size()
        if self.use_leaf_rnn:
            hs = []
            cs = []
            zero_state = Variable(sentence_embedding.data.new(batch_size, self.hidden_dim)
                                  .zero_())
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(
                    input=sentence_embedding[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = []
                cs_bw = []
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = list(length.data)
                input_bw = basic.reverse_padded_sequence(
                    inputs=sentence_embedding, lengths=lengths_list, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = basic.reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = basic.reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
        else:
            state = basic.apply_nd(fn=self.word_linear, input=sentence_embedding)
            hs, cs = state.chunk(num_chunks=2, dim=2)

        h_res, c_res, structure, samples = [], [], [], {}
        samples['h'], samples['probs'], samples['trees'] = [], [], []
        length = list(length.data)
        
        # iterate each sentence
        for i in range(batch_size):
            sentence = list(sentence_word[i].data)

            if self.rank_input == 'word':
                embedding = sentence_embedding[i]
            elif self.rank_input == 'h':
                embedding = hs[i]

            # calculate global scores for each word
            probs = defaultdict(list)
            state, tree = self.greedy_build(sentence, embedding, hs[i].unsqueeze(1), cs[i].unsqueeze(1), 0, length[i], probs)
            h, c = state
            h_res.append(h)
            c_res.append(c)
            structure.append(tree)
            
            ##################################
            # Monte Carlo
            for j in range(self.sample_num):
                if j > 0: # if j==0, just use the state+probs from greedy_build to avoid one extra sample
                    probs = defaultdict(list)
                    state, tree = self.sample(sentence, embedding, hs[i].unsqueeze(1), cs[i].unsqueeze(1), 0, length[i], probs)
                samples['h'].append(state[0])
                samples['probs'].append(probs) # a list of dict of Variable
                samples['trees'].append(tree)
        h_res, c_res = torch.stack(h_res, dim=0), torch.stack(c_res, dim=0)
        h_res, c_res = h_res.squeeze(1), c_res.squeeze(1)
        samples['h'] = torch.stack(samples['h'], dim=0).squeeze(1)

        return h_res, c_res, structure, samples

