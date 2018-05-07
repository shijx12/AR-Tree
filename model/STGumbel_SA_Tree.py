import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.nn.functional import log_softmax, softmax
from utils.helper import unwrap_scalar_variable
from collections import defaultdict

from .basic import NaryLSTMLayer, TriPadLSTMLayer, reverse_padded_sequence, apply_nd, st_gumbel_softmax
import numpy as np
import random
from IPython import embed


class STGumbelSaTree(nn.Module):

    def __init__(self, **kwargs):
        super(STGumbelSaTree, self).__init__()
        self.vocab = kwargs['vocab']
        hidden_dim = self.hidden_dim = kwargs['hidden_dim'] 
        self.use_leaf_rnn = kwargs['use_leaf_rnn'] 
        self.bidirectional = kwargs['bidirectional'] 
        self.cell_type = kwargs['cell_type'] 
        self.rich_state = kwargs['rich_state'] 
        self.rank_init = kwargs['rank_init'] 
        self.rank_input = kwargs['rank_input'] 
        self.rank_detach = kwargs['rank_detach'] 
        self.temperature = kwargs.get('temperature', 1) # TODO anneal
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
        if self.rich_state: # TODO position embedding
            l = x.size(0) # length
            pos_feat = [[abs(l-1-i-i) / l] for i in range(l)] # (length, 1)
            pos_feat = Variable(torch.from_numpy(np.asarray(pos_feat)).float().cuda())
            x_y = torch.cat((x, pos_feat), dim=1)
            s = self.rank(x_y)
        else:
            s = self.rank(x)
        return s


    def build(self, sentence, embedding, hs, cs, start, end):
        """
        Args:
            hs: (length, hidden_dim)
            cs: (length, hidden_dim)
            start: int
            end: int
        Output:
            h, c: (1, hidden_dim), embedding of sentence[start:end]
        """
        if end == start:
            return None, ''
        elif end == start+1:
            word = sentence[start]
            return (hs[start].unsqueeze(0), cs[start].unsqueeze(0)), f'({word})' if word!='' else ''
        
        logits = self.calc_score(embedding[start:end])
        pos = start + unwrap_scalar_variable(torch.max(logits, dim=0)[1])

        gate = st_gumbel_softmax(logits.squeeze(), self.temperature) # (end-start,)
        h = torch.matmul(gate, hs[start:end]).unsqueeze(0)
        c = torch.matmul(gate, cs[start:end]).unsqueeze(0) # (1, hidden_dim)

        left_state, left_word = self.build(sentence, embedding, hs, cs, start, pos)
        right_state, right_word = self.build(sentence, embedding, hs, cs, pos+1, end)
        output_state = self.treelstm_layer(left_state, right_state, (h, c))
        word = sentence[pos]
        if word != '':
            word = f'({left_word}{word}{right_word})'
        return output_state, word 


    def forward(self, sentence_embedding, sentence_word, length):
        """
        Args:
            sentence_embedding: (batch_size, max_length, word_dim). word embedding
                            if not self.use_leaf_rnn, it consists of (hs, cs)
            sentence_word: (batch_size, max_length). word id
                            if it is a list, it contains strings directly
            length: (batch_size, ). sentence length
        """
        batch_size = length.size(0)
        if self.use_leaf_rnn:
            max_length = sentence_embedding.size(1)
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
                input_bw = reverse_padded_sequence(
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
                hs_bw = reverse_padded_sequence(
                    inputs=hs_bw, lengths=lengths_list, batch_first=True)
                cs_bw = reverse_padded_sequence(
                    inputs=cs_bw, lengths=lengths_list, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
        else:
            # for phrase-level
            hs, cs = sentence_embedding
            assert self.rank_input == 'h'

        h_res, c_res, structure = [], [], []
        length = list(length.data)
        
        # iterate each sentence
        for i in range(batch_size):
            if type(sentence_word)==list:
                sentence = sentence_word[i]
            elif type(sentence_word)==Variable:
                sentence = list(map(self.vocab.id_to_word, list(sentence_word[i].data)))
            assert type(sentence[0])==str

            if self.rank_input == 'word':
                embedding = sentence_embedding[i]
            elif self.rank_input == 'h':
                embedding = hs[i]

            # calculate global scores for each word
            state, tree = self.build(sentence, embedding, hs[i], cs[i], 0, length[i])
            h, c = state
            h_res.append(h)
            c_res.append(c)
            structure.append(tree)
            
        h_res, c_res = torch.stack(h_res, dim=0), torch.stack(c_res, dim=0)
        h_res, c_res = h_res.squeeze(1), c_res.squeeze(1)

        return h_res, c_res, structure

