import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init, Parameter
from utils.helper import unwrap_scalar_variable

from . import basic
from basic import NaryLSTMLayer, TriPadLSTMLayer
import numpy as np

class tfidfTree(nn.Module):

    def __init__(self, **kwargs):
        super(tfidfTree, self).__init__()
        self.vocab = kwargs['vocab']
        hidden_dim = self.hidden_dim = kwargs['hidden_dim'] 
        self.use_leaf_rnn = kwargs['use_leaf_rnn'] 
        self.bidirectional = kwargs['bidirectional'] 
        self.cell_type = kwargs['cell_type'] 
        # only used in __init__
        word_dim = kwargs['word_dim']
        assert self.vocab.id_to_df and self.vocab.id_to_word


        ComposeCell = None
        if self.cell_type == 'Nary':
            ComposeCell = NaryLSTMLayer
        elif self.cell_type == 'TriPad':
            ComposeCell = TriPadLSTMLayer

        assert not (self.bidirectional and not self.use_leaf_rnn)

        if self.use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
            if bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(
                    input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_dim)
        real_hidden_dim = 2*hidden_dim if self.bidirectional else hidden_dim
        self.treelstm_layer = ComposeCell(real_hidden_dim)
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
                # Set forget bias to 1
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            init.kaiming_normal(self.word_linear.weight.data)
            init.constant(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()

    def calc_score(self, w):
        # w: word ids of a sentence
        s = []
        for w_i in w:
            df = self.vocab.id_to_df(w_i)
            s.append(Variable(torch.Tensor([1/(1+df)]))) # idf
        return torch.stack(s, dim=0)


    def greedy_build(self, sentence, scores, hs, cs, start, end):
        """
        Args:
            scores: (length,). each word's score
            hs: (length, 1, hidden_dim)
            cs: (length, 1, hidden_dim)
            start: int
            end: int
        Output:
            h, c: (1, hidden_dim), embedding of sentence[start:end]
            all probabilities 
        """
        if end == start:
            return None, ''
        elif end == start+1:
            word = self.vocab.id_to_word(sentence[start])
            return (hs[start], cs[start]), f'({word})'
        
        pos = start + unwrap_scalar_variable(torch.max(scores[start:end], dim=0)[1])  # argmax, type is Integer
        left_state, left_word = self.attend_compose(sentence, scores, hs, cs, start, pos)
        right_state, right_word = self.attend_compose(sentence, scores, hs, cs, pos+1, end)
        output_state = self.treelstm_layer(left_state, right_state, (hs[pos], cs[pos]))
        word = self.vocab.id_to_word(sentence[pos])
        return output_state, f'({left_word}{word}{right_word})'

    def forward(self, sentence_embedding, sentence_word, length):
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


        h_res, c_res, structure = [], [], []
        length = list(length.data)
        # iterate each sentence
        for i in range(batch_size):
            sentence = list(sentence_word[i].data)
            scores = self.calc_score(sentence)
            state, tree = self.greedy_build(sentence, scores, hs[i].unsqueeze(1), cs[i].unsqueeze(1), 0, length[i])
            h, c = state
            h_res.append(h)
            c_res.append(c)
            structure.append(tree)

        h_res, c_res = torch.stack(h_res, dim=0), torch.stack(c_res, dim=0)
        h_res, c_res = h_res.squeeze(1), c_res.squeeze(1)

        return h_res, c_res, structure

