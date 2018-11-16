import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import log_softmax, softmax
from collections import defaultdict

from . import basic
from .basic import TriPadLSTMLayer, Node
import numpy as np
import random


class RL_AR_Tree(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.vocab = kwargs['vocab']
        self.leaf_rnn_type = kwargs['leaf_rnn_type'] 
        self.sample_num = kwargs.get('sample_num', 3) 
        self.rank_input = kwargs['rank_input'] 
        word_dim = kwargs['word_dim']
        hidden_dim = self.hidden_dim = kwargs['hidden_dim'] 
        assert self.vocab.id_to_word



        if self.leaf_rnn_type == 'bilstm':
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim//2) # dim//2 per direction
            self.leaf_rnn_cell_bw = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim//2)
        elif self.leaf_rnn_type == 'lstm':
            self.leaf_rnn_cell = nn.LSTMCell(input_size=word_dim, hidden_size=hidden_dim)
        self.treelstm_layer = TriPadLSTMLayer(hidden_dim)
        
        if self.rank_input == 'w':
            rank_dim = word_dim
        elif self.rank_input == 'h':
            rank_dim = hidden_dim

        self.rank = nn.Sequential(
                nn.Linear(in_features=rank_dim, out_features=128, bias=False),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=1, bias=False),
            )
        self.reset_parameters()

    def reset_parameters(self):
        if self.leaf_rnn_type in {'bilstm', 'lstm'}:
            init.kaiming_normal_(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal_(self.leaf_rnn_cell.weight_hh.data)
            init.constant_(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant_(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            if self.leaf_rnn_type == 'bilstm':
                init.kaiming_normal_(self.leaf_rnn_cell_bw.weight_ih.data)
                init.orthogonal_(self.leaf_rnn_cell_bw.weight_hh.data)
                init.constant_(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
                init.constant_(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        self.treelstm_layer.reset_parameters()
        for layer in self.rank:
            if type(layer)==nn.Linear:
                init.kaiming_normal_(layer.weight.data)


    def calc_score(self, x):
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
            return None, None
        elif end == start+1:
            root = Node(sentence[start])
            return (hs[start], cs[start]), root 
        
        scores = self.calc_score(embedding[start:end])
        pos = start + torch.max(scores, dim=0)[1].item()
        word = sentence[pos]
        collector[word].append((end - start) * log_softmax(scores, dim=0)[pos-start])

        left_state, left_tree = self.greedy_build(sentence, embedding, hs, cs, start, pos, collector)
        right_state, right_tree = self.greedy_build(sentence, embedding, hs, cs, pos+1, end, collector)
        output_state = self.treelstm_layer(left_state, right_state, (hs[pos], cs[pos]))
        root = Node(word, left_tree, right_tree)
        return output_state, root


    def sample(self, sentence, embedding, hs, cs, start, end, collector):
        """
        To sample a tree structure for REINFORCE.
        """
        if end == start:
            return None, None
        elif end == start+1:
            root = Node(sentence[start])
            return (hs[start], cs[start]), root 

        scores = self.calc_score(embedding[start:end])
        probs = softmax(scores, dim=0)
        cum = 0
        p = random.random()
        pos = end - 1
        for i in range(start, end):
            cum = cum + probs[i-start].item()
            if cum >= p:
                pos = i
                break
        word = sentence[pos]
        collector[word].append((end - start) * torch.log(1e-9 + probs[pos-start]))  # collect log-probability of pos-th word

        left_state, left_tree = self.sample(sentence, embedding, hs, cs, start, pos, collector)
        right_state, right_tree = self.sample(sentence, embedding, hs, cs, pos+1, end, collector)
        output_state = self.treelstm_layer(left_state, right_state, (hs[pos], cs[pos]))
        root = Node(word, left_tree, right_tree)
        return output_state, root


    def forward(self, sentence_embedding, sentence_word, length):
        """
        Args:
            sentence_embedding: (batch_size, max_length, word_dim). word embedding
            sentence_word: (batch_size, max_length). word id
            length: (batch_size, ). sentence length
        """
        batch_size, max_length, _ = sentence_embedding.size()
        if self.leaf_rnn_type in {'bilstm', 'lstm'}:
            hs = []
            cs = []
            zero_dim = self.hidden_dim if self.leaf_rnn_type == 'lstm' else self.hidden_dim//2
            zero_state = torch.zeros(batch_size, zero_dim).to(sentence_embedding.device)
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

            if self.leaf_rnn_type == 'bilstm':
                hs_bw = []
                cs_bw = []
                h_bw_prev = c_bw_prev = zero_state
                lengths_list = length.tolist() 
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

        h_res, c_res, structure, samples = [], [], [], {}
        samples['h'], samples['probs'], samples['trees'] = [], [], []
        length = length.tolist() 
        
        # iterate each sentence
        for i in range(batch_size):
            sentence = list(map(lambda i: self.vocab.id_to_word[i], sentence_word[i].tolist()))

            if self.rank_input == 'w':
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
                if j > 0: # if j==0, just use the state+probs from greedy_build
                    probs = defaultdict(list)
                    state, tree = self.sample(sentence, embedding, hs[i].unsqueeze(1), cs[i].unsqueeze(1), 0, length[i], probs)
                samples['h'].append(state[0])
                samples['probs'].append(probs) # a list of dict of Variable
                samples['trees'].append(tree)
        h_res, c_res = torch.stack(h_res, dim=0), torch.stack(c_res, dim=0)
        h_res, c_res = h_res.squeeze(1), c_res.squeeze(1)
        samples['h'] = torch.stack(samples['h'], dim=0).squeeze(1)

        return h_res, c_res, structure, samples

