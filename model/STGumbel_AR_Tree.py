import torch
from torch import nn
from torch.nn import init
from .basic import NaryLSTMLayer, TriPadLSTMLayer, reverse_padded_sequence, st_gumbel_softmax
import numpy as np
from IPython import embed


class STGumbel_AR_Tree(nn.Module):

    def __init__(self, **kwargs):
        super(STGumbel_AR_Tree, self).__init__()
        self.vocab = kwargs['vocab']
        self.leaf_rnn_type = kwargs['leaf_rnn_type'] 
        self.cell_type = kwargs['cell_type'] 
        self.rank_input = kwargs['rank_input'] 
        self.temperature = 1 # TODO anneal
        word_dim = kwargs['word_dim']
        hidden_dim = self.hidden_dim = kwargs['hidden_dim'] 
        assert self.vocab.id_to_word

        ComposeCell = None
        if self.cell_type == 'Nary':
            ComposeCell = NaryLSTMLayer
        elif self.cell_type == 'TriPad':
            ComposeCell = TriPadLSTMLayer


        if self.leaf_rnn_type == 'bilstm':
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim//2) # dim//2 per direction
            self.leaf_rnn_cell_bw = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim//2)
        self.treelstm_layer = ComposeCell(hidden_dim)
        
        if self.rank_input == 'w':
            rank_dim = word_dim
        elif self.rank_input == 'h':
            rank_dim = hidden_dim
        rank_dim += 1 # word_embedding | pos

        self.rank = nn.Sequential(
                nn.Linear(in_features=rank_dim, out_features=256, bias=False),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=1, bias=False),
            )
        self.reset_parameters()

    def reset_parameters(self):
        if self.leaf_rnn_type == 'bilstm':
            init.kaiming_normal_(self.leaf_rnn_cell.weight_ih.data)
            init.orthogonal_(self.leaf_rnn_cell.weight_hh.data)
            init.constant_(self.leaf_rnn_cell.bias_ih.data, val=0)
            init.constant_(self.leaf_rnn_cell.bias_hh.data, val=0)
            # Set forget bias to 1
            self.leaf_rnn_cell.bias_ih.data.chunk(4)[1].fill_(1)
            init.kaiming_normal_(self.leaf_rnn_cell_bw.weight_ih.data)
            init.orthogonal_(self.leaf_rnn_cell_bw.weight_hh.data)
            init.constant_(self.leaf_rnn_cell_bw.bias_ih.data, val=0)
            init.constant_(self.leaf_rnn_cell_bw.bias_hh.data, val=0)
            self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        for layer in self.rank:
            if type(layer)==nn.Linear:
                init.kaiming_normal_(layer.weight.data)


    def calc_score(self, x):
        # x: word embeddings (batch_size, rank_dim-1)
        l = x.size(0) # length
        pos_feat = [[2*i/l-1] for i in range(l)] # (length, 1)
        pos_feat = torch.Tensor(pos_feat).float().to(x.device)
        x_y = torch.cat((x, pos_feat), dim=1)
        s = self.rank(x_y)
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
        #pos = start + torch.max(logits, dim=0)[1].item()

        gate = st_gumbel_softmax(logits.t(), self.temperature) # (1, end-start)
        h = torch.matmul(gate, hs[start:end])
        c = torch.matmul(gate, cs[start:end]) # (1, hidden_dim)
        pos = start + torch.max(gate, dim=1)[1].item() # TODO: gate or logits ?

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
        if self.leaf_rnn_type == 'bilstm':
            max_length = sentence_embedding.size(1)
            hs = []
            cs = []
            zero_state = torch.Tensor(batch_size, self.hidden_dim//2).zero_().to(sentence_embedding.device)
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

            hs_bw = []
            cs_bw = []
            h_bw_prev = c_bw_prev = zero_state
            lengths_list = length.tolist() 
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
            cs = torch.cat([cs, cs_bw], dim=2) # (batch_size, max_len, dim_h)
        else:
            # for phrase-level
            hs, cs = sentence_embedding
            assert self.rank_input == 'h'

        h_res, c_res, structure = [], [], []
        # iterate each sentence
        for i in range(batch_size):
            if isinstance(sentence_word, list):
                sentence = sentence_word[i]
            elif isinstance(sentence_word, torch.Tensor):
                sentence = list(map(self.vocab.id_to_word, sentence_word[i].tolist()))
            assert type(sentence[0])==str

            if self.rank_input == 'w':
                embedding = sentence_embedding[i]
            elif self.rank_input == 'h':
                embedding = hs[i]

            # calculate global scores for each word
            state, tree = self.build(sentence, embedding, hs[i], cs[i], 0, length[i].item())
            h, c = state
            h_res.append(h)
            c_res.append(c)
            structure.append(tree)
            
        h_res, c_res = torch.stack(h_res), torch.stack(c_res)
        h_res, c_res = h_res.squeeze(1), c_res.squeeze(1)

        return h_res, c_res, structure

