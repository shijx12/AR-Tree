import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.nn.functional import log_softmax, softmax
import random
from utils.helper import unwrap_scalar_variable
from collections import defaultdict

from . import basic
import conf
import numpy as np
from IPython import embed


# N-ary Tree-LSTM in the paper of treelstm
class NaryLSTMLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(NaryLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=3 * hidden_dim,
                                    out_features=5 * hidden_dim)
        self.zero = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.comp_linear.weight.data)
        init.constant(self.comp_linear.bias.data, val=0)

    def forward(self, l=None, r=None, x=None):
        """
        Args:
            l: (h_l, c_l) tuple, where h and c have the size (batch_size, hidden_dim)
            r: (h_r, c_r) tuple
            x: (h_x, c_x) tuple. 
        Returns:
            h, c : The hidden and cell state of the composed parent
        """
        hx, cx = x
        if l==None:
            l = (self.zero, self.zero)
        if r==None:
            r = (self.zero, self.zero)
        hr, cr = r
        hl, cl = l
        h_cat = torch.cat([hx, hl, hr], dim=1)
        comp_vector = self.comp_linear(h_cat)
        i, fl, fr, u, o = torch.chunk(comp_vector, chunks=5, dim=1)
        c = (cl*(fl + 1).sigmoid() + cr*(fr + 1).sigmoid() + u.tanh()*i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c 


class TriPadLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(TriPadLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=3 * hidden_dim,
                                    out_features=6 * hidden_dim)
        self.zero = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.comp_linear.weight.data)
        init.constant(self.comp_linear.bias.data, val=0)

    def forward(self, l=None, r=None, m=None):
        """
        Args:
            l: (h_l, c_l) tuple, where h and c have the size (batch_size, hidden_dim)
            r: (h_r, c_r) tuple
            m: (h_m, c_m) tuple. 
        Returns:
            h, c : The hidden and cell state of the composed parent
        """
        hm, cm = m
        if l==None:
            l = (self.zero, self.zero)
        if r==None:
            r = (self.zero, self.zero)
        hr, cr = r
        hl, cl = l
        h_cat = torch.cat([hl, hm, hr], dim=1)
        comp_vector = self.comp_linear(h_cat)
        i, fl, fm, fr, u, o = torch.chunk(comp_vector, chunks=6, dim=1)
        c = (cl*(fl + 1).sigmoid() + cm*(fm + 1).sigmoid() + cr*(fr + 1).sigmoid()
            + u.tanh()*i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c 


class AttTreeLSTM(nn.Module):

    def __init__(self, vocab, word_dim, hidden_dim, use_leaf_rnn, 
                 bidirectional, cell_type, att_type, sample_num, 
                 rich_state, rank_init, rank_input, rank_detach, rank_tanh):
        super(AttTreeLSTM, self).__init__()
        self.vocab = vocab
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.use_leaf_rnn = use_leaf_rnn
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.att_type = att_type
        self.sample_num = sample_num
        self.rich_state = rich_state
        self.rank_init = rank_init
        self.rank_input = rank_input
        self.rank_detach = rank_detach


        ComposeCell = None
        if self.cell_type == 'Nary':
            ComposeCell = NaryLSTMLayer
        elif self.cell_type == 'TriPad':
            ComposeCell = TriPadLSTMLayer

        assert not (self.bidirectional and not self.use_leaf_rnn)

        if use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
            if bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(
                    input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_dim)
        if self.bidirectional:
            self.treelstm_layer = ComposeCell(2 * hidden_dim)
            self.comp_query = nn.Parameter(torch.FloatTensor(2 * hidden_dim))
            self.zero_holder = Variable(torch.zeros(2 * hidden_dim).cuda(), requires_grad=False)
        else:
            self.treelstm_layer = ComposeCell(hidden_dim)
            self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))
            self.zero_holder = Variable(torch.zeros(hidden_dim).cuda(), requires_grad=False)
        
        if rank_input == 'word':
            rank_dim = word_dim
        elif rank_input == 'h':
            if self.bidirectional:
                rank_dim = 2 * hidden_dim
            else:
                rank_dim = hidden_dim
        if self.rich_state:
            rank_dim += 1 # word_embedding | pos
        
        if self.att_type == 'rank0': # most simple
            self.rank = nn.Sequential(
                        nn.Linear(in_features=rank_dim, out_features=1, bias=False),
                    )
        elif self.att_type == 'rank1':
            if rank_tanh:
                self.rank = nn.Sequential(
                        nn.Linear(in_features=rank_dim, out_features=128, bias=False),
                        nn.ReLU(),
                        nn.Linear(in_features=128, out_features=1, bias=False),
                        nn.Tanh()
                    )
            else:
                self.rank = nn.Sequential(
                        nn.Linear(in_features=rank_dim, out_features=128, bias=False),
                        nn.ReLU(),
                        nn.Linear(in_features=128, out_features=1, bias=False),
                    )
        elif self.att_type == 'rank2':
            if rank_tanh:
                self.rank = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(in_features=rank_dim, out_features=256, bias=False),
                        nn.ReLU(),
                        nn.Linear(in_features=256, out_features=128, bias=False),
                        nn.ReLU(),
                        nn.Linear(in_features=128, out_features=64, bias=False),
                        nn.ReLU(),
                        nn.Linear(in_features=64, out_features=1, bias=False),
                        nn.Tanh()
                    )
            else:
                self.rank = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(in_features=rank_dim, out_features=256, bias=False),
                        nn.ReLU(),
                        nn.Linear(in_features=256, out_features=128, bias=False),
                        nn.ReLU(),
                        nn.Linear(in_features=128, out_features=64, bias=False),
                        nn.ReLU(),
                        nn.Linear(in_features=64, out_features=1, bias=False),
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
                # Set forget bias to 1
                self.leaf_rnn_cell_bw.bias_ih.data.chunk(4)[1].fill_(1)
        else:
            init.kaiming_normal(self.word_linear.weight.data)
            init.constant(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()
        init.normal(self.comp_query.data, mean=0, std=0.01)
        for layer in self.rank:
            if type(layer)==nn.Linear:
                if self.rank_init == 'normal':
                    init.normal(layer.weight.data, mean=0, std=0.01)
                elif self.rank_init == 'kaiming':
                    init.kaiming_normal(layer.weight.data)
                else:
                    raise Exception('unsupported rank init')


    def calc_score(self, w, x):
        # w: word ids of a sentence
        # x: word embeddings (batch_size, rank_dim)
        if self.att_type == 'corpus':
            s = []
            for w_i in w:
                tf = self.vocab.id_to_tf(w_i)
                s.append(Variable(torch.Tensor([1/(1+tf)])))
            return torch.stack(s, dim=0)
        else:
            if self.rich_state:
                if self.rank_detach:
                    x = self.identity(x).detach() # x must be a subsequence
                l = x.size(0) # length
                pos_feat = [[abs(l-1-i-i) / l] for i in range(l)] # (length, 1)
                pos_feat = Variable(torch.from_numpy(np.asarray(pos_feat)).float().cuda())
                x_y = torch.cat((x, pos_feat), dim=1)
                s = self.rank(x_y)
            else:
                if self.rank_detach:
                    x = self.identity(x).detach() # no gradient from score to word embedding
                s = self.rank(x)
        return s


    def attend_compose(self, sentence, embedding, scores, hs, cs, start, end, collector):
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
        
        if self.rich_state: # re-calculate scores
            score = self.calc_score(None, embedding[start:end])
            pos = start + unwrap_scalar_variable(torch.max(score, dim=0)[1])
            word = self.vocab.id_to_word(sentence[pos])
            collector[word].append((end - start) * log_softmax(score, dim=0)[pos-start])
        else:
            pos = start + unwrap_scalar_variable(torch.max(scores[start:end], dim=0)[1])  # argmax, type is Integer
            ############################
            # longer sequence, larger weight for action selection. TODO
            word = self.vocab.id_to_word(sentence[pos])
            collector[word].append((end - start) * log_softmax(scores[start:end], dim=0)[pos-start])

        left_state, left_word = self.attend_compose(sentence, embedding, scores, hs, cs, start, pos, collector)
        right_state, right_word = self.attend_compose(sentence, embedding, scores, hs, cs, pos+1, end, collector)
        output_state = self.treelstm_layer(left_state, right_state, (hs[pos], cs[pos]))
        word = self.vocab.id_to_word(sentence[pos])
        return output_state, f'({left_word}{word}{right_word})'


    def sample(self, sentence, embedding, scores, hs, cs, start, end, collector):
        """
        Only used if self.att_type='rank'. To sample a tree structure for REINFORCE.
        Args:
            scores: (length, ).  scores_i = exp(self.w_rank * x_i)
            start:end is the interval.
        Output:
            h, c
            all probabilities of selected word
        """
        if end == start:
            return None, ''
        elif end == start+1:
            word = self.vocab.id_to_word(sentence[start])
            return (hs[start], cs[start]), f'({word})'

        if self.rich_state: # re-calculate scores on subsequence
            score = self.calc_score(None, embedding[start:end])
            probs = softmax(score, dim=0)
        else:
            probs = softmax(scores[start:end], dim=0)
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

        left_state, left_word = self.sample(sentence, embedding, scores, hs, cs, start, pos, collector)
        right_state, right_word = self.sample(sentence, embedding, scores, hs, cs, pos+1, end, collector)
        output_state = self.treelstm_layer(left_state, right_state, (hs[pos], cs[pos]))
        word = self.vocab.id_to_word(sentence[pos])
        return output_state, f'({left_word}{word}{right_word})'


    def display_structure(self, sentence, scores, start, end):
        """
        Return a string representing tree structure
        """
        if end == start:
            return ''
        elif end == start+1:
            word = self.vocab.id_to_word(sentence[start])
            return f'({word})'
        pos = start + unwrap_scalar_variable(torch.max(scores[start:end], dim=0)[1]) # argmax
        left_state = self.display_structure(sentence, scores, start, pos)
        right_state = self.display_structure(sentence, scores, pos+1, end)
        word = self.vocab.id_to_word(sentence[pos])
        return f'({left_state}{word}{right_state})'


    def queue_state(self, sentence, scores, queue, h_in, h_out, c_in, c_out, i):
        # update the queue with tuple: (left_idx, middle_idx, right_idx)
        # -1 means None/zero
        # (-1, middel_idx, -1) is not allowed
        def _recurse(start, end):
            if end == start:
                return -1, []
            elif end == start+1:
                # NOTE: leaf word's embedding is copied from h_in/c_in into h_out/c_out
                h_out[i][start] = h_in[i][start]
                c_out[i][start] = c_in[i][start]
                return start, []
            pos = start + unwrap_scalar_variable(torch.max(scores[start:end], dim=0)[1]) # argmax
            collector = [log_softmax(scores[start:end], dim=0)[pos-start]]
            left_idx, left_collector = _recurse(start, pos)
            right_idx, right_collector = _recurse(pos+1, end)
            queue.append((left_idx, pos, right_idx))
            collector += left_collector + right_collector
            return pos, collector
        _, probs = _recurse(0, len(sentence))
        return probs


    def forward(self, sentence_embedding, sentence_word, length, display=False):
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


        if conf.fast: # TODO actually it is slower...
            assert self.cell_type == 'TriPad' and self.sample_num <= 1, 'multiple samples are not supported'
            h_in, c_in = hs, cs
            h_out, c_out = [[None for _ in range(max_length)] for __ in range(batch_size)], \
                    [[None for _ in range(max_length)] for __ in range(batch_size)]
            h_res, c_res = [None for _ in range(batch_size)], [None for _ in range(batch_size)]
            queue = [[] for _ in range(batch_size)]
            length = list(length.data)
            max_steps = 0
            structure = []
            samples = {'h':[None for _ in range(batch_size)], 'probs':[]}
            # construct queue state
            for i in range(batch_size):
                sentence = list(sentence_word[i].data)
                scores = self.calc_score(sentence, sentence_embedding[i])
                probs = self.queue_state(sentence, scores, queue[i], h_in, h_out, c_in, c_out, i)
                if len(probs) > 0:
                    probs = torch.stack(probs, dim=0) 
                samples['probs'].append(probs)
                max_steps = max(max_steps, len(queue[i]))
                # if sentence has only one word, then directly assign it to h_res/c_res
                if length[i]==1:
                    h_res[i] = h_in[i][0]
                    c_res[i] = c_in[i][0]
                    samples['h'][i] = h_in[i][0]
                if display:
                    structure.append(self.display_structure(sentence, scores, 0, length[i]))
            # compose the same level's computations into one batch
            for step in range(max_steps):
                hl, cl, hr, cr, hm, cm = [[self.zero_holder for _ in range(batch_size)] for __ in range(6)]
                for i in range(batch_size):
                    if step < len(queue[i]):
                        left_idx, middle_idx, right_idx = queue[i][step]
                        # two kids are from h_out and c_out
                        if left_idx != -1:
                            hl[i] = h_out[i][left_idx]
                            cl[i] = c_out[i][left_idx]
                            assert hl[i] is not None and cl[i] is not None
                        if right_idx != -1:
                            hr[i] = h_out[i][right_idx]
                            cr[i] = c_out[i][right_idx]
                            assert hr[i] is not None and cr[i] is not None
                        # parent is from h_in and c_in
                        hm[i] = h_in[i][middle_idx]
                        cm[i] = c_in[i][middle_idx]
                hl, cl, hr, cr, hm, cm = torch.stack(hl, dim=0), \
                        torch.stack(cl, dim=0), \
                        torch.stack(hr, dim=0), \
                        torch.stack(cr, dim=0), \
                        torch.stack(hm, dim=0), \
                        torch.stack(cm, dim=0)
                # batch computation
                h_step_out, c_step_out = self.treelstm_layer((hl, cl), (hr, cr), (hm, cm))
                # put embedding into h_out and c_out
                for i in range(batch_size):
                    if step < len(queue[i]):
                        left_idx, middle_idx, right_idx = queue[i][step]
                        h_out[i][middle_idx] = h_step_out[i]
                        c_out[i][middle_idx] = c_step_out[i]
                        # put embedding into h_res[i] and c_res[i] if queue[i] just out
                        if step == len(queue[i])-1:
                            h_res[i] = h_step_out[i]
                            c_res[i] = c_step_out[i]
                            samples['h'][i] = h_step_out[i]
            h_res, c_res = torch.stack(h_res, dim=0), torch.stack(c_res, dim=0)
            samples['h'] = torch.stack(samples['h'], dim=0)
            return h_res, c_res, structure, samples
        else:
            h_res, c_res, structure, samples = [], [], [], {}
            samples['h'], samples['probs'], samples['trees'] = [], [], []
            length = list(length.data)
            total_score, total_len = 0, 0
            
            self.scores = []


            current_sample_num = self.sample_num


            # iterate each sentence
            for i in range(batch_size):
                sentence = list(sentence_word[i].data)

                if self.rank_input == 'word':
                    embedding = sentence_embedding[i]
                elif self.rank_input == 'h':
                    embedding = hs[i]

                # calculate global scores for each word
                scores = self.calc_score(sentence, embedding)
                self.scores.append(scores)

                probs = defaultdict(list)
                state, tree = self.attend_compose(sentence, embedding, scores, hs[i].unsqueeze(1), cs[i].unsqueeze(1), 0, length[i], probs)
                h, c = state
                h_res.append(h)
                c_res.append(c)
                structure.append(tree)
                
                ##################################
                # sample for learning to rank
                if self.att_type != 'corpus':
                    # for j in range(self.sample_num):
                    for j in range(current_sample_num):
                        if j > 0: # if j==0, just use the state+probs from attend_compose to avoid one extra sample
                            probs = defaultdict(list)
                            state, tree = self.sample(sentence, embedding, scores, hs[i].unsqueeze(1), cs[i].unsqueeze(1), 0, length[i], probs)
                        samples['h'].append(state[0])
                        samples['probs'].append(probs) # a list of dict of Variable
                        samples['trees'].append(tree)
            h_res, c_res = torch.stack(h_res, dim=0), torch.stack(c_res, dim=0)
            h_res, c_res = h_res.squeeze(1), c_res.squeeze(1)
            if self.att_type != 'corpus':
                samples['h'] = torch.stack(samples['h'], dim=0).squeeze(1)

            return h_res, c_res, structure, samples

