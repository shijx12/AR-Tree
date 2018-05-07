import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init, Parameter
from utils.helper import unwrap_scalar_variable
from .STGumbel_SA_Tree import STGumbelSaTree
from .basic import LayerNormLSTMCell, ForgetMoreLSTMCell
from IPython import embed


class SelfSegmentalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfSegmentalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.boundary_detector = nn.Sequential(
                nn.Linear(in_features=2*hidden_size, out_features=300),
                nn.Tanh(),
                nn.Linear(in_features=300, out_features=1),
                ) 

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.lstm_cell.weight_ih.data)
        init.orthogonal(self.lstm_cell.weight_hh.data)
        init.constant(self.lstm_cell.bias_ih.data, val=0)
        init.constant(self.lstm_cell.bias_hh.data, val=0)
        for layer in self.boundary_detector:
            if type(layer) == nn.Linear:
                init.kaiming_normal(layer.weight.data)
                init.constant(layer.bias.data, val=0)

    def forward(self, embeddings, sentence_words, lengths, temperature):
        """
        Args:
            embeddings: (batch_size, max_length, input_size). word embeddings of sentences
            sentence_words: list of (batch_size, max_length). word strings
            lengths: (batch_size, ). sentence lengths
        return:
            phrase_hs: (batch_size, max_phrase_length, hidden_size)
            phrase_cs
            phrase_words 
            phrase_lengths: (batch_size, ). The number of phrases of each sentence
        """
        batch_size, max_length, _ = embeddings.size()
        lengths = lengths.unsqueeze(dim=1)
        phrase_hs = [[] for _ in range(batch_size)]
        phrase_cs = [[] for _ in range(batch_size)]
        phrase_words = [[] for _ in range(batch_size)]
        b = [0] * batch_size # last boundary index
        phrase_lengths = Variable(embeddings.data.new(batch_size, 1).zero_().int(), requires_grad=False)
        h_prev = c_prev = Variable(embeddings.data.new(batch_size, self.hidden_size).zero_())
        
        
        for step in range(1, max_length+1): # word index from 1 to max_length
            h, c = self.lstm_cell(input=embeddings[:, step-1, :], hx=(h_prev, c_prev))
            bd_input = torch.cat([h_prev, embeddings[:, step-1, :]], dim=1)
            z_soft = self.boundary_detector(bd_input)
            z_soft = functional.sigmoid(z_soft / temperature)
            # When detector's output >= 0.5, or lengths[i]==step, there exists a boundary after the (step-1)-th word. When lengths[i]<step, boundary never exists.
            # With bernoulli, temperature annealing is not suitable.
            z_hard = (torch.bernoulli(z_soft).byte() | lengths.eq(step)) & lengths.ge(step)
            # z_hard = (z_soft.ge(0.5) | lengths.eq(step)) & lengths.ge(step)
            gate = (z_hard.float()-z_soft).detach()+z_soft
            gate_h, gate_c = gate * h, gate * c
            for i in range(batch_size):
                phrase_hs[i].append(gate_h[i])
                phrase_cs[i].append(gate_c[i])
                if unwrap_scalar_variable(z_hard[i]) == 1: # Find a boundary of sentence i
                    phrase_words[i].append(' '.join(sentence_words[i][b[i]:step]))
                    b[i] = step
                else:
                    phrase_words[i].append('')
            phrase_lengths += 1 
            h_prev = h
            c_prev = (1-z_hard.float()) * c
            # print(torch.stack((z_soft, z_hard.float()), dim=1).squeeze())
        
        ''' 
        for step in range(1, max_length+1):
            h, c = self.lstm_cell(input=embeddings[:, step-1, :], hx=(h_prev, c_prev))
            if step < max_length:
                h_next, c_next = self.lstm_cell(input=embeddings[:, step, :], hx=(h, c))
                f1 = h
                f2 = h_next
                f3 = h_next - h
                f4 = h * h_next
                bd_input = torch.cat([f1, f2, f3, f4], dim=1) # TODO
                z_soft = self.boundary_detector(bd_input)
                z_soft = functional.sigmoid(z_soft / temperature)
            else:
                z_soft = Variable(embeddings.data.new(batch_size, 1).zero_(), requires_grad=False)
            z_hard = (z_soft.ge(0.5) | lengths.eq(step)) & lengths.ge(step)
            z_hard = z_hard.float()
            gate = (z_hard.float()-z_soft).detach()+z_soft
            for i in range(batch_size):
                phrase_hs[i].append(gate[i]*h[i])
                phrase_cs[i].append(gate[i]*c[i]) # gate measures whether h,c can be utilized by SA-Tree
                if unwrap_scalar_variable(z_hard[i]) == 1:
                    phrase_words[i].append(' '.join(sentence_words[i][b[i]:step]))
                    b[i] = step
                else:
                    phrase_words[i].append('')
            phrase_lengths += 1 
            h_prev = (1-z_hard) * h
            c_prev = (1-z_hard) * c
            '''


        # padding and construct Variable
        max_length = max([len(_) for _ in phrase_words])
        zero_pad = Variable(embeddings.data.new(self.hidden_size).zero_())
        phrase_hs = torch.stack(
                [torch.stack(s+[zero_pad]*(max_length-len(s)), dim=0) for s in phrase_hs],
                dim=0)
        phrase_cs = torch.stack(
                [torch.stack(s+[zero_pad]*(max_length-len(s)), dim=0) for s in phrase_cs],
                dim=0)
        return phrase_hs, phrase_cs, phrase_words, phrase_lengths.view(-1)



class SelfSegAttenTree(nn.Module):

    def __init__(self, **kwargs):
        super(SelfSegAttenTree, self).__init__()
        self.vocab = kwargs['vocab']
        hidden_dim = self.hidden_dim = kwargs['hidden_dim'] 
        self.cell_type = kwargs['cell_type'] 
        self.rich_state = kwargs['rich_state'] 
        self.rank_init = kwargs['rank_init'] 
        self.rank_detach = kwargs['rank_detach'] 
        word_dim = kwargs['word_dim']
        assert self.vocab.id_to_word
        kwargs['use_leaf_rnn'] = False
        kwargs['bidirectional'] = False
        kwargs['rank_input'] = 'h'
        kwargs['word_dim'] = hidden_dim
        
        self.temperature = 1
        self.selfseglstm = SelfSegmentalLSTM(word_dim, hidden_dim)
        self.attentree = STGumbelSaTree(**kwargs)

    def forward(self, sentence_embeddings, sentence_words, lengths):
        sentence_words = [list(map(self.vocab.id_to_word, list(sentence_words[i].data))) for i in range(sentence_words.size(0))]
        phrase_hs, phrase_cs, phrase_words, phrase_lengths = self.selfseglstm(sentence_embeddings, sentence_words, lengths, self.temperature)
        h_res, c_res, structure = self.attentree((phrase_hs, phrase_cs), phrase_words, phrase_lengths)
        return h_res, c_res, structure

