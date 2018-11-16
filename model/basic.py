import torch
from torch import nn
from torch.nn import functional, init, Parameter
import numpy as np

class Node():
    def __init__(self, word, left=None, right=None):
        self.word = word
        self.left = left
        self.right = right


class NaryLSTMLayer(nn.Module): # N-ary Tree-LSTM in the paper of treelstm
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=3 * hidden_dim,
                                    out_features=5 * hidden_dim)
        self.zero = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.comp_linear.weight.data)
        init.constant_(self.comp_linear.bias.data, val=0)

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

class TriPadLSTMLayer(nn.Module): # used in my paper
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=3 * hidden_dim,
                                    out_features=6 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.comp_linear.weight.data)
        init.constant_(self.comp_linear.bias.data, val=0)

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
        zero = torch.zeros(1, self.hidden_dim).to(hm.device)
        if l is None:
            l = (zero, zero)
        if r is None:
            r = (zero, zero)
        hr, cr = r
        hl, cl = l
        h_cat = torch.cat([hl, hm, hr], dim=1)
        comp_vector = self.comp_linear(h_cat)
        i, fl, fm, fr, u, o = torch.chunk(comp_vector, chunks=6, dim=1)
        c = cl*(fl+1).sigmoid() + cm*(fm+1).sigmoid() + cr*(fr+1).sigmoid() + u.tanh()*i.sigmoid()
        h = o.sigmoid() * c.tanh()
        return h, c 


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        nn.LayerNorm = LayerNorm
        self.ln_i2h = nn.LayerNorm(4 * hidden_size)
        self.ln_h2h = nn.LayerNorm(4 * hidden_size)
        self.ln_c = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.i2h.weight.data)
        init.orthogonal(self.h2h.weight.data)
        init.constant(self.i2h.bias.data, val=0)
        init.constant(self.h2h.bias.data, val=0)

    def forward(self, input, hx):
        h_prev, c_prev = hx
        hi = self.ln_i2h(self.i2h(input))
        hh = self.ln_h2h(self.h2h(h_prev))
        i, f, u, o = torch.chunk(hi + hh, chunks=4, dim=1)
        c = self.ln_c(c_prev*(f + 1).sigmoid() + u.tanh()*i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


class ForgetMoreLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ForgetMoreLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.i2h.weight.data)
        init.orthogonal(self.h2h.weight.data)
        init.constant(self.i2h.bias.data, val=0)
        init.constant(self.h2h.bias.data, val=0)

    def forward(self, input, hx):
        h_prev, c_prev = hx
        hi = self.i2h(input)
        hh = self.h2h(h_prev)
        i, f, u, o = torch.chunk(hi + hh, chunks=4, dim=1)
        c = c_prev*(f + 1).sigmoid() + u.tanh()*i.sigmoid()
        h = o.sigmoid() * c.tanh()
        return h, c




def apply_nd(fn, input):
    """
    Apply fn whose output only depends on the last dimension values
    to an arbitrary n-dimensional input.
    It flattens dimensions except the last one, applies fn, and then
    restores the original size.
    """

    x_size = input.size()
    x_flat = input.view(-1, x_size[-1])
    output_flat = fn(x_flat)
    output_size = x_size[:-1] + (output_flat.size(-1),)
    output_flat = output_flat.view(*output_size)
    return output_flat


def affine_nd(input, weight, bias):
    """
    An helper function to make applying the "wx + b" operation for
    n-dimensional x easier.

    Args:
        input (tensor): An arbitrary input data, whose size is
            (d0, d1, ..., dn, input_dim)
        weight (tensor): A matrix of size (output_dim, input_dim)
        bias (tensor): A bias vector of size (output_dim,)

    Returns:
        output: The result of size (d0, ..., dn, output_dim)
    """

    input_size = input.size()
    input_flat = input.view(-1, input_size[-1])
    bias_expand = bias.unsqueeze(0).expand(input_flat.size(0), bias.size(0))
    output_flat = torch.addmm(bias_expand, input_flat, weight)
    output_size = input_size[:-1] + (weight.size(1),)
    output = output_flat.view(*output_size)
    return output


def dot_nd(query, candidates):
    """
    Perform a dot product between a query and n-dimensional candidates.

    Args:
        query (tensor): A vector to query, whose size is
            (query_dim,)
        candidates (tensor): A n-dimensional tensor to be multiplied
            by query, whose size is (d0, d1, ..., dn, query_dim)

    Returns:
        output: The result of the dot product, whose size is
            (d0, d1, ..., dn)
    """

    cands_size = candidates.size()
    cands_flat = candidates.view(-1, cands_size[-1])
    output_flat = torch.mv(cands_flat, query)
    output = output_flat.view(*cands_size[:-1])
    return output


def convert_to_one_hot(indices, num_classes):
    """
    Args:
        indices (tensor): A vector containing indices,
            whose size is (batch_size,).
        num_classes (tensor): The number of classes, which would be
            the second dimension of the resulting one-hot matrix.

    Returns:
        result: The one-hot matrix of size (batch_size, num_classes).
    """

    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = torch.Tensor(batch_size, num_classes).zero_().to(indices.device)\
                .scatter_(1, indices.data, 1)
    return one_hot


def masked_softmax(logits, mask=None):
    eps = 1e-20
    dim = 0 if logits.ndimension() == 1 else 1
    probs = functional.softmax(logits, dim=dim)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(dim, keepdim=True)
    return probs

def weighted_softmax(logits, base, mask=None, weights_mask=None):
    """
    weights_mask maintains the length of corresponding intervals.
    for example: regardless of the batch dimension, i-th logits corresponds an interval of length l, then its probability will multiply pow(base, l)
    """
    eps = 1e-20
    probs = functional.softmax(logits, dim=1)
    if weights_mask is not None and mask is not None:
        weights = torch.pow(base, weights_mask.float())
        probs = probs * mask.float() * weights + eps
        probs = probs / probs.sum(1, keepdim=True)
    return probs

def greedy_select(logits, mask=None):
    probs = masked_softmax(logits=logits, mask=mask)
    one_hot = convert_to_one_hot(indices=probs.max(1)[1],
                                 num_classes=logits.size(1))
    return one_hot


def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    """
    Return the result of Straight-Through Gumbel-Softmax Estimation.
    It approximates the discrete sampling via Gumbel-Softmax trick
    and applies the biased ST estimator.
    In the forward propagation, it emits the discrete one-hot result,
    and in the backward propagation it approximates the categorical
    distribution via smooth Gumbel-Softmax distribution.

    Args:
        logits (Tensor): A un-normalized probability values,
            which has the size (batch_size, num_classes) or (num_classes, )
        temperature (float): A temperature parameter. The higher
            the value is, the smoother the distribution is.
        mask (Tensor, optional): If given, it masks the softmax
            so that indices of '0' mask values are not selected.
            The size is (batch_size, num_classes).
        weights (Tensor, optional) : Must have the same size with mask
    Returns:
        y: The sampled output, which has the property explained above.
    """

    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_(0.001, 0.999)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    y = masked_softmax(logits=y / temperature, mask=mask)
    if logits.ndimension() == 1: # no batch dimension
        y_argmax = y.max(0)[1]
        y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(0)).float().squeeze()
    else:
        y_argmax = y.max(1)[1]
        y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y


def sequence_mask(sequence_length, max_length=None):
    if max_length is None:
        max_length = sequence_length.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length).to(sequence_length.device)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Tensor): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Tensor with the same size as inputs, but with each sequence
        reversed according to its length.
    """

    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length-1::-1]
    reversed_indices = torch.LongTensor(reversed_indices).unsqueeze(2)\
                        .expand_as(inputs).to(inputs.device)
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs
