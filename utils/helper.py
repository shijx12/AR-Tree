from torch.autograd import Variable
import torch
import numpy as np
import conf
from tensorboard import summary

def add_scalar_summary(summary_writer, name, value, step):
    value = unwrap_scalar_variable(value)
    summ = summary.scalar(name=name, scalar=value)
    summary_writer.add_summary(summary=summ, global_step=step)

def add_histo_summary(summary_writer, name, value, step):
    value = value.view(-1).data.cpu().numpy()
    summ = summary.histogram(name=name, values=value)
    summary_writer.add_summary(summary=summ, global_step=step)


def wrap_with_variable(tensor, volatile, gpu):
    if gpu > -1:
        return Variable(tensor.cuda(gpu), volatile=volatile)
    else:
        return Variable(tensor, volatile=volatile)


def unwrap_scalar_variable(var):
    if isinstance(var, Variable):
        return var.data[0]
    else:
        return var

def unwrap_variable(var):
    if isinstance(var, Variable):
        return var.data
    else:
        return var

def _parse_word_depth(length, select_masks):
    """
    Args:
        length: (batch_size, ). Variable containing an IntTensor
            the sentence length of a batch
        select_masks: a list of length (max_len-2), i-th element is a Tensor of shape (batch_size, max_len-1-i)
            parse results from <BinaryTreeLSTM.select_composition>
    Return:
        Parsed results for each sentence. 
        Specifically, parsed_results[i][j] is the depth of j-th word of i-th sentence. 
    """
    batch_size = length.size(0)
    length = unwrap_variable(length)
    parsed_results = [] # for each batch/sentence
    for idx in range(batch_size):
    # idx-th batch, length of sentence = l
        l = length[idx]
        chart = np.zeros((l, l))
        for i in range(l-1,0,-1):
            # i-th row of chart, update (i-1)-th row of chart based on (i-1)-th element of select_masks
            bias = 0 # next row bias
            for j in range(l-i):
                # j-th element of chart[i] and select_masks[i-1][idx]
                # Note: len(select_masks)==(max_len-2), so the case of l==max_len should be considered
                if i > len(select_masks) or select_masks[i-1][idx][j] == 1:
                    chart[i-1][j+bias] = chart[i][j]+1
                    bias += 1
                    chart[i-1][j+bias] = chart[i][j]+1
                else:
                    chart[i-1][j+bias] = chart[i][j]
            if conf.debug and bias != 1:
                from IPython import embed; embed()
            assert bias==1, 'Wrong select_masks!'
        # from IPython import embed; embed()
        parsed_results.append(chart[0])
    return parsed_results



def parse_tree_avg_depth(length, select_masks):
    depth_cumsum = word_cumsum = 0
    parsed_results = _parse_word_depth(length, select_masks)
    for sent in parsed_results:
        depth_cumsum += sent.sum()
        word_cumsum += sent.shape[0]
    return depth_cumsum, word_cumsum


def parse_tree(vocab, words, length, select_masks):
    """
    Args:
        vocab: Vocab instance, whose id_to_word method will be used to convert words 
        words: Tensor, id for words in sentences, (batch_size, max_len)
    Return:
        Parse tree represented by brackets, a list of string whose len=batch_size
    """
    batch_size = length.size(0)
    length = unwrap_variable(length)
    parsed_results = _parse_word_depth(length, select_masks)
    trees = []
    for idx in range(batch_size):
    # idx-th batch/sentence
        l = length[idx]
        words_ = [vocab.id_to_word(words[idx][j]) for j in range(l)]
        depths = [parsed_results[idx][j] for j in range(l)]
        stack = []
        ptr = 0
        for j in range(2*l-1):
            if len(stack) < 2 or stack[-1][1] != stack[-2][1]:
                stack.append((words_[ptr], depths[ptr]))
                ptr += 1
            else:
                parent = (f'({stack[-2][0]} {stack[-1][0]})', stack[-1][1]-1)
                stack.pop(); stack.pop()
                stack.append(parent)
        if conf.debug and not (len(stack)==1 and stack[0][1]==0):
            print('Wrong stack when parse tree')
            from IPython import embed; embed()
        assert len(stack)==1 and stack[0][1]==0, 'Wrong final stack states!'
        trees.append(stack[0][0])
    return trees
        
