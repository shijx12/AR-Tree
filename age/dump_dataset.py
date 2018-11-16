import os
import sys
import string
import numpy
import pickle
import numpy as np
import nltk
import argparse


###################################################
#                Overall stats
###################################################
#                       # entries            # dims
# adapted word2vec         530158               100
# glove                   2196016               300
###################################################
#                            age1     age2     yelp
# train data                    -    68485   500000
# dev/test data                 -     4000     2000
# word2vec known token              126862   233293
#            UNK token               93124   184427
# glove    known token              126862   233293
#            UNK token               49268   104717 
###################################################
#                                   120739
#                                    43256    88984
parser = argparse.ArgumentParser() 
parser.add_argument('--glove-path', required=True, help='840B.300d.txt file')
parser.add_argument('--data-dir', required=True)
parser.add_argument('--save-path', required=True)
args = parser.parse_args()

glove_path = args.glove_path
data_dir = args.data_dir
save_path = args.save_path


print("loading GloVe...")
w1 = {}
for line in open(glove_path):
    line=line.split(' ')
    w1[line[0]] = np.asarray([float(x) for x in line[1:]]).astype('float32')



f1 = os.path.join(data_dir, 'age2_train')
f2 = os.path.join(data_dir, 'age2_valid')
f3 = os.path.join(data_dir, 'age2_test')
# note that class No. = rating -1
classname = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
f = [f1, f2, f3]


print("processing dataset, 3 dots to punch: ")
w2 = {}
w_referred = {'<pad>': 0}  # reserve 0 for future padding
vocab_count = 1  # 0 is reserved for future padding
train_dev_test = []
for file in f:
    pairs = []
    for line in open(file):
        line=line.strip().split()
        s1 = line[1:]
        s1[0]=s1[0].lower()

        rate_score = classname[line[0]]

        s1_words = []
        for word in s1:
            if word not in w_referred:
                w_referred[word] = vocab_count
                vocab_count += 1
            s1_words.append(w_referred[word])
            if word not in w1:
                if word not in w2:
                    w2[word]=[]
                # find the WE for its surounding words
                for neighbor in s1:
                    if neighbor in w1:
                        w2[word].append(w1[neighbor])

        pairs.append((numpy.asarray(s1_words).astype('int32'),
                      rate_score))
    train_dev_test.append(pairs)

print("augmenting word embedding vocabulary...")
mean_words = np.zeros((len(w1['the']),))
mean_words_std = 1e-1

npy_rng = np.random.RandomState(123)
for k in w2:
    if len(w2[k]) != 0:
        w2[k] = sum(w2[k]) / len(w2[k])  # mean of all surounding words
    else:
        w2[k] = mean_words + npy_rng.randn(mean_words.shape[0]) * \
                             mean_words_std * 0.1

w2.update(w1)

print("generating weight values...")
# reverse w_referred's key-value;
inv_w_referred = {v: k for k, v in w_referred.items()}

# number   --inv_w_referred-->   word   --w2-->   embedding
ordered_word_embedding = [numpy.zeros((1, len(w1['the'])), dtype='float32'), ] + \
    [w2[inv_w_referred[n]].reshape(1, -1) for n in range(1, len(inv_w_referred))]

# to get the matrix
weight = numpy.concatenate(ordered_word_embedding, axis=0)


print("dumping converted datasets...")
save_file = open(save_path, 'wb')
pickle.dump(train_dev_test, save_file)
pickle.dump(weight, save_file)
pickle.dump(w_referred, save_file)
pickle.dump(inv_w_referred, save_file)
save_file.close()

