# Attentive Recursive Tree (AR-Tree)
This repository is the pytorch implementation of paper

[Learning to Embed Sentences Using Attentive Recursive Trees](https://arxiv.org/abs/1811.02338).
> [Jiaxin Shi](https://shijx12.github.io), Lei Hou, Juanzi Li, [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/index.html), [Hanwang Zhang](http://www.ntu.edu.sg/home/hanwangzhang/#aboutme).

In this paper, we propose an Attentive Recursive Tree model (AR-Tree), where the words are dynamically located according to their importance in the task. Specifically, we construct the latent tree for a sentence in a proposed important-first strategy, and place more attentive words nearer to the root; thus, AR-Tree can inherently emphasize important words during the bottom-up composition of the sentence embedding.
If you find this code useful in your research, please cite
``` tex
@InProceedings{jiaxin_ARTree,
author = {Jiaxin Shi, Lei Hou, Juanzi Li, Zhiyuan Liu, Hanwang Zhang},
title = {Learning to Embed Sentences Using Attentive Recursive Trees},
booktitle = {AAAI},
year = {2019}
}
```

## Requirements
- python==3.6
- pytorch==0.4.0
- ete3
- torchtext
- nltk

## Preprocessing
Before training the model, you need to first prepare data.
First of all, you need to download the [GloVe 300d pretrained vector](http://nlp.stanford.edu/data/glove.840B.300d.zip) as we use it for initialization in all experiments.
After unzipping it, you need to convert the txt file to pickle file by 
``` shell
python pickle_glove.py --txt </path/to/840B.300d.txt> --pt </output/file/name>
```
Next we begin to prepare training corpus.

#### SNLI
1. Download the [SNLI 1.0 corpus](https://nlp.stanford.edu/projects/snli/snli_1.0.zip).
2. Preprocess the original SNLI corpus and create the cache file by the following command:
``` shell
python snli/dump_dataset.py --data </path/to/the/corpus> --out </path/to/the/output/file>
```
The output file will be used in the data loader when training or testing.

#### SST
1. Download the [SST corpus](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip). OK that's enough, the torchtext package will help us.

#### Age
1. We have attach this corpus as the file `age/age2.zip`. You need to unzip it first.
2. Create the cache file by the following command:
``` shell
python age/dump_dataset.py --glove-path </path/to/840B.300d.txt> --data-dir </path/to/unzipped/folder> --save-path </output/file/name>
```


## Train
You can directly run these scripts to train the AR-Tree on different datasets: 
- `snli/run_snli.sh` to train on SNLI.
- `sst/run_sst2.sh` to train on SST2.
- `sst/run_sst5.sh` to train on SST5.
- `age/run_age.sh` to train on Age.
*Note* that you should change the argument value of `--data-path`, `--glove-path`, and `--save-dir` according to your directory.

We implement two training strategies, which can be specified by the argument `--model-type`.
The reinforcement learning described in our paper is selected by `--model-type RL`.
Another implementation is `--model-type STG`, which uses straight-through gumble softmax instead of REINFORCE.
`--model-type Choi` corresponds to [Choi's TreeLSTM model](https://arxiv.org/abs/1707.02786), regarded as a baseline in our paper.

## Test
You can run `evaluate.py` for testing:
``` shell
python evaluate.py --ckpt </path/to/checkpoint> --data-path </path/to/data> --mode ['vis', 'val']
```
Note that `--mode vis` is used for visualization of the learned tree structures, while `--mode val` is to calculate the accuracy on the test set.

## Acknowledgement
We refer to some codes of these repos:
- [Choi's implementation](https://github.com/jihunchoi/unsupervised-treelstm) of his paper [Learning to Compose Task-Specific Tree Structures](https://arxiv.org/abs/1707.02786).
- [the implementation of self-attentive](https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding) of paper [A Structured Self-Attentive Sentence Embedding](https://arxiv.org/abs/1703.03130).
Appreciate for their great contributions!
