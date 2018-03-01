import torch
from torch import nn
from torch.nn import init

from model.treelstm import BinaryTreeLSTM
from model.att_tree import AttTreeLSTM


class SNLIClassifier(nn.Module):

    def __init__(self, num_classes, input_dim, hidden_dim, num_layers,
                 use_batchnorm, dropout_prob):
        super(SNLIClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        if use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=4 * input_dim)
            self.bn_mlp_output = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        mlp_layers = []
        for i in range(num_layers):
            layer_in_features = hidden_dim if i > 0 else 4 * input_dim
            linear_layer = nn.Linear(in_features=layer_in_features,
                                     out_features=hidden_dim)
            relu_layer = nn.ReLU()
            mlp_layer = nn.Sequential(linear_layer, relu_layer)
            mlp_layers.append(mlp_layer)
        self.mlp = nn.Sequential(*mlp_layers)
        self.clf_linear = nn.Linear(in_features=hidden_dim,
                                    out_features=num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_batchnorm:
            self.bn_mlp_input.reset_parameters()
            self.bn_mlp_output.reset_parameters()
        for i in range(self.num_layers):
            linear_layer = self.mlp[i][0]
            init.kaiming_normal(linear_layer.weight.data)
            init.constant(linear_layer.bias.data, val=0)
        init.uniform(self.clf_linear.weight.data, -0.005, 0.005)
        init.constant(self.clf_linear.bias.data, val=0)

    def forward(self, pre, hyp):
        f1 = pre
        f2 = hyp
        f3 = torch.abs(pre - hyp)
        f4 = pre * hyp
        mlp_input = torch.cat([f1, f2, f3, f4], dim=1)
        if self.use_batchnorm:
            mlp_input = self.bn_mlp_input(mlp_input)
        mlp_input = self.dropout(mlp_input)
        mlp_output = self.mlp(mlp_input)
        if self.use_batchnorm:
            mlp_output = self.bn_mlp_output(mlp_output)
        mlp_output = self.dropout(mlp_output)
        logits = self.clf_linear(mlp_output)
        return logits


class SNLIBinaryModel(nn.Module):

    def __init__(self, num_classes, num_words, word_dim, hidden_dim,
                 clf_hidden_dim, clf_num_layers, use_leaf_rnn, intra_attention,
                 use_batchnorm, dropout_prob, bidirectional, 
                 weighted_by_interval_length, weighted_base,
                 weighted_update, cell_type):
        super(SNLIBinaryModel, self).__init__()
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_num_layers = clf_num_layers
        self.use_leaf_rnn = use_leaf_rnn
        self.intra_attention = intra_attention
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional

        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        self.encoder = BinaryTreeLSTM(word_dim=word_dim, hidden_dim=hidden_dim,
                                      use_leaf_rnn=use_leaf_rnn,
                                      intra_attention=intra_attention,
                                      gumbel_temperature=1,
                                      bidirectional=bidirectional,
                                      weighted_by_interval_length=weighted_by_interval_length,
                                      weighted_base=weighted_base,
                                      weighted_update=weighted_update,
                                      cell_type=cell_type)
        if bidirectional:
            clf_input_dim = 2 * hidden_dim
        else:
            clf_input_dim = hidden_dim
        self.classifier = SNLIClassifier(
            num_classes=num_classes, input_dim=clf_input_dim,
            hidden_dim=clf_hidden_dim, num_layers=clf_num_layers,
            use_batchnorm=use_batchnorm, dropout_prob=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, pre, pre_length, hyp, hyp_length):
        pre_embeddings = self.word_embedding(pre)
        hyp_embeddings = self.word_embedding(hyp)
        pre_embeddings = self.dropout(pre_embeddings)
        hyp_embeddings = self.dropout(hyp_embeddings)
        supplements = {}
        pre_h, _, pre_select_masks = self.encoder(input=pre_embeddings, length=pre_length, return_select_masks=True)
        hyp_h, _, hyp_select_masks = self.encoder(input=hyp_embeddings, length=hyp_length, return_select_masks=True)
        logits = self.classifier(pre=pre_h, hyp=hyp_h)
        supplements['pre_select_masks'] = pre_select_masks
        supplements['hyp_select_masks'] = hyp_select_masks
        return logits, supplements


class SNLIAttModel(nn.Module):

    def __init__(self, vocab, num_classes, num_words, word_dim, hidden_dim,
                 clf_hidden_dim, clf_num_layers, use_leaf_rnn, 
                 use_batchnorm, dropout_prob, bidirectional, cell_type,
                 att_type, sample_num, rich_state):
        super(SNLIAttModel, self).__init__()
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_num_layers = clf_num_layers
        self.use_leaf_rnn = use_leaf_rnn
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional
        self.att_type = att_type
        self.sample_num = sample_num

        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        self.encoder = AttTreeLSTM(vocab=vocab,
                                    word_dim=word_dim, 
                                    hidden_dim=hidden_dim,
                                    use_leaf_rnn=use_leaf_rnn,
                                    bidirectional=bidirectional,
                                    cell_type=cell_type,
                                    att_type=att_type,
                                    sample_num=sample_num,
                                    rich_state=rich_state)
        if bidirectional:
            clf_input_dim = 2 * hidden_dim
        else:
            clf_input_dim = hidden_dim
        self.classifier = SNLIClassifier(
            num_classes=num_classes, input_dim=clf_input_dim,
            hidden_dim=clf_hidden_dim, num_layers=clf_num_layers,
            use_batchnorm=use_batchnorm, dropout_prob=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, pre, pre_length, hyp, hyp_length, display=False):
        pre_embeddings = self.word_embedding(pre)
        hyp_embeddings = self.word_embedding(hyp)
        pre_embeddings = self.dropout(pre_embeddings)
        hyp_embeddings = self.dropout(hyp_embeddings)
        pre_h, _, pre_tree, pre_samples = self.encoder(
                sentence_embedding=pre_embeddings, 
                sentence_word=pre, 
                length=pre_length, display=display)
        hyp_h, _, hyp_tree, hyp_samples = self.encoder(
                sentence_embedding=hyp_embeddings, 
                sentence_word=hyp, 
                length=hyp_length, display=display)
        logits = self.classifier(pre=pre_h, hyp=hyp_h)
        supplements = {'pre_tree': pre_tree, 'hyp_tree': hyp_tree}
        ######################
        # samples prediction for REINFORCE
        if self.att_type != 'corpus':
            sample_logits = self.classifier(pre=pre_samples['h'], hyp=hyp_samples['h']) if self.sample_num > 0 else None
            supplements['sample_logits'] = sample_logits
            supplements['pre_probs'] = pre_samples['probs']
            supplements['hyp_probs'] = hyp_samples['probs']
        return logits, supplements
