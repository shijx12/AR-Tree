from torch import nn
from torch.nn import init

from model.treelstm import BinaryTreeLSTM
from model.att_tree import AttTreeLSTM


class SSTClassifier(nn.Module):

    def __init__(self, num_classes, input_dim, hidden_dim, num_layers,
                 use_batchnorm, dropout_prob):
        super(SSTClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        if use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=input_dim)
            self.bn_mlp_output = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        mlp_layers = []
        for i in range(num_layers):
            layer_in_features = hidden_dim if i > 0 else input_dim
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
        init.uniform(self.clf_linear.weight.data, -0.002, 0.002)
        init.constant(self.clf_linear.bias.data, val=0)

    def forward(self, sentence):
        mlp_input = sentence
        if self.use_batchnorm:
            mlp_input = self.bn_mlp_input(mlp_input)
        mlp_input = self.dropout(mlp_input)
        mlp_output = self.mlp(mlp_input)
        if self.use_batchnorm:
            mlp_output = self.bn_mlp_output(mlp_output)
        mlp_output = self.dropout(mlp_output)
        logits = self.clf_linear(mlp_output)
        return logits


class SSTModel(nn.Module):

    def __init__(self, num_classes, num_words, word_dim, hidden_dim,
                 clf_hidden_dim, clf_num_layers, use_leaf_rnn, bidirectional,
                 use_batchnorm, dropout_prob,
                 weighted_by_interval_length, weighted_base, weighted_update,
                 cell_type):
        super(SSTModel, self).__init__()
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_num_layers = clf_num_layers
        self.use_leaf_rnn = use_leaf_rnn
        self.bidirectional = bidirectional
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(dropout_prob)
        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        self.encoder = BinaryTreeLSTM(word_dim=word_dim, hidden_dim=hidden_dim,
                                      use_leaf_rnn=use_leaf_rnn,
                                      intra_attention=False,
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
        self.classifier = SSTClassifier(
            num_classes=num_classes, input_dim=clf_input_dim,
            hidden_dim=clf_hidden_dim, num_layers=clf_num_layers,
            use_batchnorm=use_batchnorm, dropout_prob=dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, words, length, display=False):
        words_embed = self.word_embedding(words)
        words_embed = self.dropout(words_embed)
        sentence_vector, _, select_masks = self.encoder(input=words_embed, length=length, return_select_masks=True)
        logits = self.classifier(sentence_vector)
        supplements = {'select_masks': select_masks}
        return logits, supplements


class SSTAttModel(nn.Module):

    def __init__(self, vocab, num_classes, num_words, word_dim, hidden_dim,
                 clf_hidden_dim, clf_num_layers, use_leaf_rnn, bidirectional,
                 use_batchnorm, dropout_prob,
                 cell_type, att_type, sample_num):
        super(SSTAttModel, self).__init__()
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_num_layers = clf_num_layers
        self.use_leaf_rnn = use_leaf_rnn
        self.bidirectional = bidirectional
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob
        self.att_type = att_type
        self.sample_num = sample_num

        self.dropout = nn.Dropout(dropout_prob)
        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        self.encoder =  AttTreeLSTM(vocab=vocab,
                                    word_dim=word_dim, 
                                    hidden_dim=hidden_dim,
                                    use_leaf_rnn=use_leaf_rnn,
                                    bidirectional=bidirectional,
                                    cell_type=cell_type,
                                    att_type=att_type,
                                    sample_num=sample_num)
        if bidirectional:
            clf_input_dim = 2 * hidden_dim
        else:
            clf_input_dim = hidden_dim
        self.classifier = SSTClassifier(
            num_classes=num_classes, input_dim=clf_input_dim,
            hidden_dim=clf_hidden_dim, num_layers=clf_num_layers,
            use_batchnorm=use_batchnorm, dropout_prob=dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, words, length, display=False):
        words_embed = self.word_embedding(words)
        words_embed = self.dropout(words_embed)
        sentence_vector, _, trees, samples = self.encoder(
                sentence_embedding=words_embed, 
                sentence_word=words,
                length=length, display=display)
        logits = self.classifier(sentence_vector)
        supplements = {'trees': trees}
        ###########################
        # samples prediction for REINFORCE
        if self.att_type != 'corpus' and self.sample_num > 0:
            sample_logits = self.classifier(samples['h'])
            supplements['sample_logits'] = sample_logits
            supplements['probs'] = samples['probs']
        return logits, supplements
