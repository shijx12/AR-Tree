import torch
from torch import nn
from torch.nn import init




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


class SNLIModel(nn.Module):

    def __init__(self, typ, **kwargs):
        '''
        arguments:
        typ=='Choi':
            num_classes, num_words, word_dim, hidden_dim,
            clf_hidden_dim, clf_num_layers, use_leaf_rnn, intra_attention,
            use_batchnorm, dropout_prob, bidirectional, 
            weighted_by_interval_length, weighted_base,
            weighted_update, cell_type
        typ=='RL-SA':
            vocab, num_classes, num_words, word_dim, hidden_dim,
            clf_hidden_dim, clf_num_layers, use_leaf_rnn, 
            use_batchnorm, dropout_prob, bidirectional, cell_type, att_type, sample_num, 
            rich_state, rank_init, rank_input, rank_detach, rank_tanh
        '''
        super(SNLIModel, self).__init__()
        num_classes = kwargs['num_classes']
        hidden_dim = kwargs['hidden_dim']
        clf_hidden_dim = kwargs['clf_hidden_dim']
        clf_num_layers = kwargs['clf_num_layers']
        use_batchnorm = kwargs['use_batchnorm']
        dropout_prob = kwargs['dropout_prob']
        bidirectional = kwargs['bidirectional']
        self.typ = typ

        self.word_embedding = nn.Embedding(num_embeddings=kwargs['num_words'],
                                           embedding_dim=kwargs['word_dim'])
        if typ == 'Choi':
            from model.Choi_Tree import BinaryTreeLSTM
            kwargs['gumbel_temperature'] = 1
            self.encoder = BinaryTreeLSTM(**kwargs)
        elif typ == 'tfidf-SA':
            from model.tfidf_Tree import tfidfTree
            self.encoder = tfidfTree(**kwargs)
        elif typ == 'RL-SA':
            from model.RL_SA_Tree import RlSaTree
            self.encoder = RlSaTree(**kwargs)
        elif typ == 'STG-SA':
            from model.STGumbel_SA_Tree import STGumbelSaTree
            self.encoder = STGumbelSaTree(**kwargs)

        clf_input_dim = 2*hidden_dim if bidirectional else hidden_dim
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

        ############################################################################
        if self.typ == 'Choi': 
            supplements = {}
            pre_h, _, pre_select_masks = self.encoder(input=pre_embeddings, length=pre_length, return_select_masks=True)
            hyp_h, _, hyp_select_masks = self.encoder(input=hyp_embeddings, length=hyp_length, return_select_masks=True)
            logits = self.classifier(pre=pre_h, hyp=hyp_h)
            supplements['pre_select_masks'] = pre_select_masks
            supplements['hyp_select_masks'] = hyp_select_masks
        ############################################################################
        elif self.typ == 'tfidf-SA' or self.typ == 'STG-SA':
            pre_h, _, pre_tree = self.encoder(sentence_embedding=pre_embeddings, sentence_word=pre, length=pre_length)
            hyp_h, _, hyp_tree = self.encoder(sentence_embedding=hyp_embeddings, sentence_word=hyp, length=hyp_length)
            logits = self.classifier(pre=pre_h, hyp=hyp_h)
            supplements = {'pre_tree': pre_tree, 'hyp_tree': hyp_tree}
        ############################################################################
        elif self.typ == 'RL-SA':
            pre_h, _, pre_tree, pre_samples = self.encoder(sentence_embedding=pre_embeddings, sentence_word=pre, length=pre_length)
            hyp_h, _, hyp_tree, hyp_samples = self.encoder(sentence_embedding=hyp_embeddings, sentence_word=hyp, length=hyp_length)
            logits = self.classifier(pre=pre_h, hyp=hyp_h)
            supplements = {'pre_tree': pre_tree, 'hyp_tree': hyp_tree}
            # samples prediction for REINFORCE
            sample_logits = self.classifier(pre=pre_samples['h'], hyp=hyp_samples['h'])
            supplements['sample_logits'] = sample_logits
            supplements['pre_probs'] = pre_samples['probs']
            supplements['hyp_probs'] = hyp_samples['probs']
            supplements['pre_sample_trees'] = pre_samples['trees']
            supplements['hyp_sample_trees'] = hyp_samples['trees']
        ############################################################################

        return logits, supplements

