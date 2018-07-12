from torch import nn
from torch.nn import init

class Classifier(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.use_batchnorm = kwargs['use_batchnorm']
        input_dim = kwargs['hidden_dim'] # input of classifier is hidden sentence embedding 
        hidden_dim = kwargs['clf_hidden_dim']

        if self.use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=input_dim)
            self.bn_mlp_output = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(kwargs['dropout'])
        mlp_layers = []
        for i in range(kwargs['clf_num_layers']):
            layer_in_features = hidden_dim if i > 0 else input_dim
            linear_layer = nn.Linear(in_features=layer_in_features,
                                     out_features=hidden_dim)
            mlp_layers.append(linear_layer)
            mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)
        self.clf_linear = nn.Linear(in_features=hidden_dim,
                                    out_features=kwargs['num_classes'])
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_batchnorm:
            self.bn_mlp_input.reset_parameters()
            self.bn_mlp_output.reset_parameters()
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    init.constant_(layer.bias, val=0)
        init.kaiming_normal_(self.clf_linear.weight)
        init.constant_(self.clf_linear.bias, val=0)

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


class SingleModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        model_type = self.model_type = kwargs['model_type']

        if model_type == 'Choi':
            from model.Choi_Treelstm import BinaryTreeLSTM
            Encoder = BinaryTreeLSTM
        elif model_type == 'tfidf-SA':
            from model.tfidf_Tree import tfidfTree
            Encoder = tfidfTree
        elif model_type == 'RL-SA':
            from model.RL_SA_Tree import RlSaTree
            Encoder = RlSaTree
        elif model_type == 'STG-SA':
            from model.STGumbel_AR_Tree import STGumbel_AR_Tree
            Encoder = STGumbel_AR_Tree
        elif model_type == 'SSA':
            from model.Self_Seg_Att_Tree import SelfSegAttenTree
            Encoder = SelfSegAttenTree

        self.word_embedding = nn.Embedding(num_embeddings=kwargs['num_words'],
                                           embedding_dim=kwargs['word_dim'])
        self.encoder = Encoder(**kwargs)
        self.classifier = Classifier(**kwargs)
        self.dropout = nn.Dropout(kwargs['dropout'])
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.word_embedding.weight, mean=0, std=0.01)

    def forward(self, words, length):
        words_embed = self.word_embedding(words)
        words_embed = self.dropout(words_embed)

        ############################################################################
        if self.model_type == 'Choi': 
            h, _, select_masks = self.encoder(input=words_embed, length=length, return_select_masks=True)
            logits = self.classifier(h)
            supplements = {'select_masks': select_masks}
        ############################################################################
        elif self.model_type == 'tfidf-SA' or self.model_type == 'STG-SA' or self.model_type == 'SSA':
            h, _, tree = self.encoder(words_embed, words, length)
            logits = self.classifier(h)
            supplements = {'tree': tree}
        ############################################################################
        elif self.model_type == 'RL-SA':
            h, _, tree, samples = self.encoder(words_embed, words, length)
            logits = self.classifier(h)
            supplements = {'tree': tree}
            # samples prediction for REINFORCE
            sample_logits = self.classifier(samples['h'])
            supplements['sample_logits'] = sample_logits
            supplements['probs'] = samples['probs']
            supplements['sample_trees'] = samples['trees']
            supplements['sample_h'] = samples['h']
        return logits, supplements
