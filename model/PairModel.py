import torch
from torch import nn
from torch.nn import init

class Classifier(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.use_batchnorm = kwargs['use_batchnorm']
        input_dim = kwargs['hidden_dim'] # input of classifier is hidden sentence embedding 
        hidden_dim = kwargs['clf_hidden_dim']

        if self.use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=4 * input_dim)
            self.bn_mlp_output = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(kwargs['dropout'])
        mlp_layers = []
        for i in range(kwargs['clf_num_layers']):
            layer_in_features = hidden_dim if i > 0 else 4 * input_dim
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
        init.normal_(self.clf_linear.weight.data, std=0.01)
        init.constant_(self.clf_linear.bias.data, val=0)

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


class PairModel(nn.Module):

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
        init.normal_(self.word_embedding.weight.data, mean=0, std=0.01)

    def forward(self, pre, pre_length, hyp, hyp_length):
        pre_embeddings = self.word_embedding(pre)
        hyp_embeddings = self.word_embedding(hyp)
        pre_embeddings = self.dropout(pre_embeddings)
        hyp_embeddings = self.dropout(hyp_embeddings)

        ############################################################################
        if self.model_type == 'Choi': 
            supplements = {}
            pre_h, _, pre_select_masks = self.encoder(input=pre_embeddings, length=pre_length, return_select_masks=True)
            hyp_h, _, hyp_select_masks = self.encoder(input=hyp_embeddings, length=hyp_length, return_select_masks=True)
            logits = self.classifier(pre=pre_h, hyp=hyp_h)
            supplements['pre_select_masks'] = pre_select_masks
            supplements['hyp_select_masks'] = hyp_select_masks
        ############################################################################
        elif self.model_type == 'tfidf-SA' or self.model_type == 'STG-SA':
            pre_h, _, pre_tree = self.encoder(sentence_embedding=pre_embeddings, sentence_word=pre, length=pre_length)
            hyp_h, _, hyp_tree = self.encoder(sentence_embedding=hyp_embeddings, sentence_word=hyp, length=hyp_length)
            logits = self.classifier(pre=pre_h, hyp=hyp_h)
            supplements = {'pre_tree': pre_tree, 'hyp_tree': hyp_tree}
        ############################################################################
        elif self.model_type == 'RL-SA':
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

