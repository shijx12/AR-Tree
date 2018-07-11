from torchtext import data, datasets
from torch.utils.data import DataLoader
from snli.utils.dataset import SNLIDataset
import pickle

class SNLI(object):
    def __init__(self, args):
        with open(args.train_data, 'rb') as f:
            train_dataset: SNLIDataset = pickle.load(f)
        with open(args.valid_data, 'rb') as f:
            valid_dataset: SNLIDataset = pickle.load(f)
        with open(args.test_data, 'rb') as f:
            test_dataset: SNLIDataset = pickle.load(f)

        self.train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2,
                                  collate_fn=train_dataset.collate,
                                  pin_memory=True)
        self.valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=2,
                                  collate_fn=valid_dataset.collate,
                                  pin_memory=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=2,
                                  collate_fn=valid_dataset.collate,
                                  pin_memory=True)
        self.num_train_batches = len(train_loader)

        num_classes = len(train_dataset.label_vocab)
        print(f'Number of classes: {num_classes}')
        args.num_classes = num_classes
        args.num_words = len(train_dataset.word_vocab)
        args.vocab = train_dataset.word_vocab

    def wrap_to_model_arg(self, pre, hyp, pre_length, hyp_length): # should match the kwargs of model.forward
        return {
                'pre': pre,
                'hyp': hyp,
                'pre_length': pre_length,
                'hyp_length': hyp_length,
                }

    def train_minibatch_generator(self):
        for i, batch in enumerate(self.train_loader):
            label = batch['label']
            batch.pop('label')
            model_arg = batch
            yield model_arg, label


    def dev_minibatch_generator(self):
        for batch in self.valid_loader:
            label = batch['label']
            batch.pop('label')
            model_arg = batch
            yield model_arg, label

    def test_minibatch_generator(self):
        for batch in self.test_loader:
            label = batch['label']
            batch.pop('label')
            model_arg = batch
            yield model_arg, label

