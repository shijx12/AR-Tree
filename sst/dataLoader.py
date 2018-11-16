from torchtext import data, datasets

class SST(object):
    def __init__(self, args):
        text_field = data.Field(lower=False, include_lengths=True, batch_first=True)
        label_field = data.Field(sequential=False)

        filter_pred = None
        if not args.fine_grained:
            filter_pred = lambda ex: ex.label != 'neutral'
        dataset_splits = datasets.SST.splits(
            root=args.data_path, text_field=text_field, label_field=label_field,
            fine_grained=args.fine_grained, train_subtrees=True,
            filter_pred=filter_pred)
        train_dataset, valid_dataset, test_dataset = dataset_splits
        text_field.build_vocab(*dataset_splits, vectors=args.glove)
        label_field.build_vocab(*dataset_splits)
        self.train_loader, self.valid_loader, self.test_loader = data.BucketIterator.splits(
                datasets=dataset_splits, batch_size=args.batch_size, device=args.device, sort_within_batch=True)

        text_field.vocab.id_to_word = text_field.vocab.itos
        num_classes = len(label_field.vocab)
        print(f'Number of classes: {num_classes}')
        ####### required items
        self.num_train_batches = len(self.train_loader)
        self.num_valid = len(valid_dataset)
        self.num_test = len(test_dataset)
        self.weight = text_field.vocab.vectors
        args.num_classes = num_classes
        args.num_words = len(text_field.vocab)
        args.vocab = text_field.vocab
        #######

    def wrap_to_model_arg(self, words, length): # should match the kwargs of model.forward
        return {
                'words': words,
                'length': length
                }

    def train_minibatch_generator(self):
        for i, batch in enumerate(self.train_loader):
            if i >= self.num_train_batches:
                break
            words, length = batch.text
            label = batch.label
            model_arg = self.wrap_to_model_arg(words, length)
            yield model_arg, label


    def dev_minibatch_generator(self):
        for batch in self.valid_loader:
            words, length = batch.text
            label = batch.label
            model_arg = self.wrap_to_model_arg(words, length)
            yield model_arg, label

    def test_minibatch_generator(self):
        for batch in self.test_loader:
            words, length = batch.text
            label = batch.label
            model_arg = self.wrap_to_model_arg(words, length)
            yield model_arg, label

