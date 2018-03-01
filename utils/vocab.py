class Vocab(object):

    def __init__(self, vocab_dict, add_pad, add_unk, id_tf=None, id_df=None):
        self._vocab_dict = vocab_dict
        self._id_tf = id_tf 
        # word_id: word_tf. NOTE: it is total frequency in the whole corpus, can't be used for tf-idf
        self._id_df = id_df
        self._reverse_vocab_dict = dict()
        if add_pad:
            self.pad_word = '<pad>'
            self.pad_id = len(self._vocab_dict)
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = len(self._vocab_dict)
            self._vocab_dict[self.unk_word] = self.unk_id
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    @classmethod
    def from_file(cls, path, add_pad, add_unk, max_size=None):
        vocab_dict = dict()
        id_tf, id_df = dict(), dict()
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_size and i >= max_size:
                    break
                word, tf, df = line.strip().split()
                vocab_dict[word] = i
                id_tf[i] = int(tf)
                id_df[i] = int(df)
        return cls(vocab_dict=vocab_dict, add_pad=add_pad, add_unk=add_unk, id_tf=id_tf, id_df=id_df)

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, id_):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]

    def id_to_tf(self, id_):
        assert self._id_tf is not None, 'vocab has no cnt_dict'
        return self._id_tf.get(id_, 0)

    def id_to_df(self, id_):
        assert self._id_df is not None
        return self._id_df.get(id_, 0)

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return len(self._vocab_dict)
