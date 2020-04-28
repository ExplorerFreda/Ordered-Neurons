import os
import pickle 
import re
import torch
from torch.utils.data import Dataset

from collections import Counter


class Dictionary(object):
    def __init__(self, add_pad=False):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
        if add_pad:
            self.pad_id = self.add_word('[PAD]')
        else:
            self.pad_id = None

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __getitem__(self, word):
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class SentDataset(Dataset):
    def __init__(self, ids):
        super(SentDataset, self).__init__()
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return self.ids[item]


class Corpus(object):
    def __init__(self, path):
        dict_path = os.path.join(path, 'dict.pkl')
        corpus_path = os.path.join(path, 'corpus.pt')
        if os.path.exists(corpus_path):
            self.dictionary, self.train, self.train_sens, self.valid, self.valid_sens, self.test, self.test_sens = \
                torch.load(corpus_path)
        else:
            self.dictionary = Dictionary(add_pad=True)
            self.train, self.train_sens = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid, self.valid_sens = self.tokenize(os.path.join(path, 'dev.txt'))
            self.test, self.test_sens = self.tokenize(os.path.join(path, 'test.txt'))
            torch.save(
                (self.dictionary, self.train, self.train_sens, self.valid, self.valid_sens, self.test, self.test_sens), 
                corpus_path
            )
            pickle.dump(self.dictionary, open(dict_path, 'wb'))
            if os.path.exists(os.path.join(path, 'trees')):
                os.system('ln -s ../dict.pkl {}'.format(os.path.join(path, 'trees', 'dict.pkl')))

    def filter_words(self, orig_words):
        words = []
        for w in orig_words:
            w = w.lower()
            w = re.sub('[0-9]+', 'N', w)
            words.append(w)
        return words

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        sents = list()
        # Add words to the dictionary
        with open(path, 'r') as f:
            for line in f:
                line = self.filter_words(line.lower().strip().split())
                words = ['[CLS]'] + line + ['[SEP]']
                for word in words:
                    self.dictionary.add_word(word)
                sents.append(words)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = list()
            for line in f:
                line = self.filter_words(line.lower().strip().split())
                words = ['[CLS]'] + line + ['[SEP]']
                ids.append([self.dictionary.word2idx[word] for word in words])

        return SentDataset(ids), sents


def collate_fn(data):  # pad_id = 0
    data.sort(key=lambda x: len(x), reverse=True)
    max_length = max(len(item) for item in data)
    for i, item in enumerate(data):
        data[i] = item + [0] * (max_length - len(item))
    return torch.tensor(data).long()
