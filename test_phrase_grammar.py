import argparse
import re

import nltk
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm 

import data
import data_ptb
from utils import batchify, get_batch, repackage_hidden, evalb

from data_ptb import word_tags


criterion = nn.CrossEntropyLoss()
def evaluate(data_source, batch_size=1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output = model.decoder(output)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)

def corpus2idx(sentence):
    arr = np.array([data.dictionary.word2idx[c] for c in sentence.split()], dtype=np.int32)
    return torch.from_numpy(arr[:, None]).long()


# Test model
def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


# def build_tree(depth, sen):
#     assert len(depth) == len(sen)
#     assert len(depth) >= 0
#
#     if len(depth) == 1:
#         parse_tree = sen[0]
#     else:
#         idx_max = numpy.argmax(depth[1:]) + 1
#         parse_tree = []
#         if len(sen[:idx_max]) > 0:
#             tree0 = build_tree(depth[:idx_max], sen[:idx_max])
#             parse_tree.append(tree0)
#         if len(sen[idx_max:]) > 0:
#             tree1 = build_tree(depth[idx_max:], sen[idx_max:])
#             parse_tree.append(tree1)
#     return parse_tree


def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1

def MRG(tr):
    if isinstance(tr, str):
        #return '(' + tr + ')'
        return tr + ' '
    else:
        s = '( '
        for subtr in tr:
            s += MRG(subtr)
        s += ') '
        return s

def MRG_labeled(tr):
    if isinstance(tr, nltk.Tree):
        if tr.label() in word_tags:
            return tr.leaves()[0] + ' '
        else:
            s = '(%s ' % (re.split(r'[-=]', tr.label())[0])
            for subtr in tr:
                s += MRG_labeled(subtr)
            s += ') '
            return s
    else:
        return ''

def mean(x):
    return sum(x) / len(x)


def test(model, corpus, info, cuda, evalb_dir):
    model.eval()

    parse_trees = []
    for sen, sen_tree in tqdm(list(zip(*info))):
        x = numpy.array([corpus.dictionary[w] for w in sen])
        input = Variable(torch.LongTensor(x[:, None]))
        if cuda:
            input = input.cuda()

        hidden = model.init_hidden(1)
        _, hidden = model(input, hidden)

        distance = model.distance[0].squeeze().data.cpu().numpy()
        distance_in = model.distance[1].squeeze().data.cpu().numpy()

        sen_cut = sen[1:-1]
        # gates = distance.mean(axis=0)
        for gates in [
            # distance[0],
            distance[1],
            # distance[2],
            # distance.mean(axis=0)
        ]:
            depth = gates[1:-1]
            sen = sen_tree.pos()
            parse_tree = build_tree(depth, sen)
            parse_trees.append(parse_tree)

    return evalb(parse_trees, info[1], evalb_dir)



if __name__ == '__main__':
    marks = [' ', '-', '=']

    numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

    parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

    # Model parameters.
    parser.add_argument('--data', type=str, default='data/ptb',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='PTB.pt',
                        help='model checkpoint to use')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--wsj10', action='store_true',
                        help='use WSJ10')
    args = parser.parse_args()
    args.bptt = 70

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # Load model
    with open(args.checkpoint, 'rb') as f:
        model, _, _ = torch.load(f)
        torch.cuda.manual_seed(args.seed)
        model.cpu()
        if args.cuda:
            model.cuda()

    # Load data
    import hashlib

    fn = 'corpus.{}.data'.format(hashlib.md5('data/penn'.encode()).hexdigest())
    print('Loading cached dataset...')
    corpus = torch.load(fn)
    dictionary = corpus.dictionary

    # test_batch_size = 1
    # test_data = batchify(corpus.test, test_batch_size, args)
    # test_loss = evaluate(test_data, test_batch_size)
    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    #     test_loss, math.exp(test_loss), test_loss / math.log(2)))
    # print('=' * 89)

    print('Loading PTB dataset...')
    corpus = data_ptb.Corpus(args.data)
    corpus.dictionary = dictionary

    test(model, corpus, args.cuda, prt=True)