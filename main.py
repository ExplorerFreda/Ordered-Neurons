import argparse
import time
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset

import data
import data_ptb
import model
from utils import make_batch
from test_phrase_grammar import test


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--chunk_size', type=int, default=10,
                    help='number of units per chunk')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.5,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.45,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=4321,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--finetuning', type=int, default=100,
                    help='When (which epochs) to switch to finetuning')
parser.add_argument('--philly', action='store_true',
                    help='Use philly cluster')
parser.add_argument('--evalb-dir', type=str, default='../EVALB/')
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# construct loggers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)
handler = logging.FileHandler(args.save.replace('.pt', '.log'), 'w')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)


import os
import hashlib

corpus = data.Corpus(args.data)
tree_corpus = data_ptb.Corpus(os.path.join(args.data, 'trees'))

eval_batch_size = 10
test_batch_size = 1

train_loader = DataLoader(corpus.train, args.batch_size, shuffle=True, collate_fn=data.collate_fn)
val_loader = DataLoader(corpus.valid, args.batch_size, shuffle=False, collate_fn=data.collate_fn)
test_loader = DataLoader(corpus.test, args.batch_size, shuffle=False, collate_fn=data.collate_fn)


###############################################################################
# Build the model
###############################################################################

criterion = None

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.chunk_size, args.nlayers,
                       args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        for rnn in model.rnn.cells:
            rnn.hh.dropout = args.wdrop
###
if not criterion:
    criterion = nn.CrossEntropyLoss(ignore_index=corpus.dictionary.pad_id)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


###############################################################################
# Training code
###############################################################################

def evaluate(data_loader, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        if args.model == 'QRNN':
            model.reset()
        total_loss = 0
        total_items = 0
        for i, batch in enumerate(data_loader):
            batch, targets, lengths = make_batch(batch)
            hidden = model.init_hidden(batch.size(1))
            output, hidden, rnn_hs, dropped_rnn_hs = model(batch, hidden, return_h=True)
            decoded = model.decoder(output)
            total_loss += criterion(decoded, targets.view(-1)) * batch.size(1)
            total_items += batch.size(1)
        return total_loss.item() / total_items

def train():
    model.train()
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN':
        model.reset()
    total_loss = 0
    total_items = 0
    start_time = time.time()
    for i, batch in enumerate(train_loader):
        batch, targets, lengths = make_batch(batch)
        hidden = model.init_hidden(batch.size(1))
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(batch, hidden, return_h=True)
        # output, hidden = model(data, hidden, return_h=False)
        decoded = model.decoder(output)
        raw_loss = criterion(decoded, targets.view(-1))

        loss = raw_loss
        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(
                args.alpha * dropped_rnn_h.pow(2).mean()
                for dropped_rnn_h in dropped_rnn_hs[-1:]
            )
        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + sum(
                args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                for rnn_h in rnn_hs[-1:]
            )
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data * batch.size(1)
        total_items += batch.size(1)
        if (i + 1) % args.log_interval == 0:
            cur_loss = total_loss.item() / total_items
            elapsed = time.time() - start_time
            logger.info(
                '| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                    epoch, i + 1, len(train_loader), optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2))
            )
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
        scheduler = lr_scheduler.StepLR(optimizer, 50, 0.5)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        dev_f1 = test(model, corpus, (tree_corpus.valid_sens, tree_corpus.valid_trees), args.cuda, args.evalb_dir)
        test_f1 = test(model, corpus, (tree_corpus.test_sens, tree_corpus.test_trees), args.cuda, args.evalb_dir)
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_loader, args.batch_size)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f} | dev f1 {:8.3f} | test f1 {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2),
                dev_f1, test_f1)
            )
            logger.info('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save)
                logger.info('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

            if epoch == args.finetuning:
                logger.info('Switching to finetuning')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                best_val_loss = []

            if epoch > args.finetuning and len(best_val_loss) > args.nonmono and val_loss2 > min(
                    best_val_loss[:-args.nonmono]):
                logger.info('Done!')
                import sys

                sys.exit(1)

        else:
            val_loss = evaluate(val_loader, args.batch_size)
            logger.info('-' * 89)
            logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f} | dev f1 {:8.3f} | test f1 {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2),
                dev_f1, test_f1)
            )
            logger.info('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                logger.info('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'adam':
                scheduler.step(val_loss)

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                    len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                logger.info('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                logger.info('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                logger.info('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)
        for few_num in [15, 25, 55, 105]:
            logger.info('few-shot num: {:d}'.format(few_num))
            for idx in range(0, few_num*5, few_num):
                dev_f1 = test(model, corpus, (tree_corpus.valid_sens[idx:idx+few_num], tree_corpus.valid_trees[idx:idx+few_num]), args.cuda, args.evalb_dir)
                logger.info('| few-shot dev f1 {:5.2f} |'.format(dev_f1))
        logger.info("PROGRESS: {}%".format((epoch / args.epochs) * 100))

except KeyboardInterrupt:
    logger.info('-' * 89)
    logger.info('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_loader, args.batch_size)
logger.info('=' * 89)
logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
logger.info('=' * 89)
