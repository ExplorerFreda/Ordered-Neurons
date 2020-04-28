import os
import subprocess
import torch
import tempfile
import re
import nltk

punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']


def list2tree(node):
    if isinstance(node, list):
        tree = []
        for child in node:
            tree.append(list2tree(child))
        return nltk.Tree('NT', tree)
    elif isinstance(node, tuple):
        return nltk.Tree(node[1], [node[0]])


def process_tree(node):  # remove punct from tree
    if isinstance(node, str):
        return node
    label = node.label()
    if label in punctuation_tags:
        return None
    children = list()
    for child in node:
        proc_child = process_tree(child)
        if proc_child is not None:
            children.append(proc_child)
    if len(children) > 0:
        return nltk.Tree(label, children)
    else:
        return None


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def make_batch(source):
    targets = source[:, 1:].t().contiguous()
    source = source[:, :-1].t().contiguous()
    if torch.cuda.is_available():
        targets = targets.cuda()
        source = source.cuda()
    return source, targets, source.size(0)


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def load_embeddings_txt(path):
  words = pd.read_csv(path, sep=" ", index_col=0,
                      na_values=None, keep_default_na=False, header=None,
                      quoting=csv.QUOTE_NONE)
  matrix = words.values
  index_to_word = list(words.index)
  word_to_index = {
    word: ind for ind, word in enumerate(index_to_word)
  }
  return matrix, word_to_index, index_to_word


def evalb(pred_tree_list, targ_tree_list, evalb_dir='./EVALB'):
    temp_path = tempfile.TemporaryDirectory(prefix="evalb-")
    temp_file_path = os.path.join(temp_path.name, "pred_trees.txt")
    temp_targ_path = os.path.join(temp_path.name, "true_trees.txt")
    temp_eval_path = os.path.join(temp_path.name, "evals.txt")

    temp_tree_file = open(temp_file_path, "w")
    temp_targ_file = open(temp_targ_path, "w")

    for pred_tree, targ_tree in zip(pred_tree_list, targ_tree_list):
        temp_tree_file.write(re.sub('[ |\n]+', ' ', str(process_tree(list2tree(pred_tree)))) + '\n')
        temp_targ_file.write(re.sub('[ |\n]+', ' ', str(process_tree(targ_tree))) + '\n')
    
    temp_tree_file.close()
    temp_targ_file.close()

    evalb_param_path = os.path.join(evalb_dir, "fhs.prm")
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        temp_targ_path,
        temp_file_path,
        temp_eval_path)

    subprocess.run(command, shell=True)

    with open(temp_eval_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                evalb_fscore = float(match.group(1))
                break

    temp_path.cleanup()

    return evalb_fscore
