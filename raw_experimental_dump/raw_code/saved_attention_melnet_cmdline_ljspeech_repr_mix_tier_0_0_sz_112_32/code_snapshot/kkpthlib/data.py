import os
import torch
import numpy as np
import re

from collections import Counter

"""
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
"""

def make_batches_from_list(list_of_data, batch_size, sequence_length, overlap=0, cut_points=None,
                           fill_value=None, remove_cuts=True):
    """
    this function truncates ragged batches
    """
    # want contiguity between batch 0, mb index 0 and batch 1, mb index 0, etc
    # easiest way to do this, is take the whole sequence length and divide it into batch_size chunks
    # for example [1 2 3 4 5 6] -> [1 2 3][4 5 6]
    chunk_size = len(list_of_data) // int(batch_size)
    # use the classic zip-splat-iter trick
    # naturally truncates due to zip

    # one above the max should be guaranteed unique...
    if cut_points is not None:
        # TODO: handle non-integer case?
        assert fill_value is not None

    # group into long chunks, total of batch_size
    rs = list(zip(*[iter(list_of_data)] * chunk_size))
    if cut_points is not None:
        # group into long chunks, total of batch_size
        assert len(cut_points) == len(list_of_data)
        cuts = list(zip(*[iter(cut_points)] * chunk_size))

    assert len(rs) == batch_size
    assert overlap < sequence_length
    ro = []
    for j in range(len(rs)):
        r = rs[j]
        rj = []
        for i in range(len(rs[0]) // (sequence_length - overlap)):
            start = i * (sequence_length - overlap)
            stop = start + sequence_length
            if cut_points is not None:
                # tweak so that is starts and stops on a boundary?
                candidates = [s for s in range(start, min(stop, len(cuts[j]))) if cuts[j][s]]
                start = candidates[0]
                stop = candidates[-1]
                slice_values = [sl for sl in np.arange(start, stop)]
                if remove_cuts:
                    # remove the cut values from the actual subsequence - they were only for marking valid boundaries
                    slice_values = [sl for sl in slice_values if sl not in candidates]
                proposed_seq = [r[sli] for sli in slice_values]
                leftover = sequence_length - len(proposed_seq)
                if leftover > 0:
                    # now we are right back to padding / masking again...
                    rj.append(list(proposed_seq) + [fill_value for i in range(leftover)])
                else:
                    rj.append(proposed_seq)
            else:
                rj.append(r[start:stop])
        if len(rj[0]) != len(rj[-1]):
            # cut the last ragged one?
            assert len(rj) > 1
            rj = rj[:-1]
        assert len(rj[0]) == len(rj[-1])
        ro.append(rj)
    assert len(ro[0][0]) == len(ro[-1][-1])
    return np.array(ro).transpose(1, 2, 0)


class LookupDictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word, with_count=None):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]

        if with_count is not None:
            # with_count used when reconstructing index after prune
            self.counter[token_id] = with_count
            self.total += with_count
        else:
            self.counter[token_id] += 1
            self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def _prune_to_top_k_counts(self, count):
        # reserve space for <eos> <unk>
        tup_list = self.counter.most_common(count)
        tup_list = [(self.idx2word[t[0]], t[1]) for t in tup_list]
        old_total = sum(self.counter.values())

        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
        for (w, c) in tup_list:
            if w == tup_list[-1][0]:
                if "<unk>" not in self.word2idx:
                    new_total = sum(self.counter.values())
                    self.add_word("<unk>", with_count=old_total - new_total)
                    continue
            self.add_word(w, with_count=c)


class WordCorpus(object):
    def __init__(self, train_data_file_path, valid_data_file_path=None, test_data_file_path=None,
                 cleaner_fn=None, use_eos=True, max_vocabulary_size=-1):
        """
            def ident(line):
                return line.strip().split() + ['<eos>']
        WordCorpus(cleaner_fn=ident)
        """
        self.dictionary = LookupDictionary()
        if cleaner_fn is None:
            def ident(line):
                l = line.strip().split()
                if use_eos:
                    l = l + ['<eos>']
                return l
            self.cleaner_fn = ident
        elif cleaner_fn == "lower_ascii_keep_standard_punctuation":
            def lap(line):
                clean1 = re.sub(r'[^A-Za-z\?\-\!;,\. ]','', line.lower().strip())
                clean2 = re.findall(r"[\w]+|[^\s\w]", clean1)
                if use_eos:
                    clean2 = clean2 + ["<eos>"]
                return clean2
            self.cleaner_fn = lap
        else:
            self.cleaner_fn = cleaner_fn
        self.max_vocabulary_size = max_vocabulary_size

        base = [train_data_file_path]
        if valid_data_file_path is not None:
            base = base + [valid_data_file_path]
        if test_data_file_path is not None:
            base = base + [test_data_file_path]

        self.build_vocabulary(base)

        if self.max_vocabulary_size > -1:
            self.dictionary._prune_to_top_k_counts(self.max_vocabulary_size)

        self.train = self.tokenize(train_data_file_path)
        if valid_data_file_path is not None:
            self.valid = self.tokenize(valid_data_file_path)
        if test_data_file_path is not None:
            self.test = self.tokenize(test_data_file_path)

    def build_vocabulary(self, file_paths):
        """Tokenizes a text file."""
        for path in file_paths:
            assert os.path.exists(path)
            # Add words to the dictionary
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = self.cleaner_fn(line)
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = self.cleaner_fn(line)
                tokens += len(words)

        # Tokenize file content
        ids = []
        with open(path, 'r') as f:
            for line in f:
                words = self.cleaner_fn(line)
                for word in words:
                    if word in self.dictionary.word2idx:
                        token = self.dictionary.word2idx[word]
                    else:
                        token = self.dictionary.word2idx["<unk>"]
                    ids.append(token)
        return ids
