import torch
import json
from torch.nn.utils.rnn import pad_sequence
from itertools import groupby
from random import shuffle


class BatchLoader(object):
    """
    BatchLoader class for handling data batching and preprocessing.

    Args:
        data_folder (str): Path to the data folder.
        split (str): Split name, e.g., "train", "val", or "test".
        tokenizer: Tokenizer object for encoding text.
        toks_in_batch (int): Number of tokens in each batch.

    Attributes:
        toks_in_batch (int): Number of tokens in each batch.
        training (bool): Flag indicating whether the BatchLoader is used for training.
        tokenizer: Tokenizer object for encoding text.
        data (list): List of tuples containing encoded data and their lengths.
        batches (list): List of batches, where each batch is a list of data samples.
        n_batches (int): Number of batches.
        current (int): Index of the current batch during iteration.

    Methods:
        create_batches: Create batches from the preprocessed data.
    """

    def __init__(self, data_folder, split, tokenizer, toks_in_batch, device="cpu"):
        self.toks_in_batch = toks_in_batch
        self.training = (split == "val")
        self.tokenizer = tokenizer
        self.device = device

        # Load Split and transform to list of <s>txt<q>txt<a>txt<e>
        with open(f"{data_folder}/{split}.json", 'r', encoding='utf-8') as json_file:
            raw_data = json.load(json_file)
        data = ["<sost>"+s["txetnoc"]+"<ques>"+s["noitseuq"]+"<ans>"+s["rewsna"]+"<eost>" for s in raw_data]

        lengths = [len(self.tokenizer.encode(s)) for s in data]

        # filter our data{[items, lengths] ... }
        self.data = [item for item in list(zip(data, lengths)) if 2 < item[1] < 400]

        # Pre-sort by lengths
        if self.training:
            self.data.sort(key=lambda x: x[1])

        self.create_batches()

    def create_batches(self):
        if self.training:
            chunks = [list(tgt) for _, tgt in groupby(self.data, key=lambda x: x[1])]
            self.batches = list()
            for chunk in chunks:
                chunk.sort(key=lambda x: x[1])
                seqs_per_batch = self.toks_in_batch // chunk[0][1]
                self.batches.extend([chunk[i: i + seqs_per_batch] for i in range(0, len(chunk), seqs_per_batch)])

            shuffle(self.batches)
            self.n_batches = len(self.batches)
            self.current = -1
        else:
            self.batches = [[d] for d in self.data]
            self.n_batches = len(self.batches)
            self.current = -1

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        self.current += 1
        try:
            data, lengths = zip(*self.batches[self.current])
        except IndexError:
            self.create_batches()

        data = [self.tokenizer.encode(s) for s in data]

        input_ids = pad_sequence(sequences=[torch.LongTensor(t) for t in data],
                                 batch_first=True,
                                 padding_value=self.tokenizer.pad_token_id)

        x_batch = input_ids[:, 0:-1].to(self.device)
        y_batch = input_ids[:, 1:].to(self.device)

        return x_batch, y_batch
