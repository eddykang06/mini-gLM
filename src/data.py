"""Configuring/loading data"""

import numpy as np
import pandas as pd
import subprocess
import sys
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from src.tokenize import BPETokenizer
from pyfaidx import Fasta
from pathlib import Path


def sample_from_fasta(
        data_dir: Path,
        out_dir: Path,
        n_seqs: int,
        min_length: int,
        max_length: int,
        seed: int | None = None,
) -> pd.DataFrame:
    """
    Function to sample random sequences of a specified length range from .fa file, weighted by chromsome length
    Final output is written to csv.

    Args:
        data_dir        : Path to dir containing data files
        out_dir         : Path to write final csv to
        n_seqs          : Number of sequences to sample
        min_length      : Minimum sequence length to sample
        max_length      : Maximum sequence length to sample
        seed            : Random seed for numpy
    
    Returns:
        csv : Dataframe of sampled sequences with start, end, length, and chromosome ID
    """
    # Store paths to data
    fasta_file = data_dir / "hg38.fa"
    chrom_size_file = data_dir / "hg38.chrom.sizes"

    # Set random seed if specified
    rng = np.random.default_rng(seed)

    # Store file using pyfaidx
    f = Fasta(fasta_file)

    # Load the chromosome and sort
    size_df = pd.read_table(chrom_size_file, sep = "\t", index_col = 0, header = None)
    size_df.columns = ["length"]
    size_df["length"] = size_df["length"].astype(int)
    size_df = size_df.iloc[size_df.index.argsort()]

    # Check that the fasta chromsomes match the size df
    index = sorted(size_df.index.to_numpy())
    chromosomes = sorted(np.array(list(f.keys())))

    if index != chromosomes:
        raise KeyError("Fasta chromosomes IDs and length data IDs are incompatible")

    # Normalize to obtain sampling weights for each chromosome
    weights = size_df["length"].to_numpy() / size_df["length"].to_numpy().sum()

    # Initiate list to store sequences and metadata
    seqs = []
    chroms = []
    starts = []
    ends = []
    lengths = []

    # Counter so that rejected sequences don't count toward total count
    counter = 0

    # Number of sequences to sample
    while counter < n_seqs:

        # Weighted sampling of chromosome
        chrom = rng.choice(chromosomes, p = weights)
        length = int(size_df.loc[chrom, "length"])

        # Randomly sample a sequence
        seq_length = rng.integers(min_length, min(length, max_length) + 1)
        start_idx = rng.integers(0, length - seq_length + 1)
        seq = f[chrom][start_idx:start_idx + seq_length]
        seq = str(seq).upper()

        # Split across "N" character if exists and take first substr
        seq = seq.split("N")[0]

        # Convert to uppercase, remove non-alpha and non-ACGT characters 
        seq = "".join(char for char in seq if char in "ACGT")
        new_length = len(seq)

        # Check that sequences is still above minimum length
        if new_length >= min_length:
            counter += 1 

            # Append sequence and all metadata
            seqs.append(seq)
            chroms.append(chrom)
            starts.append(start_idx)
            ends.append(start_idx + seq_length)
            lengths.append(new_length)

    # Construct dataframe
    df = pd.DataFrame({
        "sequence": seqs,
        "chromosome_id": chroms,
        "start": starts,
        "end": ends,
        "length": lengths
    })

    # Write to csv
    out_file = out_dir / "pretraining.csv.gz"
    df.to_csv(out_file, compression = "gzip", index = False)

    return df


class hg38Data(Dataset):
    """
    Custom dataset to load hg38 pre-training data from HuggingFace using trained tokenizer params store in .pkl files.
    """
    def __init__(self, data_path, merge_rules_path, token_map_path):

        # Get sequences from HF path to tokenized dataset
        self.merge_rules_path = merge_rules_path
        self.token_map_path = token_map_path

        with open(self.merge_rules_path, "rb") as f:
            self.merge_rules = pickle.load(f)
        with open(self.token_map_path, "rb") as f:
            self.token_map = pickle.load(f)

        # Get tokenizer learned params and load them into the tokenizer
        self.tokenizer = BPETokenizer(
            merge_rules = self.merge_rules,
            token_to_idx = self.token_map
        )

        # Load data from HF and tokenize
        self.sequence_list = pd.read_csv(data_path, compression = "gzip", usecols = ["sequence"])["sequence"].to_list()
        self.tokenized_list = self.tokenizer.tokenize(self.sequence_list).sort(key = len)

    def __len__(self):
        return len(self.tokenized_list)

    def __getitem__(self, idx):
        tokenized = self.tokenized_list[idx]
        tokenized = torch.tensor(tokenized)

        return tokenized


class DynamicBatchSampler(BatchSampler):
    """
    Dynamic batching. For each batch, we set a constant batch_space, meaning the length of the longest sequence * # of sequences is constant per batch.
    This way, each batch contains a different dimension, but an approximately equal amount of tokens. This allows us to save attention compute cost by 
    grouping similar length sequences together, reducing the # of padding tokens needed. We also shuffle the order in which the batches are fed 
    to the DataLoader to minimize sequence length bias when training.
    
    Note: this implementation also assumes that the tokenized sequence data is already sorted from shortest to longest.
    """

    def __init__(self, dataset, batch_space):
        self.batch_space = batch_space
        self.dataset = dataset
        self.dataset_size = len(self.dataset)
    
    def __iter__(self):
        batches = []
        batch = []

        for i in range(self.dataset_size):
            batch.append(i)

            if len(self.dataset[i]) * len(batch) >= self.batch_space:
                batches.append(batch)
                batch = []
        
        if batch:
            batches.append(batch)
        
        np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        num_batches = 0
        batch_size = 0
    
        # Iterate through same logic as the iter function
        for i in range(self.dataset_size):
            batch_size += 1

            if len(self.dataset[i]) * batch_size >= self.batch_space:
                num_batches += 1
                batch_size = 0
            
        if batch_size > 0:
            num_batches += 1

        return num_batches


class MLMCollator():
    """
    Collator with right padding within batch, attention mask generation, and BERT-style training token selection.
    """
    def __init__(self, vocab_size, predict_prob, masking_prob, randomize_prob):
        self.vocab_size = vocab_size
        self.padding_token = int(self.vocab_size + 1)
        self.masking_token = int(self.vocab_size + 2)
        self.predict_prob = predict_prob
        self.masking_prob = masking_prob
        self.mutate_prob = randomize_prob
    
    def __call__(self, batch):

        # Right padding to [B, L_max], where L_max is the length of the longest sequence in the batch
        labels = pad_sequence(
            sequences = batch, 
            batch_first = True, 
            padding_value = self.padding_token
        )
        B, L = labels.shape

        # Generate the attention mask [B, 1, 1, L]
        attn_mask = labels == self.padding_token
        attn_mask_reshaped = attn_mask.unsqueeze(-2).unsqueeze(-2)

        # Select 15% of tokens in batch (not including padding tokens)
        predict_mask = 1 < torch.rand(B, L) + attn_mask < 1 + self.predict_prob

        # 80% masked, 10% mutated, 10% unchanged
        mask_mask = 1 < torch.rand(B, L) + predict_mask < 1 + self.masking_prob
        mutate_mask = 1 < torch.rand(B, L) + mask_mask + predict_mask < 1 + self.mutate_prob

        # Convert masked tokens
        converted = labels.copy()
        converted[mask_mask] = self.masking_token

        # Convert mutated tokens
        num_mutated = mutate_mask.sum()
        converted[mutate_mask] = torch.randint(self.vocab_size, (num_mutated,))

        # Strap everything into a clean output as dict (reference in training loop)
        return {
            "batch": converted,
            "labels": labels,
            "predict_mask": predict_mask,
            "attention_mask": attn_mask_reshaped
        }
