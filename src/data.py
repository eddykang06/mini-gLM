"""Configuring/loading data"""

import numpy as np
import pandas as pd
import json, yaml
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from pyfaidx import Fasta
from pathlib import Path
from huggingface_hub import hf_hub_download


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


class TokenizedDataset(Dataset):
    """
    Custom dataset to load and tokenize hg38 pre-training data from HuggingFace using trained tokenizer params files.
    """
    def __init__(
        self, 
        data_path: Path, 
    ):
        # Get tokenized data from path
        self.data_path = data_path

        columns = ["tokenized", "tokenized_length"]
        df = pd.read_parquet(
            data_path,
            columns = columns
        )
        df = df.sort_values("tokenized_length").reset_index(drop = True)
        self.tokenized_list = df["tokenized"].to_list()

    def __len__(self):
        return len(self.tokenized_list)

    def __getitem__(self, idx):
        tokenized = self.tokenized_list[idx]
        tokenized = np.fromstring(
            tokenized.strip("[]"),
            sep = " ",
            dtype = np.int64
        )
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

    def __init__(
        self, 
        dataset: Dataset, 
        attention_space: int
    ):
        self.attention_space = attention_space
        self.dataset = dataset
        self.dataset_size = len(self.dataset)
    
    def __iter__(self):
        batches = []
        batch = []

        for i in range(self.dataset_size):
            batch.append(i)

            if len(self.dataset[i]) * len(batch)**2 >= self.attention_space:
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

            if len(self.dataset[i]) * batch_size**2 >= self.attention_space:
                num_batches += 1
                batch_size = 0
            
        if batch_size > 0:
            num_batches += 1

        return num_batches


class MLMCollator:
    """
    Collator with right padding within batch, attention mask generation, and BERT-style training token selection.
    """
    def __init__(
        self, 
        vocab_size: int, 
        predict_prob: float, 
        masking_prob: float, 
        mutate_prob: float
    ):
        self.vocab_size = vocab_size
        self.padding_token = vocab_size
        self.masking_token = vocab_size + 1

        self.predict_prob = predict_prob
        self.masking_prob = masking_prob
        self.mutate_prob = mutate_prob

        if masking_prob + mutate_prob > 1.0:
            raise ValueError("masking_prob + randomize_prob must be <= 1.0")
    
    def __call__(self, batch):

        # Right padding to [B, L_max], where L_max is the length of the longest sequence in the batch
        labels = pad_sequence(
            sequences = batch, 
            batch_first = True, 
            padding_value = self.padding_token
        ).long()

        B, L = labels.shape
        device = labels.device

        # Generate the attention mask [B, L]
        attention_mask = labels != self.padding_token

        # Select prediction targets from real tokens
        predict_mask = (torch.rand(B, L, device = device) < self.predict_prob) & attention_mask

        # Select tokens to be masked and mutated
        corruption_rand = torch.rand(B, L, device=device)
        mask_mask = predict_mask & (corruption_rand < self.masking_prob)

        mutate_mask = (
            predict_mask
            & (corruption_rand >= self.masking_prob)
            & (corruption_rand < self.masking_prob + self.mutate_prob)
        )

        # Convert masked tokens
        converted = labels.clone()
        converted[mask_mask] = self.masking_token

        # Convert mutated tokens
        num_mutated = mutate_mask.sum().item()

        if num_mutated > 0:
            converted[mutate_mask] = torch.randint(
                low = 0,
                high = self.vocab_size, 
                size = (num_mutated,),
                device = device,
                dtype = torch.long
            )

        # Strap everything into a clean output as dict (reference in training loop)
        return {
            "batch": converted,
            "labels": labels,
            "predict_mask": predict_mask,
            "attention_mask": attention_mask
        }


def get_pretraining_data(root: Path):
    """
    Get train and val data from HuggingFace dataset and load as torch datasets

    Args:
        root : Path to root directory

    Returns:
        train_dataset : Training data converted to Dataset
        val_dataset   : Validation data convert to Dataset
    
    """
    config_path = Path(root / "configs" / "data_loader.yaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["pretraining"]
    
    repo_id = cfg["repo_id"]
    repo_type = cfg["repo_type"]
    train_folder = cfg["train_folder"]
    val_folder = cfg["val_folder"]
    train_file = cfg["train_file"]
    val_file = cfg["val_file"]

    train_path = hf_hub_download(
        repo_id = repo_id,
        repo_type = repo_type,
        subfolder = train_folder,
        filename = train_file
    )
    val_path = hf_hub_download(
        repo_id = repo_id,
        repo_type = repo_type,
        subfolder = val_folder,
        filename = val_file
    )

    train_dataset = TokenizedDataset(
        data_path = train_path
    )
    val_dataset = TokenizedDataset(
        data_path = val_path
    )

    return train_dataset, val_dataset