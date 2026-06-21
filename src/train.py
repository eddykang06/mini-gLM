"""Training loop and configuration for MLM objective"""

import pickle
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from src.tokenize import BPETokenizer


class hg38Data(Dataset):
    """
    Custom dataset to load hg38 pre-training data from HuggingFace using trained tokenizer params store in .pkl files.
    """
    def __init__(self, data_path, merge_rules_path, token_map_path):

        # Get sequences from HF path to tokenized dataset
        self.merge_rules_path = merge_rules_path
        self.token_map_path = token_map_path

        with open(self.merge_rules_path, "rb") as f:
            self.merge_rules = pickle.load(self.merge_rules_path)
        with open(self.token_map_path, "rb") as f:
            self.token_map = pickle.load(self.token_map_path)

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
        
        # Include final batch
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
        padded = pad_sequence(
            sequences = batch, 
            batch_first = True, 
            padding_value = self.padding_token
        )
        B, L = padded.shape

        # Generate the attention mask [B, 1, 1, L]
        attn_mask = (padded == self.padding_token).unsqueeze(-2).unsqueeze(-2)

        # Select 15% of tokens in batch
        predict_mask = torch.rand(B, L) < self.predict_prob

        # Get idx and true tokens for each
        predict_idx = torch.nonzero(predict_mask)
        predict_tokens = batch[predict_idx]

        # 80% masked, 10% mutated, 10% unchanged
        mask_mask = 1 < torch.rand(B, L) + predict_mask < 1 + self.masking_prob
        mutate_mask = 1 < torch.rand(B, L) + mask_mask + predict_mask > 1 + self.mutate_prob

        # Convert masked tokens
        converted = padded.copy()
        converted[mask_mask] = self.masking_token

        # Convert mutated tokens
        num_mutated = mutate_mask.sum()
        converted[mutate_mask] = torch.randint(self.vocab_size, (num_mutated,))

        # Strap everything into a clean output as dict (reference in training loop)
        return {
            "batch": converted,
            "predict_idx": predict_idx,
            "predict_tokens": predict_tokens,
            "attention_mask": attn_mask
        }