"""Functions for byte-pair encoding and tokenizer implementation"""

import numpy as np
from collections import Counter


def merge_token_pairs(
        token_list: list, 
        token_pair: tuple
) -> list:
    """
    Given a list of tokens and a token pair tuple, return a list where all instances of the adjacent token pair have been merged

    Ex: ["A", "C", "G", "T"], ("A", "C") -> ["AC", "G", "T"]
    """
    merged_list = []
    counter = 0

    while counter < len(token_list):
        pair = tuple(token_list[counter:counter+2])
        if pair == token_pair:
            merged = "".join(pair)
            merged_list.append(merged)
            counter += 2
        else:
            old_token = pair[0]
            merged_list.append(old_token)
            counter += 1

    return merged_list


def train_bpe_tokenizer(
        sequence_list: list[str], 
        final_vocab_size: int
) -> tuple[list[str], dict[tuple, str]]:
    """
    Train a BPE tokenizer on a list of sequences with a specified final vocab size

    Args:
        sequence_list    : List of sequences as strings 
        final_vocab_size : Desired final vocab size
        
    Returns:
        vocab       : List of tokens
        merge_rules : Dictionary mapping token pairs to new merged tokens in training order
    """
    
    merge_rules = {}

    # Pre-tokenization of sequences is done with simple initial vocabulary
    vocab = ["A", "C", "G", "T"]
    split_corpus = [list(seq) for seq in sequence_list]

    while len(vocab) < final_vocab_size:
        
        # Store all pair counts in corpus
        pair_counts = Counter()

        for seq in split_corpus:
             for a, b in zip(seq, seq[1:])
                pair_counts([a, b]) += 1

        # Store token pairs + counts and find most frequent adjacent tokens
        best_pair, _ = pair_counts.most_common(1)[0]
        
        # Join all instances of pair in corpus and add token
        split_corpus = [merge_token_pairs(seq, best_pair) for seq in split_corpus]

        # Add new token to vocbaulary
        new_token = "".join(best_pair)
        vocab.append(new_token)

        # Store merge rule
        merge_rule = {best_pair: new_token}
        merge_rules.update(merge_rule)

    return vocab, merge_rules


def tokenize_sequences(
    sequence_list : list[str],
    merge_rules: dict[tuple: str],
    token_to_idx: dict[str: int]
) -> list[list]:  
    """
    Convert a list of sequences to tokens using specified merge rules and token-ID mapping
    
    Args:
        sequence_list : List of sequnces as string
        merge_rules   : Dictionary mapping token tuples to new merged tokens
        token_to_idx  : Dictionary mapping tokens to integer IDs
    
    Returns:
        tokenized_sequences : List of tokenized sequences as lists of integer token IDs
    """
    tokenized_sequences = []
    for seq in sequence_list:
        
        # Separate sequence into individual strings
        seq = list(seq)

        # Apply learned merge rules in order
        for pair in list(merge_rules.keys()):
            seq = merge_token_pairs(seq, pair)            

        # Map to token IDs
        tokenized = [token_to_idx[token] for token in seq]
        tokenized_sequences.append(tokenized)
    
    return tokenized_sequences


class BPETokenizer():
    """
    Tokenizer class to enable tokenizer training and tokenization of unseen sequences
    """
    def __init__(self, vocab = None, merge_rules = None, token_to_idx = None,):
        self.vocab = vocab if vocab is not None else []
        self.vocab_size = len(self.vocab)
        self.merge_rules = merge_rules if merge_rules is not None else {}
        self.token_to_idx = token_to_idx if token_to_idx is not None else {}
        self.idx_to_token = {id: token for token, id in self.token_to_idx.items()}
    
    # Train a tokenizer on a given corpus of DNA sequences
    def train(
        self, 
        sequence_list: list[str], 
        final_vocab_size: int
    ):

        vocab, merge_rules = train_bpe_tokenizer(
            sequence_list = sequence_list,
            final_vocab_size = final_vocab_size,
        )

        # Create vector of token IDs
        idx = list(range(len(vocab)))

        # Update vocab, merge rules, and token ID mapping
        self.vocab = vocab
        self.merge_rules = merge_rules
        self.token_to_idx = {token: id for token, id in zip(vocab, idx)}
    
    # Tokenize a new list of sequences into a token IDs
    def tokenize(
        self, 
        sequence_list: list[str]
    ) -> list[list]:

        tokenized = tokenize_sequences(
            sequence_list = sequence_list,
            merge_rules = self.merge_rules,
            token_to_idx = self.token_to_idx
        ) 

        return tokenized

    # Train a tokenizer and apply the tokenizer to the same corpus
    def train_tokenize(
        self, 
        sequence_list: list[str], 
        final_vocab_size: int
    ) -> list[list]:
        
        vocab, merge_rules = train_bpe_tokenizer(
            sequence_list = sequence_list,
            final_vocab_size = final_vocab_size,
        )

        # Create vector of token IDs
        idx = list(range(len(vocab)))

        # Update vocab, merge rules, and token ID mapping
        self.vocab = vocab
        self.merge_rules = merge_rules
        self.token_to_idx = {token: id for token, id in zip(vocab, idx)}

        # Tokenize the same sequences
        tokenized = tokenize_sequences(
            sequence_list = sequence_list,
            merge_rules = self.merge_rules,
            token_to_idx = self.token_to_idx
        ) 

        return tokenized