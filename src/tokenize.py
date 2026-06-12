"""Functions for byte-pair encoding and tokenizer implementation"""

import numpy as np

def find_all_adjacent_pairs(
        token_list: list
) -> dict:
    """
    Given a list of characters, return a dictionary with keys as unique adjacent character tuples and values as the # of times 
    that adjacent pair appears in the list

    Ex: ["A", "C", "G"] -> {("A", "C"): 1, ("C", "G"): 1}
    """
    all_pairs = []

    for i in range(len(token_list) - 1):
        pair = ",".join(token_list[i:i+2])
        all_pairs.append(pair)
        
    unique_pairs, counts = np.unique_counts(all_pairs)
    unique_pair_counts = {tuple(pair.split(",")): int(count) for pair, count in zip(unique_pairs, counts)}

    return unique_pair_counts


def merge_token_pairs(
        token_list: list, 
        token_pair: tuple
) -> list:
    """
    Given a list of tokens and a tuple of token pairs, return a list where all instances of the adjacent token pair have been merged

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
        final_vocab_size: int, 
        seed = None
) -> tuple[list[str], dict[tuple, str]]:
    
    rng = np.random.default_rng(seed)
    merge_rules = {}

    # Pre-tokenization of sequences is done with simple initial vocabulary
    vocab = ["A", "C", "G", "T"]
    split_corpus = [list(seq) for seq in sequence_list]

    while len(vocab) < final_vocab_size:
        
        # Store all pair counts in corpus
        pair_counts_all = {}

        for seq in split_corpus:

            # Find within-sequence counts of all adjacent pairs 
            pair_counts_seq = find_all_adjacent_pairs(seq)

            for pair, count in pair_counts_seq.items():
                pair_counts_all[pair] = pair_counts_all.get(pair, 0) + count

        # Store pairs and counts and find most frequent
        pairs = list(pair_counts_all.keys())
        counts = np.array(list(pair_counts_all.values()))
        best_idx = np.argwhere(counts == counts.max()).ravel()

        # If multiple pairs are found, then randomly select one
        if len(best_idx) > 1:
            random_idx = rng.choice(best_idx)
            best_pair = pairs[random_idx]
        
        else:
            best_idx = best_idx[0]
            best_pair = pairs[best_idx]
        
        # Join all instances of pair in corpus and add token
        split_corpus = [merge_token_pairs(seq, best_pair) for seq in split_corpus]

        # Add new token to vocbaulary
        new_token = "".join(best_pair)
        vocab.append(new_token)

        # Store merge rule
        merge_rule = {best_pair: new_token}
        merge_rules.update(merge_rule)

    return vocab, merge_rules


# Add function to perform tokenization using a learned vocab



# Add class for BPE tokenizer