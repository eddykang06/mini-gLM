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