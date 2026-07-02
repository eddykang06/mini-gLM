# mini-gLM: A minimal genomic language model

## Overview
mini-gLM is a minimal genomic language model trained on sequences from the hg38 human genome assembly. mini-gLM emphasizes practical model development: a compact architecture, efficient training, and reproducible sequence modeling with minimal compute.

## Model description
mini-gLM is an encoder-only language model trained using a bidirectional masked langauge modeling objective. Input DNA sequences were tokenized using byte-pair encoding (BPE), followed by relative positional encoding using ALiBi. The model utilizes a Mixture-of-Experts architecture with SwiGLU experts, along with alternating sparse and dense attention mechanisms.

## Repository structure
```text
mini-gLM/
├── configs/            # Experiment and data-loading configs
├── notebooks/          # Tokenization and training exploration
└── src/                
    ├── data.py         # Sequence sampling, datasets, batching, MLM masking
    ├── tokenize.py     # DNA byte-pair encoding tokenization
    ├── transformer.py  # Custom FlexAttention configuration, ALiBi, SwiGLU, MoE transformer blocks
    ├── model.py        # Dense and MoE architectures
    ├── train.py        # Training + validation loop
    └── finetune.py     # Fine-tuning scaffold
```
## Data
Pre-training data consisted of 1 million sequences of length 500-5000 bp sampled from the 2013 [hg38](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/) human genome assembly. The annotated pre-training dataset is available on Hugging Face [here](https://huggingface.co/datasets/eddykang06/hg38-pretraining). 

## Training details
Training scheme and engineering highlights:
- FlexAttention for efficient, flexible attention map computation
- Dynamic batching for consistent token count per batch and minimal padding token usage
- Mixed precision training (bf16, fp32)
- A100 GPU through Google Cloud

## Pre-trained weights
Coming soon!

## Requirements and setup
Coming soon!

## Fine-tuning
Coming soon!
