# mini-gLM: A minimal genomic language model
## [In progress]

## Overview
mini-gLM is a minimal genomic language model trained on sequences from the hg38 human genome assembly. 

## Model description
mini-gLM uses byte-pair encoding (BPE) to tokenize DNA sequences, followed by relative positional encoding using ALiBi. The model follows an encoder-only architecture consisting of sparse transformers with SwiGLU Mixture-of-Experts routing. The pre-training objective was masked token prediction, allowing mini-gLM to capture bidirectional sequence context. 

## Repository structure

## Data
Pre-training data consisted of 1 million sequences of length 500-5000 bp sampled from the 2013 [hg38](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/) human genome assembly, weighted by chromosome length. The annotated pre-training dataset is available on Hugging Face [here](https://huggingface.co/datasets/eddykang06/hg38-pretraining). 

## Training details
Flex attention, dynamic batching, mixed precision (bf16 and fp32), A100 GPU

## Pre-trained weights
Pre-trained model weights are available at...

## Setup

## Fine-tuning
