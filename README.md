# mini-gLM: A lightweight genomic language model
## [In progress]

## Overview
mini-gLM is a lightweight genomic language model trained on sequences from the hg38 human genome assembly. 

## Model description
mini-gLM uses byte-pair encoding (BPE) to tokenize DNA sequences, followed by a sparse attention transformer architecture. The pre-training objective was masked token prediction, allowing mini-gLM to capture bidirectional sequence context. 

## Repo structure

## Data
Pre-training data consisted of x sequences of min_length-max_length bp sampled from the 2013 [hg38](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/) human genome assembly. 

## Pre-trained weights
Pre-trained model weights are available at...

## Setup

## Fine-tuning
