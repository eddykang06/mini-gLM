"""Training loop and configuration for MLM objective"""

import numpy as np
import pandas as pd
import pickle
import wandb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from typing import Callable
from src.tokenize import BPETokenizer
from src.model import DenseGLM



## Configure training loop for the dense GLM (no auxiliary loss)
## Add mixed precision and flash attention??
def train_dense_glm(
    max_epochs: int,
    lr: float,
    model_params: dict,
    weight_decay: float,
    train_sampler: BatchSampler,
    val_sampler: BatchSampler,
    collate_fn: Callable,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: str,
    wandb_run: wandb,
    log_every: int
):
    
    # Load train and val data
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_sampler = train_sampler,
        collate_fn = collate_fn
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_sampler = val_sampler,
        collate_fn = collate_fn
    )

    # Initialize model
    model = DenseGLM(**model_params)
    model.to(device).float()

    # Muon optimizer
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr = lr, 
        weight_decay = weight_decay)

    # Initialize metrics to store
    num_steps = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    time_to_best_val_loss = 0.0
    tokens_to_best_val_loss = 0.0

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index = -100)

    # Start timer
    run_start_time = time.perf_counter()

    model.train()

    for epoch in range(max_epochs):

        for batch_items in train_loader:
        
            # Store batch items (match with output of collate_fn)
            batch = batch_items["batch"].to(device).long()
            labels = batch_items["labels"].to(device).long()
            predict_mask = batch_items["predict_mask"].to(device)
            attention_mask = batch_items["attention_mask"].to(device)

            # Generate predicted tokens and apply prediction mask to labels
            logits = model(batch, attention_mask)
            labels[~predict_mask] = -100

            # Calculate CE loss
            loss = loss_fn(
                logits.view(-1, logits.shape(-1)),  # [B*L, vocab_size]
                labels.view(-1)                    # [B*L]
            )

            # clear gradient, backprop, update params
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            # Step counter
            num_steps += 1

            # Every x steps, calculate all metrics
            if num_steps % log_every == 0:
                
                # Calculate time elapsed
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                now = time.perf_counter()
                time_elapsed = now - run_start_time
                                
                val_loss = 0.0
                model.eval()

                # Evaluate validation loss
                for val_batch_items in val_loader:

                    with torch.no_grad():

                        # Get batch items
                        val_batch = val_batch_items["batch"].to(device).long()
                        val_labels = val_batch_items["labels"].to(device).long()
                        val_predict_mask = val_batch_items["predict_mask"].to(device)
                        val_attention_mask = val_batch_items["attention_mask"].to(device)

                        # Get logits and CE loss
                        val_logits = model(val_batch, val_attention_mask)
                        val_labels[~val_predict_mask] = -100

                        val_batch_loss = loss_fn(
                            val_logits.view(-1, val_logits.shape(-1)),  # [B*L, vocab_size]
                            val_labels.view(-1)                     # [B*L]
                        )

                        # Store loss
                        val_loss += val_batch_loss.item() * val_logits.size(0)
                
                # Calculate all meaningful metrics
                train_loss = loss.item()
                tokens_seen += attention_mask.sum().item()
                val_loss = val_loss / len(val_dataset)
                
                # update best validation loss if applicable
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    tokens_to_best_val_loss = tokens_seen
                    time_to_best_val_loss = time_elapsed

                # Report metrics
                wandb_run.log({
                    "num_steps": num_steps,
                    "tokens_seen": tokens_seen,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "time_elapsed": time_elapsed
                })

    # Run summary metrics
    wandb_run.summary["best_val_loss"] = best_val_loss
    wandb_run.summary["time_to_best_val_loss"] = time_to_best_val_loss
    wandb_run.summary["tokens_to_best_val_loss"] = tokens_to_best_val_loss
    wandb_run.summary["tokens_per_second"] = tokens_seen / time_elapsed
    wandb_run.summary["steps_per_minute"] = num_steps * 60 / time_elapsed
    wandb_run.summary["total_time"] = time_elapsed

    print("Training complete!")

    return model


