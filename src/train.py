"""Training loop and configuration for MLM objective"""

import wandb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, BatchSampler
from typing import Callable
from src.model import DenseGLM


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
    val_every: int,
):
    """
    Train a gLM with dense attention and standard transformer blocks
    """
    
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
    tokens_to_best_val_loss = 0

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index = -100)
    loss_fn_sum = nn.CrossEntropyLoss(ignore_index = -100, reduction = "sum")

    # Start timer
    run_start_time = time.perf_counter()

    model.train()

    for epoch in range(max_epochs):

        for batch_items in train_loader:
        
            # Store batch items (match with output of collate_fn)
            batch = batch_items["batch"].to(device).long()
            labels = batch_items["labels"].to(device).long().clone()
            predict_mask = batch_items["predict_mask"].to(device).bool()
            attention_mask = batch_items["attention_mask"].to(device).bool()

            # Generate predicted tokens and apply prediction mask to labels
            logits = model(batch, attention_mask)
            labels[~predict_mask] = -100

            # Calculate CE loss
            loss = loss_fn(
                logits.reshape(-1, logits.shape(-1)),  # [B*L, vocab_size]
                labels.reshape(-1)                    # [B*L]
            )

            # clear gradient, backprop, update params
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            # Calculate per step metrics
            num_steps += 1
            train_loss = loss.item()
            tokens_seen += attention_mask.sum().item()

            # Calculate time elapsed
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            time_elapsed = now - run_start_time

            # Report metrics
            wandb_run.log({
                "epoch": epoch,
                "tokens_seen": tokens_seen,
                "train_loss": train_loss,
                "time_elapsed": time_elapsed
            },
            step = num_steps)

            # Log validation metrics in specified intervals
            if num_steps % val_every == 0:
    
                val_loss_sum = 0.0
                val_target_count = 0
                model.eval()

                # Evaluate validation loss
                for val_batch_items in val_loader:

                    with torch.no_grad():

                        # Get batch items
                        val_batch = val_batch_items["batch"].to(device).long()
                        val_labels = val_batch_items["labels"].to(device).long().clone()
                        val_predict_mask = val_batch_items["predict_mask"].to(device).bool()
                        val_attention_mask = val_batch_items["attention_mask"].to(device).bool()

                        # Get logits and CE loss
                        val_logits = model(val_batch, val_attention_mask)
                        val_labels[~val_predict_mask] = -100

                        val_batch_loss_sum = loss_fn_sum(
                            val_logits.reshape(-1, val_logits.size(-1)),  # [B*L, vocab_size]
                            val_labels.reshape(-1)                      # [B*L]
                        )
                        
                        val_loss_sum += val_batch_loss_sum.item()
                        val_target_count += (val_labels != -100).sum().item()

                # Normalize loss over prediction tokens
                val_loss = val_loss_sum / max(val_target_count, 1)
                
                # Verbose
                print(f"Step: {num_steps}, train loss: {train_loss}, validation loss: {val_loss}")
                
                # Update best validation loss if applicable
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    tokens_to_best_val_loss = tokens_seen
                    time_to_best_val_loss = time_elapsed

                    # Save model checkpoint
                    torch.save({
                        "epoch": epoch,
                        "num_steps": num_steps,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "val_loss": val_loss
                    }, f"model_at_step_{num_steps}.pt"
                    )

                    print(f"Saved best model at {num_steps} steps, val_loss = {val_loss:.4f}")

                # Report validation loss and reset train
                wandb_run.log(
                    {
                        "val_loss": val_loss
                     },
                    step = num_steps)
                
                model.train()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.perf_counter() - run_start_time

    # Run summary metrics
    wandb_run.summary["best_val_loss"] = best_val_loss
    wandb_run.summary["time_to_best_val_loss"] = time_to_best_val_loss
    wandb_run.summary["tokens_to_best_val_loss"] = tokens_to_best_val_loss
    wandb_run.summary["tokens_per_second"] = tokens_seen / total_time
    wandb_run.summary["steps_per_minute"] = num_steps * 60 / total_time
    wandb_run.summary["total_time"] = total_time

    print("Training complete!")

    return model


def train_moe_glm():
# For the MoE transformer**
# logits, aux_loss = model(batch, attention_mask)
# Then add scaled aux_loss to the overal loss function*


def train_sparse_glm():
