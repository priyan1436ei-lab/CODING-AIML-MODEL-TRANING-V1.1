"""
Causal Language Model Pretraining Pipeline.
Trains the Universal Coding Model on verified multi-language source code and documentation.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.configuration import UniversalCodingConfig
from model.architecture import UniversalCodingForCausalLM
from model.modeling import create_model, count_parameters, save_checkpoint
from tokenizer.tokenizer import UniversalCodingTokenizer
from data.loader import PretrainingDataset, find_dataset_root
from training.trainer import build_optimizer, CosineWarmupScheduler, compute_perplexity


def train_pretrain(
    config: Optional[UniversalCodingConfig] = None,
    tokenizer: Optional[UniversalCodingTokenizer] = None,
    dataset_root: Optional[str] = None,
    epochs: int = 1,
    max_steps: Optional[int] = 50,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 10,
    max_seq_len: int = 512,
    device: str = "cpu",
    output_dir: str = "checkpoints/pretrain",
    log_interval: int = 5,
    eval_interval: int = 25,
) -> Dict[str, Any]:
    """Executes Causal Pretraining loop with metric logging and checkpointing."""
    print("=" * 80)
    print("      STAGE 1: CAUSAL PRETRAINING INITIALIZATION")
    print("=" * 80)

    # Initialize Tokenizer & Model
    if tokenizer is None:
        tokenizer = UniversalCodingTokenizer()

    if config is None:
        config = UniversalCodingConfig.from_preset("educational_tiny", vocab_size=tokenizer.vocab_size)
    else:
        config.vocab_size = tokenizer.vocab_size

    model = UniversalCodingForCausalLM(config).to(device)
    params_info = count_parameters(model)
    print(f"Model Architecture  : {config.num_hidden_layers} Layers, {config.hidden_size} Hidden, {config.num_attention_heads} Heads")
    print(f"Total Parameters    : {params_info['total_parameters']:,} ({params_info['memory_mb']} MB)")
    print(f"Target Device       : {device.upper()}")

    # Setup Datasets
    root = find_dataset_root(dataset_root)
    train_dataset = PretrainingDataset(tokenizer, dataset_root=root, split="train", max_seq_len=max_seq_len)
    val_dataset = PretrainingDataset(tokenizer, dataset_root=root, split="validation", max_seq_len=max_seq_len)

    if len(train_dataset) == 0:
        raise ValueError(f"No pretraining records found at {root}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) > 0 else None

    print(f"Train Dataset Size  : {len(train_dataset)} sequences")
    print(f"Validation Size     : {len(val_dataset)} sequences")
    print(f"Effective Batch Size: {batch_size * gradient_accumulation_steps} sequences")

    # Optimizer & Scheduler
    optimizer = build_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    total_training_steps = max_steps if max_steps else (len(train_loader) * epochs) // gradient_accumulation_steps
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=warmup_steps, max_steps=total_training_steps)

    # Training state
    model.train()
    step = 0
    total_tokens_processed = 0
    best_val_loss = float("inf")
    start_time = time.time()
    history = []

    print("-" * 80)
    print(f"{'Step':<8} | {'Epoch':<6} | {'Loss':<8} | {'Perplexity':<10} | {'LR':<10} | {'Tok/Sec':<8}")
    print("-" * 80)

    for epoch in range(epochs):
        epoch_loss = 0.0
        accumulated_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"] / gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()
            num_tokens = (labels != -100).sum().item()
            total_tokens_processed += num_tokens

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                step_loss = accumulated_loss
                ppl = compute_perplexity(step_loss)
                current_lr = scheduler.get_last_lr()[0]
                elapsed = max(0.001, time.time() - start_time)
                tok_per_sec = int(total_tokens_processed / elapsed)

                if step % log_interval == 0:
                    print(f"{step:<8} | {epoch+1:<6} | {step_loss:<8.4f} | {ppl:<10.2f} | {current_lr:<10.2e} | {tok_per_sec:<8}")
                    history.append({
                        "step": step,
                        "epoch": epoch + 1,
                        "loss": round(step_loss, 4),
                        "perplexity": ppl,
                        "lr": current_lr,
                        "tokens_per_sec": tok_per_sec,
                    })

                # Evaluation check
                if val_loader is not None and step % eval_interval == 0:
                    model.eval()
                    val_loss_acc = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for val_b in val_loader:
                            v_in = val_b["input_ids"].to(device)
                            v_lbl = val_b["labels"].to(device)
                            v_out = model(input_ids=v_in, labels=v_lbl)
                            val_loss_acc += v_out["loss"].item()
                            val_batches += 1
                            if val_batches >= 10:
                                break
                    avg_val_loss = val_loss_acc / max(1, val_batches)
                    val_ppl = compute_perplexity(avg_val_loss)
                    print(f" >>> [VAL EVAL] Step {step}: Val Loss = {avg_val_loss:.4f}, Val PPL = {val_ppl:.2f}")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save_checkpoint(
                            model, optimizer, scheduler, config,
                            step=step, epoch=epoch + 1, loss=avg_val_loss,
                            output_dir=f"{output_dir}/best",
                        )
                    model.train()

                accumulated_loss = 0.0

                if max_steps and step >= max_steps:
                    break

        if max_steps and step >= max_steps:
            break

    # Save final pretraining checkpoint
    save_checkpoint(
        model, optimizer, scheduler, config,
        step=step, epoch=epoch + 1, loss=step_loss,
        output_dir=f"{output_dir}/final",
    )
    tokenizer.save_pretrained(f"{output_dir}/final")

    print("=" * 80)
    print(f"Pretraining Completed: {step} steps, {total_tokens_processed:,} tokens processed.")
    print(f"Final Checkpoint Saved: {output_dir}/final")
    print("=" * 80)

    return {
        "final_step": step,
        "total_tokens_processed": total_tokens_processed,
        "final_loss": round(step_loss, 4),
        "best_val_loss": round(best_val_loss, 4) if best_val_loss != float("inf") else round(step_loss, 4),
        "history": history,
        "checkpoint_dir": f"{output_dir}/final",
    }
