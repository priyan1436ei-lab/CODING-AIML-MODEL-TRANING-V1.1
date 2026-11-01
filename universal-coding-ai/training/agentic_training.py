"""
Agentic Coding & Autonomous Software Engineering Training Pipeline.
Trains model to formulate thoughts, select tools, execute commands in sandbox,
detect runtime defects, and repair repositories autonomously.
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
from model.modeling import (
    create_model, count_parameters, save_checkpoint, load_checkpoint
)
from tokenizer.tokenizer import UniversalCodingTokenizer
from data.loader import AgenticDataset, find_dataset_root
from training.trainer import build_optimizer, CosineWarmupScheduler, compute_perplexity


def train_agentic(
    base_checkpoint: Optional[str] = None,
    config: Optional[UniversalCodingConfig] = None,
    tokenizer: Optional[UniversalCodingTokenizer] = None,
    dataset_root: Optional[str] = None,
    epochs: int = 2,
    max_steps: Optional[int] = 30,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-4,
    warmup_steps: int = 5,
    max_seq_len: int = 512,
    device: str = "cpu",
    output_dir: str = "checkpoints/agentic",
    log_interval: int = 5,
) -> Dict[str, Any]:
    """Executes Agentic Trajectory training."""
    print("=" * 80)
    print("      STAGE 4: AGENTIC CODING & TOOL-USE TRAINING INITIALIZATION")
    print("=" * 80)

    if tokenizer is None:
        tokenizer = UniversalCodingTokenizer()

    # Load previous checkpoint (DPO, SFT, or Pretrain) if available
    ckpt_to_load = base_checkpoint
    if not ckpt_to_load:
        for candidate in ["checkpoints/dpo/final", "checkpoints/sft/final", "checkpoints/pretrain/final"]:
            if Path(candidate).exists():
                ckpt_to_load = candidate
                break

    if ckpt_to_load and Path(ckpt_to_load).exists():
        print(f"Loading Base Checkpoint: {ckpt_to_load}")
        model, config, meta = load_checkpoint(ckpt_to_load, device=device)
    else:
        if config is None:
            config = UniversalCodingConfig.from_preset("educational_tiny", vocab_size=tokenizer.vocab_size)
        model = UniversalCodingForCausalLM(config).to(device)

    # Setup Agentic Dataset
    root = find_dataset_root(dataset_root)
    agent_dataset = AgenticDataset(tokenizer, dataset_root=root, max_seq_len=max_seq_len)

    if len(agent_dataset) == 0:
        raise ValueError(f"No agentic trajectory records found at {root}")

    train_loader = DataLoader(agent_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"Agentic Trajectories: {len(agent_dataset)} multi-turn samples")

    optimizer = build_optimizer(model, learning_rate=learning_rate)
    total_training_steps = max_steps if max_steps else len(train_loader) * epochs
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=warmup_steps, max_steps=total_training_steps)

    model.train()
    step = 0
    history = []

    print("-" * 80)
    print(f"{'Step':<8} | {'Epoch':<6} | {'Agent Loss':<12} | {'Perplexity':<10} | {'LR':<10}")
    print("-" * 80)

    for epoch in range(epochs):
        accumulated_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"] / gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                step_loss = accumulated_loss
                ppl = compute_perplexity(step_loss)
                current_lr = scheduler.get_last_lr()[0]

                if step % log_interval == 0:
                    print(f"{step:<8} | {epoch+1:<6} | {step_loss:<12.4f} | {ppl:<10.2f} | {current_lr:<10.2e}")
                    history.append({
                        "step": step,
                        "epoch": epoch + 1,
                        "loss": round(step_loss, 4),
                        "perplexity": ppl,
                        "lr": current_lr,
                    })

                accumulated_loss = 0.0

                if max_steps and step >= max_steps:
                    break

        if max_steps and step >= max_steps:
            break

    # Save Agentic model
    save_checkpoint(
        model, optimizer, scheduler, config,
        step=step, epoch=epoch + 1, loss=step_loss if 'step_loss' in locals() else 0.0,
        output_dir=f"{output_dir}/final",
    )
    tokenizer.save_pretrained(f"{output_dir}/final")

    print("=" * 80)
    print(f"Agentic Alignment Completed: {step} steps.")
    print(f"Final Checkpoint Saved: {output_dir}/final")
    print("=" * 80)

    return {
        "final_step": step,
        "final_loss": round(step_loss, 4) if 'step_loss' in locals() else 0.0,
        "history": history,
        "checkpoint_dir": f"{output_dir}/final",
    }
