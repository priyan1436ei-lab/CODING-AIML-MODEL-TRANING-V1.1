"""
Supervised Fine-Tuning (SFT) Instruction-Alignment Pipeline.
Teaches the model programming instructions, code generation, debugging,
and algorithm design with loss masking on user prompt tokens.
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
    create_model, count_parameters, save_checkpoint,
    load_checkpoint, apply_lora_to_model
)
from tokenizer.tokenizer import UniversalCodingTokenizer
from data.loader import SFTDataset, find_dataset_root
from training.trainer import build_optimizer, CosineWarmupScheduler, compute_perplexity


def train_sft(
    base_checkpoint: Optional[str] = None,
    config: Optional[UniversalCodingConfig] = None,
    tokenizer: Optional[UniversalCodingTokenizer] = None,
    dataset_root: Optional[str] = None,
    epochs: int = 2,
    max_steps: Optional[int] = 40,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 5,
    max_seq_len: int = 512,
    use_lora: bool = False,
    lora_rank: int = 16,
    device: str = "cpu",
    output_dir: str = "checkpoints/sft",
    log_interval: int = 5,
) -> Dict[str, Any]:
    """Executes Supervised Fine-Tuning loop."""
    print("=" * 80)
    print("      STAGE 2: SUPERVISED FINE-TUNING (SFT) INITIALIZATION")
    print("=" * 80)

    # Initialize Tokenizer
    if tokenizer is None:
        tokenizer = UniversalCodingTokenizer()

    # Load or initialize model
    if base_checkpoint and Path(base_checkpoint).exists():
        print(f"Loading Base Checkpoint: {base_checkpoint}")
        model, config, meta = load_checkpoint(base_checkpoint, device=device)
    else:
        if config is None:
            config = UniversalCodingConfig.from_preset("educational_tiny", vocab_size=tokenizer.vocab_size)
        else:
            config.vocab_size = tokenizer.vocab_size
        model = UniversalCodingForCausalLM(config).to(device)

    # Apply LoRA if requested
    if use_lora:
        adapted_count = apply_lora_to_model(model, rank=lora_rank)
        print(f"Applied LoRA Adapters to {adapted_count} linear layers (Rank={lora_rank})")

    params_info = count_parameters(model)
    print(f"Total Parameters    : {params_info['total_parameters']:,}")
    print(f"Trainable Parameters: {params_info['trainable_parameters']:,} ({params_info['trainable_percentage']}%)")
    print(f"Target Device       : {device.upper()}")

    # Setup Dataset
    root = find_dataset_root(dataset_root)
    sft_dataset = SFTDataset(tokenizer, dataset_root=root, max_seq_len=max_seq_len)

    if len(sft_dataset) == 0:
        raise ValueError(f"No SFT records found at {root}")

    train_loader = DataLoader(sft_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"SFT Dataset Samples : {len(sft_dataset)}")
    print(f"Effective Batch Size: {batch_size * gradient_accumulation_steps}")

    # Optimizer & Scheduler
    optimizer = build_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    total_training_steps = max_steps if max_steps else (len(train_loader) * epochs) // gradient_accumulation_steps
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=warmup_steps, max_steps=total_training_steps)

    model.train()
    step = 0
    start_time = time.time()
    history = []

    print("-" * 80)
    print(f"{'Step':<8} | {'Epoch':<6} | {'SFT Loss':<10} | {'Perplexity':<10} | {'LR':<10}")
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
                    print(f"{step:<8} | {epoch+1:<6} | {step_loss:<10.4f} | {ppl:<10.2f} | {current_lr:<10.2e}")
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

    # Save SFT checkpoint
    save_checkpoint(
        model, optimizer, scheduler, config,
        step=step, epoch=epoch + 1, loss=step_loss,
        output_dir=f"{output_dir}/final",
    )
    tokenizer.save_pretrained(f"{output_dir}/final")

    print("=" * 80)
    print(f"SFT Training Completed: {step} steps.")
    print(f"Final Checkpoint Saved: {output_dir}/final")
    print("=" * 80)

    return {
        "final_step": step,
        "final_loss": round(step_loss, 4),
        "history": history,
        "checkpoint_dir": f"{output_dir}/final",
    }
