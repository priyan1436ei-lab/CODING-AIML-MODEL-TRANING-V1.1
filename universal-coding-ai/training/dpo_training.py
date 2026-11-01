"""
Direct Preference Optimization (DPO) Training Pipeline.
Aligns model toward high-quality, bug-free, and secure code solutions
over suboptimal, buggy, or vulnerable candidates.
"""

import copy
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.configuration import UniversalCodingConfig
from model.architecture import UniversalCodingForCausalLM
from model.modeling import (
    create_model, count_parameters, save_checkpoint, load_checkpoint
)
from tokenizer.tokenizer import UniversalCodingTokenizer
from data.loader import DPODataset, find_dataset_root
from training.trainer import build_optimizer, CosineWarmupScheduler


def compute_sequence_log_probs(
    model: UniversalCodingForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Computes summed token log-probabilities for target tokens (where labels != -100)."""
    outputs = model(input_ids=input_ids)
    logits = outputs["logits"]

    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log prob of actual target token
    loss_mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~loss_mask] = 0

    per_token_logps = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    per_token_logps = per_token_logps * loss_mask

    # Sum along sequence dimension
    return per_token_logps.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates DPO loss and implicit reward margins."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)
    losses = -F.logsigmoid(logits)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def train_dpo(
    sft_checkpoint: Optional[str] = None,
    config: Optional[UniversalCodingConfig] = None,
    tokenizer: Optional[UniversalCodingTokenizer] = None,
    dataset_root: Optional[str] = None,
    epochs: int = 2,
    max_steps: Optional[int] = 30,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    warmup_steps: int = 5,
    max_seq_len: int = 512,
    device: str = "cpu",
    output_dir: str = "checkpoints/dpo",
    log_interval: int = 5,
) -> Dict[str, Any]:
    """Executes DPO Preference Optimization training."""
    print("=" * 80)
    print("      STAGE 3: DIRECT PREFERENCE OPTIMIZATION (DPO) INITIALIZATION")
    print("=" * 80)

    if tokenizer is None:
        tokenizer = UniversalCodingTokenizer()

    # Load SFT model as active policy
    if sft_checkpoint and Path(sft_checkpoint).exists():
        print(f"Loading Active Policy from: {sft_checkpoint}")
        policy_model, config, _ = load_checkpoint(sft_checkpoint, device=device)
    else:
        # Fallback to SFT final or new model
        default_sft = Path("checkpoints/sft/final")
        if default_sft.exists():
            policy_model, config, _ = load_checkpoint(str(default_sft), device=device)
        else:
            if config is None:
                config = UniversalCodingConfig.from_preset("educational_tiny", vocab_size=tokenizer.vocab_size)
            policy_model = UniversalCodingForCausalLM(config).to(device)

    # Reference model is a frozen copy of initial policy
    print("Cloning and freezing Reference Model...")
    reference_model = copy.deepcopy(policy_model).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Setup DPO Dataset
    root = find_dataset_root(dataset_root)
    dpo_dataset = DPODataset(tokenizer, dataset_root=root, max_seq_len=max_seq_len)

    if len(dpo_dataset) == 0:
        raise ValueError(f"No DPO preference records found at {root}")

    train_loader = DataLoader(dpo_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"DPO Preference Pairs: {len(dpo_dataset)}")
    print(f"Beta Hyperparameter : {beta}")

    optimizer = build_optimizer(policy_model, learning_rate=learning_rate)
    total_training_steps = max_steps if max_steps else len(train_loader) * epochs
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=warmup_steps, max_steps=total_training_steps)

    policy_model.train()
    step = 0
    history = []

    print("-" * 80)
    print(f"{'Step':<8} | {'Epoch':<6} | {'DPO Loss':<10} | {'Chosen Rwd':<12} | {'Rejected Rwd':<12}")
    print("-" * 80)

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            c_in = batch["chosen_input_ids"].to(device)
            c_lbl = batch["chosen_labels"].to(device)
            r_in = batch["rejected_input_ids"].to(device)
            r_lbl = batch["rejected_labels"].to(device)

            # Active policy log probs
            pi_chosen = compute_sequence_log_probs(policy_model, c_in, c_lbl)
            pi_rejected = compute_sequence_log_probs(policy_model, r_in, r_lbl)

            # Reference model log probs (frozen)
            with torch.no_grad():
                ref_chosen = compute_sequence_log_probs(reference_model, c_in, c_lbl)
                ref_rejected = compute_sequence_log_probs(reference_model, r_in, r_lbl)

            loss, chosen_rwd, rej_rwd = dpo_loss(
                pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=beta
            )

            loss_scaled = loss / gradient_accumulation_steps
            loss_scaled.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % log_interval == 0:
                    print(f"{step:<8} | {epoch+1:<6} | {loss.item():<10.4f} | {chosen_rwd.item():<12.4f} | {rej_rwd.item():<12.4f}")
                    history.append({
                        "step": step,
                        "epoch": epoch + 1,
                        "dpo_loss": round(loss.item(), 4),
                        "chosen_reward": round(chosen_rwd.item(), 4),
                        "rejected_reward": round(rej_rwd.item(), 4),
                    })

                if max_steps and step >= max_steps:
                    break

        if max_steps and step >= max_steps:
            break

    # Save DPO aligned model
    save_checkpoint(
        policy_model, optimizer, scheduler, config,
        step=step, epoch=epoch + 1, loss=loss.item() if 'loss' in locals() else 0.0,
        output_dir=f"{output_dir}/final",
    )
    tokenizer.save_pretrained(f"{output_dir}/final")

    print("=" * 80)
    print(f"DPO Alignment Completed: {step} steps.")
    print(f"Final Checkpoint Saved: {output_dir}/final")
    print("=" * 80)

    return {
        "final_step": step,
        "final_loss": round(loss.item(), 4) if 'loss' in locals() else 0.0,
        "history": history,
        "checkpoint_dir": f"{output_dir}/final",
    }
