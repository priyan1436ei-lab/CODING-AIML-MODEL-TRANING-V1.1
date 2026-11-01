"""
Base Trainer & Optimization Infrastructure.
Features:
- Decoupled AdamW Optimizer (decay applied only to weight matrices, not bias or RMSNorm)
- Cosine Annealing Learning Rate Scheduler with Linear Warmup
- Gradient Accumulation & Gradient Norm Clipping
- Real-time Perplexity & Throughput (tokens/sec) calculation
- Automatic Checkpointing and Best-Loss Preservation
"""

import math
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.modeling import save_checkpoint


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Creates AdamW optimizer with weight decay applied exclusively to 2D parameters."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps = max(1, max_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr_factor = self.get_lr_factor(self.current_step)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * lr_factor

    def get_lr_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(self.warmup_steps)
        if step > self.max_steps:
            return self.min_lr_ratio
        progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        return {"current_step": self.current_step, "warmup_steps": self.warmup_steps, "max_steps": self.max_steps}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.current_step = state_dict.get("current_step", 0)


def compute_perplexity(loss: float) -> float:
    """Computes perplexity from cross-entropy loss."""
    try:
        return round(math.exp(min(loss, 20.0)), 2)
    except OverflowError:
        return 99999.99
