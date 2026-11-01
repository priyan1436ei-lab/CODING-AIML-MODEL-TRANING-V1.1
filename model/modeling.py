"""
High-Level Model Manager and Parameter-Efficient Adapter Framework.
Supports:
- Model instantiation, inspection, parameter counting, and memory estimation
- Custom Transformer and Pretrained Open-Weight Foundation Model adapters
- Native Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
- Safe saving and loading via PyTorch checkpoints and Safetensors
"""

import math
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn

try:
    from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from model.configuration import UniversalCodingConfig
from model.architecture import UniversalCodingForCausalLM


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer (Hu et al., 2021)."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze base parameters
        for param in self.base_linear.parameters():
            param.requires_grad = False

        # Trainable low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming uniform for A, zero for B so delta is initially 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str] = ["q_proj", "v_proj", "gate_proj", "up_proj"],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> int:
    """Wraps targeted linear layers with LoRA adapters and returns count of adapted layers."""
    adapted_count = 0
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                sub_mod = getattr(module, target)
                if isinstance(sub_mod, nn.Linear) and not isinstance(sub_mod, LoRALinear):
                    setattr(module, target, LoRALinear(sub_mod, rank=rank, alpha=alpha, dropout=dropout))
                    adapted_count += 1
    return adapted_count


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Calculates total, trainable, and memory metrics for a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Approximate memory footprint in MB
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    memory_mb = round(param_bytes / (1024**2), 2)
    memory_gb = round(param_bytes / (1024**3), 3)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "trainable_percentage": round((trainable_params / total_params * 100), 2) if total_params > 0 else 0,
        "memory_mb": memory_mb,
        "memory_gb": memory_gb,
    }


def create_model(
    config: Optional[UniversalCodingConfig] = None,
    preset: Optional[str] = None,
    device: str = "cpu",
    precision: str = "fp32",
    **config_overrides,
) -> UniversalCodingForCausalLM:
    """Instantiates a UniversalCodingForCausalLM model with configured settings."""
    if config is None:
        if preset is not None:
            config = UniversalCodingConfig.from_preset(preset, **config_overrides)
        else:
            config = UniversalCodingConfig(**config_overrides)

    model = UniversalCodingForCausalLM(config)

    # Set precision
    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.to(torch.bfloat16)
    else:
        model = model.float()

    model = model.to(device)
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    config: Optional[UniversalCodingConfig] = None,
    step: int = 0,
    epoch: int = 0,
    loss: float = 0.0,
    output_dir: str = "checkpoints/latest",
    save_safetensors_format: bool = True,
):
    """Saves complete model checkpoint with metadata, config, and optimizer states."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save Config
    if config is not None:
        config.save_pretrained(out)

    # Save Model Weights (Safetensors + PyTorch .pt)
    state_dict = model.state_dict()

    if HAS_SAFETENSORS and save_safetensors_format:
        try:
            safetensors_path = out / "model.safetensors"
            # Ensure tensors are contiguous
            contiguous_state = {k: v.contiguous() for k, v in state_dict.items()}
            save_safetensors(contiguous_state, str(safetensors_path))
        except Exception:
            pass

    # Save PyTorch checkpoint
    checkpoint_data = {
        "model_state_dict": state_dict,
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "config_dict": config.to_dict() if config else None,
    }
    if optimizer is not None:
        checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint_data, out / "checkpoint.pt")


def load_checkpoint(
    checkpoint_dir: str,
    device: str = "cpu",
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[UniversalCodingForCausalLM, UniversalCodingConfig, Dict[str, Any]]:
    """Loads a saved checkpoint into a newly constructed model."""
    path = Path(checkpoint_dir)
    config = UniversalCodingConfig.from_pretrained(path)
    model = UniversalCodingForCausalLM(config)

    meta = {}
    pt_file = path / "checkpoint.pt"
    st_file = path / "model.safetensors"

    if HAS_SAFETENSORS and st_file.exists():
        state_dict = load_safetensors(str(st_file), device=device)
        model.load_state_dict(state_dict, strict=False)
    elif pt_file.exists():
        ckpt = torch.save if False else torch.load(pt_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        meta = {
            "step": ckpt.get("step", 0),
            "epoch": ckpt.get("epoch", 0),
            "loss": ckpt.get("loss", 0.0),
        }
        if load_optimizer and optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        raise FileNotFoundError(f"No checkpoint file found in {checkpoint_dir}")

    model.to(device)
    return model, config, meta

