"""
Universal Coding AI Model - Central Training & Orchestration Engine.

Supported Commands:
  python train.py --check-hardware
  python train.py --validate-data
  python train.py --initialize-model
  python train.py --stage pretrain
  python train.py --stage sft
  python train.py --stage dpo
  python train.py --stage agentic
  python train.py --smoke-test
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from hardware_inspector import save_hardware_profile, print_hardware_summary, inspect_hardware
from data.validator import validate_dataset, print_validation_summary
from data.loader import find_dataset_root
from model.configuration import UniversalCodingConfig
from model.architecture import UniversalCodingForCausalLM
from model.modeling import create_model, count_parameters, save_checkpoint, load_checkpoint
from tokenizer.tokenizer import UniversalCodingTokenizer
from training.pretraining import train_pretrain
from training.sft_training import train_sft
from training.dpo_training import train_dpo
from training.agentic_training import train_agentic


def initialize_model_weights(
    preset: str = "educational_tiny",
    output_dir: str = "checkpoints/initialized",
    device: str = "cpu",
) -> UniversalCodingForCausalLM:
    """Initializes new model weights and validates architectural forward and backward pass."""
    print("=" * 80)
    print("      INITIALIZING UNIVERSAL CODING MODEL WEIGHTS")
    print("=" * 80)

    tokenizer = UniversalCodingTokenizer()
    config = UniversalCodingConfig.from_preset(preset, vocab_size=tokenizer.vocab_size)
    model = create_model(config=config, device=device)

    params = count_parameters(model)
    print(f"Architecture Preset  : {preset}")
    print(f"Vocab Size           : {config.vocab_size}")
    print(f"Hidden Dimension     : {config.hidden_size}")
    print(f"Attention Heads      : {config.num_attention_heads} (KV Heads: {config.num_key_value_heads})")
    print(f"Layers Count         : {config.num_hidden_layers}")
    print(f"Total Parameters     : {params['total_parameters']:,}")
    print(f"Estimated Memory     : {params['memory_mb']} MB")

    # Smoke test forward and backward pass
    print("\nExecuting forward pass smoke test...")
    dummy_input = torch.tensor([[1, 10, 20, 30, 40]], dtype=torch.long, device=device)
    dummy_labels = torch.tensor([[10, 20, 30, 40, 2]], dtype=torch.long, device=device)

    outputs = model(input_ids=dummy_input, labels=dummy_labels)
    loss = outputs["loss"]
    print(f"Smoke Test Initial Loss: {loss.item():.4f}")

    print("Executing backward pass smoke test...")
    loss.backward()
    print("Backward pass & gradient calculation verified successfully!")

    # Save initialized checkpoint
    save_checkpoint(model, config=config, output_dir=output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Checkpoint saved to: {output_dir}")
    print("=" * 80)
    return model


def run_smoke_test(device: str = "cpu"):
    """Comprehensive end-to-end verification of all pipeline stages."""
    print("=" * 80)
    print("      EXECUTING COMPREHENSIVE END-TO-END SMOKE TEST")
    print("=" * 80)

    # 1. Hardware check
    print("\n--- 1. Testing Hardware Inspection ---")
    prof = inspect_hardware()
    print(f"CPU Cores: {prof['cpu']['logical_cores']}, RAM: {prof['ram']['total_gb']} GB, PyTorch: {prof['pytorch']['version']}")

    # 2. Tokenizer test
    print("\n--- 2. Testing Tokenizer ---")
    tok = UniversalCodingTokenizer()
    sample_code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    enc = tok.encode(sample_code)
    dec = tok.decode(enc)
    assert dec == sample_code, "Tokenizer decode mismatch!"
    print(f"Encoded '{sample_code.strip()}' -> {len(enc)} tokens -> Decoded exactly!")

    # 3. Model init & step
    print("\n--- 3. Testing Model Forward & Backward Pass ---")
    init_dir = "checkpoints/smoke_init"
    model = initialize_model_weights(preset="educational_tiny", output_dir=init_dir, device=device)

    # 4. Small Pretraining Experiment
    print("\n--- 4. Testing Stage 1: Pretraining (10 steps) ---")
    pre_res = train_pretrain(
        max_steps=10,
        batch_size=2,
        gradient_accumulation_steps=1,
        device=device,
        output_dir="checkpoints/smoke_pretrain",
        log_interval=2,
    )
    assert Path("checkpoints/smoke_pretrain/final/checkpoint.pt").exists(), "Pretraining checkpoint missing!"

    # 5. Small SFT Experiment
    print("\n--- 5. Testing Stage 2: SFT (5 steps) ---")
    sft_res = train_sft(
        base_checkpoint="checkpoints/smoke_pretrain/final",
        max_steps=5,
        batch_size=2,
        gradient_accumulation_steps=1,
        device=device,
        output_dir="checkpoints/smoke_sft",
        log_interval=1,
    )
    assert Path("checkpoints/smoke_sft/final/checkpoint.pt").exists(), "SFT checkpoint missing!"

    # 6. Small DPO Experiment
    print("\n--- 6. Testing Stage 3: DPO (5 steps) ---")
    dpo_res = train_dpo(
        sft_checkpoint="checkpoints/smoke_sft/final",
        max_steps=5,
        batch_size=2,
        gradient_accumulation_steps=1,
        device=device,
        output_dir="checkpoints/smoke_dpo",
        log_interval=1,
    )
    assert Path("checkpoints/smoke_dpo/final/checkpoint.pt").exists(), "DPO checkpoint missing!"

    # 7. Small Agentic Experiment
    print("\n--- 7. Testing Stage 4: Agentic Training (5 steps) ---")
    agent_res = train_agentic(
        base_checkpoint="checkpoints/smoke_dpo/final",
        max_steps=5,
        batch_size=2,
        gradient_accumulation_steps=1,
        device=device,
        output_dir="checkpoints/smoke_agentic",
        log_interval=1,
    )
    assert Path("checkpoints/smoke_agentic/final/checkpoint.pt").exists(), "Agentic checkpoint missing!"

    # 8. Generation Test
    print("\n--- 8. Testing Inference & Generation ---")
    from inference.generator import CodeGenerator
    gen = CodeGenerator(checkpoint_dir="checkpoints/smoke_agentic/final", device=device)
    prompt = "def fibonacci(n: int) -> int:"
    output = gen.generate(prompt, max_new_tokens=40)
    print(f"Generated continuation for '{prompt}':\n{output[:100]}")

    print("=" * 80)
    print("      ALL END-TO-END SMOKE TESTS PASSED PERFECTLY!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Universal Coding AI Model Central Controller")
    parser.add_argument("--check-hardware", action="store_true", help="Inspect CPU, GPU, RAM, VRAM, and environment")
    parser.add_argument("--validate-data", action="store_true", help="Audit dataset schema, licenses, duplicates, and token counts")
    parser.add_argument("--initialize-model", action="store_true", help="Initialize and verify model weights")
    parser.add_argument("--stage", choices=["pretrain", "sft", "dpo", "agentic"], help="Execute specific training stage")
    parser.add_argument("--smoke-test", action="store_true", help="Run comprehensive multi-stage verification test")
    parser.add_argument("--preset", default="educational_tiny", help="Model preset (educational_tiny, code_micro, code_125m, code_1b)")
    parser.add_argument("--device", default="cpu", help="Target device (cpu or cuda)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--checkpoint", type=str, default=None, help="Base checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Stage configuration JSON file")

    args = parser.parse_args()

    # Load custom config if provided
    config_dict = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

    if args.check_hardware:
        prof = save_hardware_profile()
        print_hardware_summary(prof)

    elif args.validate_data:
        rep = validate_dataset()
        print_validation_summary(rep)

    elif args.initialize_model:
        initialize_model_weights(preset=args.preset, device=args.device)

    elif args.smoke_test:
        run_smoke_test(device=args.device)

    elif args.stage == "pretrain":
        train_pretrain(
            epochs=args.epochs,
            max_steps=args.max_steps or config_dict.get("max_steps", 50),
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr or config_dict.get("learning_rate", 3e-4),
            device=args.device,
        )

    elif args.stage == "sft":
        ckpt = args.checkpoint or config_dict.get("base_checkpoint")
        if not ckpt and Path("checkpoints/pretrain/final").exists():
            ckpt = "checkpoints/pretrain/final"
        train_sft(
            base_checkpoint=ckpt,
            epochs=args.epochs,
            max_steps=args.max_steps or config_dict.get("max_steps", 40),
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr or config_dict.get("learning_rate", 2e-4),
            device=args.device,
        )

    elif args.stage == "dpo":
        ckpt = args.checkpoint or config_dict.get("sft_checkpoint")
        if not ckpt and Path("checkpoints/sft/final").exists():
            ckpt = "checkpoints/sft/final"
        train_dpo(
            sft_checkpoint=ckpt,
            epochs=args.epochs,
            max_steps=args.max_steps or config_dict.get("max_steps", 30),
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr or config_dict.get("learning_rate", 5e-5),
            device=args.device,
        )

    elif args.stage == "agentic":
        ckpt = args.checkpoint or config_dict.get("base_checkpoint")
        if not ckpt:
            for cand in ["checkpoints/dpo/final", "checkpoints/sft/final", "checkpoints/pretrain/final"]:
                if Path(cand).exists():
                    ckpt = cand
                    break
        train_agentic(
            base_checkpoint=ckpt,
            epochs=args.epochs,
            max_steps=args.max_steps or config_dict.get("max_steps", 30),
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr or config_dict.get("learning_rate", 1e-4),
            device=args.device,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
