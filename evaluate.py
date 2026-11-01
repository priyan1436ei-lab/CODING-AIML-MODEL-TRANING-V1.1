"""
Universal Coding AI Model - Evaluation CLI.

Usage:
  python evaluate.py --all
  python evaluate.py --checkpoint checkpoints/sft/final --num-samples 25
"""

import argparse
from pathlib import Path

from evaluation.evaluator import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Universal Coding Model Evaluation Benchmark CLI")
    parser.add_argument("--all", action="store_true", help="Run full evaluation suite (HumanEval, MBPP, Syntax, Agent Tools)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint directory to evaluate")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of benchmark samples to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Target device (cpu or cuda)")
    parser.add_argument("--output", type=str, default="reports/evaluation_report.json", help="Path to save evaluation report JSON")

    args = parser.parse_args()

    evaluate_model(
        model_checkpoint=args.checkpoint,
        num_eval_samples=args.num_samples,
        device=args.device,
        output_report_path=args.output,
    )


if __name__ == "__main__":
    main()

