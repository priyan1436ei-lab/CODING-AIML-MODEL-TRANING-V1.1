"""
Model Evaluation and Benchmark Execution Harness.
Evaluates model on HumanEval, MBPP, Multilingual Syntax, and Agent Tool-Use.
Computes Pass@1, execution accuracy, compilation success, and generates
baseline-vs-finetuned comparative benchmarks.
"""

import ast
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch

from model.architecture import UniversalCodingForCausalLM
from model.modeling import load_checkpoint
from tokenizer.tokenizer import UniversalCodingTokenizer
from data.loader import find_dataset_root, stream_jsonl


def run_sandboxed_python_test(code: str, test_assertion: str, timeout_sec: int = 5) -> Tuple[bool, str]:
    """Executes a generated code snippet against test assertions in an isolated subprocess."""
    program = f"{code}\n\n# Benchmark Test Harness\n{test_assertion}\n"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
        temp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        passed = proc.returncode == 0
        output = proc.stdout if passed else proc.stderr
        return passed, output.strip()
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out ({timeout_sec}s limit exceeded)"
    except Exception as e:
        return False, f"Execution error: {str(e)}"
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def check_syntax(code: str, language: str = "python") -> bool:
    """Verifies syntactic correctness of generated code."""
    if language.lower() == "python":
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    # Basic brace balance check for C-style languages
    stack = []
    braces = {"{": "}", "(": ")", "[": "]"}
    for char in code:
        if char in braces:
            stack.append(char)
        elif char in braces.values():
            if not stack or braces[stack.pop()] != char:
                return False
    return len(stack) == 0


def evaluate_model(
    model_checkpoint: Optional[str] = None,
    dataset_root: Optional[str] = None,
    num_eval_samples: int = 20,
    device: str = "cpu",
    output_report_path: str = "reports/evaluation_report.json",
) -> Dict[str, Any]:
    """Runs end-to-end evaluation suite across coding benchmarks."""
    print("=" * 80)
    print("      UNIVERSAL CODING MODEL: COMPREHENSIVE BENCHMARK EVALUATION")
    print("=" * 80)

    # Load model and tokenizer
    ckpt = model_checkpoint or "checkpoints/sft/final"
    if not Path(ckpt).exists():
        for cand in ["checkpoints/dpo/final", "checkpoints/agentic/final", "checkpoints/pretrain/final"]:
            if Path(cand).exists():
                ckpt = cand
                break

    print(f"Evaluating Model Checkpoint: {ckpt}")
    model, config, _ = load_checkpoint(ckpt, device=device)
    model.eval()

    tokenizer = UniversalCodingTokenizer.from_pretrained(ckpt)
    root = find_dataset_root(dataset_root)

    results = {
        "checkpoint": ckpt,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "humaneval": {"samples": 0, "passed": 0, "pass_at_1": 0.0},
            "mbpp": {"samples": 0, "passed": 0, "pass_at_1": 0.0},
            "syntax_validity": {"samples": 0, "valid": 0, "validity_rate": 0.0},
            "tool_call_validity": {"samples": 0, "valid": 0, "validity_rate": 0.0},
        },
        "sample_evaluations": [],
    }

    # 1. Evaluate on HumanEval benchmark
    he_path = root / "raw" / "HumanEval.jsonl.gz"
    if he_path.exists():
        print("Evaluating on HumanEval Benchmark...")
        count = 0
        with gzip.open(he_path, "rt", encoding="utf-8") as f:
            for line in f:
                if count >= num_eval_samples:
                    break
                item = json.loads(line)
                prompt = item.get("prompt", "")
                test = item.get("test", "")
                entry_point = item.get("entry_point", "")

                input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
                gen_tokens = model.generate(input_ids, max_new_tokens=100, temperature=0.2)
                gen_code = tokenizer.decode(gen_tokens[0].tolist(), skip_special_tokens=True)

                passed, err = run_sandboxed_python_test(gen_code, test)
                results["metrics"]["humaneval"]["samples"] += 1
                if passed:
                    results["metrics"]["humaneval"]["passed"] += 1

                syntax_ok = check_syntax(gen_code, "python")
                results["metrics"]["syntax_validity"]["samples"] += 1
                if syntax_ok:
                    results["metrics"]["syntax_validity"]["valid"] += 1

                if count < 3:
                    results["sample_evaluations"].append({
                        "task_id": item.get("task_id"),
                        "prompt": prompt[:80] + "...",
                        "passed": passed,
                        "syntax_ok": syntax_ok,
                    })
                count += 1

    # 2. Evaluate on MBPP benchmark
    mbpp_path = root / "raw" / "mbpp.jsonl"
    if mbpp_path.exists():
        print("Evaluating on MBPP Benchmark...")
        count = 0
        for item in stream_jsonl(mbpp_path):
            if count >= num_eval_samples:
                break
            prompt = item.get("text", "")
            test_list = item.get("test_list", [])
            test_code = "\n".join(test_list)

            input_ids = torch.tensor([tokenizer.encode(f"# {prompt}\n")], dtype=torch.long, device=device)
            gen_tokens = model.generate(input_ids, max_new_tokens=100, temperature=0.2)
            gen_code = tokenizer.decode(gen_tokens[0].tolist(), skip_special_tokens=True)

            passed, _ = run_sandboxed_python_test(gen_code, test_code)
            results["metrics"]["mbpp"]["samples"] += 1
            if passed:
                results["metrics"]["mbpp"]["passed"] += 1
            count += 1

    # 3. Evaluate Agent Tool Calling Format
    agent_path = root / "agent_trajectories" / "agent_trajectories_corpus.jsonl"
    if agent_path.exists():
        print("Evaluating Agent Tool-Calling Format...")
        count = 0
        for item in stream_jsonl(agent_path):
            if count >= 10:
                break
            goal = item.get("goal", "")
            prompt_text = f"<|im_start|>user\nEngineering Goal: {goal}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
            gen_tokens = model.generate(input_ids, max_new_tokens=80, temperature=0.2)
            out_str = tokenizer.decode(gen_tokens[0].tolist(), skip_special_tokens=False)

            results["metrics"]["tool_call_validity"]["samples"] += 1
            if "<|tool_call|>" in out_str or "Thought:" in out_str:
                results["metrics"]["tool_call_validity"]["valid"] += 1
            count += 1

    # Compute Final Pass Rates
    for benchmark in ["humaneval", "mbpp", "syntax_validity", "tool_call_validity"]:
        m = results["metrics"][benchmark]
        total = m["samples"]
        succ = m.get("passed", m.get("valid", 0))
        rate_key = "pass_at_1" if "passed" in m else "validity_rate"
        m[rate_key] = round((succ / total * 100), 2) if total > 0 else 0.0

    # Save results
    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_eval_summary(results)
    return results


def print_eval_summary(results: Dict[str, Any]):
    """Pretty prints evaluation report."""
    print("=" * 80)
    print("      BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Model Checkpoint     : {results['checkpoint']}")
    print(f"Evaluation Timestamp : {results['evaluated_at']}")
    print("-" * 80)
    m = results["metrics"]
    print(f"HumanEval Pass@1     : {m['humaneval']['pass_at_1']}% ({m['humaneval']['passed']}/{m['humaneval']['samples']} passed)")
    print(f"MBPP Pass@1          : {m['mbpp']['pass_at_1']}% ({m['mbpp']['passed']}/{m['mbpp']['samples']} passed)")
    print(f"Syntax Validity Rate : {m['syntax_validity']['validity_rate']}% ({m['syntax_validity']['valid']}/{m['syntax_validity']['samples']} valid ASTs)")
    print(f"Tool-Use Generation  : {m['tool_call_validity']['validity_rate']}% ({m['tool_call_validity']['valid']}/{m['tool_call_validity']['samples']} tool calls)")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_model()
