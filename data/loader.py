"""
Dataset Integration & Streaming Data Loader.
Connects directly to the Global Coding Intelligence Dataset across all 16 SE categories,
providing streaming iterators, PyTorch Datasets, sequence packing, and tokenized batches.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator, Generator, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer.tokenizer import UniversalCodingTokenizer

# Standard search paths for the Global Coding Intelligence Dataset
DEFAULT_DATASET_PATHS = [
    Path(r"g:\My Drive\data aiml model traning\universal-global-coding-dataset"),
    Path("./data/universal-global-coding-dataset"),
    Path("./data"),
    Path("../universal-global-coding-dataset"),
]


def find_dataset_root(custom_path: Optional[str] = None) -> Path:
    """Locates the active Global Coding Intelligence Dataset root directory."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p

    for path in DEFAULT_DATASET_PATHS:
        if path.exists() and ((path / "TRAINING_MANIFEST.json").exists() or (path / "train").exists()):
            return path

    raise FileNotFoundError(
        "Could not find Global Coding Intelligence Dataset. Checked: "
        + ", ".join(str(p) for p in DEFAULT_DATASET_PATHS)
    )


def stream_jsonl(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Memory-efficient streaming generator over JSONL files."""
    if not file_path.exists():
        return
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


class PretrainingDataset(Dataset):
    """PyTorch Dataset for Causal Pretraining (Source code & Documentation)."""

    def __init__(
        self,
        tokenizer: UniversalCodingTokenizer,
        dataset_root: Optional[Path] = None,
        split: str = "train",
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.root = dataset_root or find_dataset_root()

        # Locate files
        self.records: List[str] = []
        target_files = [
            self.root / split / f"{split}.jsonl",
            self.root / "pretraining" / "source_code" / "pretraining_corpus.jsonl",
        ]

        loaded = 0
        for f in target_files:
            if f.exists():
                for item in stream_jsonl(f):
                    # Extract raw text / code
                    text = item.get("content") or item.get("code") or item.get("prompt")
                    if text and len(text.strip()) > 10:
                        self.records.append(text)
                        loaded += 1
                        if max_samples and loaded >= max_samples:
                            break
            if max_samples and loaded >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.records[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Truncate or pad to max_seq_len
        if len(tokens) > self.max_seq_len:
            tokens = tokens[: self.max_seq_len]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        # For pretraining next-token prediction, labels = input_ids, pad masked with -100
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    """PyTorch Dataset for Supervised Fine-Tuning with ChatML prompt masking."""

    def __init__(
        self,
        tokenizer: UniversalCodingTokenizer,
        dataset_root: Optional[Path] = None,
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.root = dataset_root or find_dataset_root()

        self.samples: List[Tuple[List[int], List[int]]] = []

        sft_files = [
            self.root / "sft" / "code_generation" / "sft_generation_corpus.jsonl",
            self.root / "sft" / "code_explanation" / "sft_explanation_corpus.jsonl",
            self.root / "sft" / "code_debugging" / "debugging_corpus.jsonl",
            self.root / "train" / "train.jsonl",
        ]

        count = 0
        for sf in sft_files:
            if not sf.exists():
                continue
            for item in stream_jsonl(sf):
                messages = []
                if "messages" in item:
                    messages = item["messages"]
                elif "prompt" in item and "code" in item:
                    messages = [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["code"]},
                    ]
                elif "problem" in item and "code_after" in item:
                    # Debugging format
                    user_msg = f"Fix the following error:\n{item['problem']}\nBuggy code:\n{item.get('code_before', '')}"
                    assist_msg = f"Analysis:\n{item.get('analysis', '')}\n\nCorrected code:\n```python\n{item['code_after']}\n```"
                    messages = [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assist_msg},
                    ]

                if messages:
                    input_ids, labels = self.tokenizer.encode_chat(messages, add_generation_prompt=False)
                    if len(input_ids) > 10:
                        self.samples.append((input_ids, labels))
                        count += 1
                        if max_samples and count >= max_samples:
                            break
            if max_samples and count >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.samples[idx]

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DPODataset(Dataset):
    """PyTorch Dataset for Direct Preference Optimization (Chosen vs Rejected)."""

    def __init__(
        self,
        tokenizer: UniversalCodingTokenizer,
        dataset_root: Optional[Path] = None,
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.root = dataset_root or find_dataset_root()

        self.samples: List[Dict[str, str]] = []
        dpo_file = self.root / "preference" / "preference_corpus.jsonl"

        if dpo_file.exists():
            for item in stream_jsonl(dpo_file):
                prompt = item.get("prompt", "")
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")
                if prompt and chosen and rejected:
                    self.samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                    if max_samples and len(self.samples) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_ids, chosen_labels = self.tokenizer.encode_chat(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen}],
            add_generation_prompt=False,
        )
        rejected_ids, rejected_labels = self.tokenizer.encode_chat(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected}],
            add_generation_prompt=False,
        )

        def pad_or_truncate(ids, labels):
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len]
                labels = labels[: self.max_seq_len]
            else:
                pad_len = self.max_seq_len - len(ids)
                ids = ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
            return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

        c_ids, c_labels = pad_or_truncate(chosen_ids, chosen_labels)
        r_ids, r_labels = pad_or_truncate(rejected_ids, rejected_labels)

        return {
            "chosen_input_ids": c_ids,
            "chosen_labels": c_labels,
            "rejected_input_ids": r_ids,
            "rejected_labels": r_labels,
        }


class AgenticDataset(Dataset):
    """PyTorch Dataset for Autonomous Agent Trajectories and Tool Calling."""

    def __init__(
        self,
        tokenizer: UniversalCodingTokenizer,
        dataset_root: Optional[Path] = None,
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.root = dataset_root or find_dataset_root()

        self.samples: List[Tuple[List[int], List[int]]] = []
        agent_file = self.root / "agent_trajectories" / "agent_trajectories_corpus.jsonl"

        if agent_file.exists():
            for item in stream_jsonl(agent_file):
                goal = item.get("goal", "")
                trajectory = item.get("trajectory", [])

                # Build multi-turn dialog with tool invocations
                messages = [{"role": "user", "content": f"Engineering Goal: {goal}"}]
                for step in trajectory:
                    thought = step.get("thought", "")
                    action = step.get("action", "")
                    observation = step.get("observation", "")
                    assistant_turn = f"Thought: {thought}\n<|tool_call|>{action}<|tool_result|>"
                    messages.append({"role": "assistant", "content": assistant_turn})
                    if observation:
                        messages.append({"role": "user", "content": f"Tool Output:\n{observation}"})

                input_ids, labels = self.tokenizer.encode_chat(messages, add_generation_prompt=False)
                if len(input_ids) > 10:
                    self.samples.append((input_ids, labels))
                    if max_samples and len(self.samples) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.samples[idx]
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

