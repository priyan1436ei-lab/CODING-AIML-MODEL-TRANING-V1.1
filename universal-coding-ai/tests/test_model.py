"""
Unit tests for Universal Coding AI Model architecture, tokenizer, sandbox, and training.
"""

import pytest
import torch
from pathlib import Path

from model.configuration import UniversalCodingConfig
from model.architecture import UniversalCodingForCausalLM, RMSNorm, RotaryEmbedding
from model.modeling import count_parameters, apply_lora_to_model
from tokenizer.tokenizer import UniversalCodingTokenizer
from sandbox.sandbox_manager import SandboxManager
from tools.workspace_tools import ToolRegistry


def test_tokenizer_encoding_and_decoding():
    tokenizer = UniversalCodingTokenizer()
    code = "def solve_puzzle(n: int) -> bool:\n    return n % 2 == 0\n"
    tokens = tokenizer.encode(code)
    assert len(tokens) > 0
    decoded = tokenizer.decode(tokens)
    assert decoded == code


def test_tokenizer_chat_formatting():
    tokenizer = UniversalCodingTokenizer()
    messages = [
        {"role": "user", "content": "Write quicksort"},
        {"role": "assistant", "content": "def quicksort(arr): return sorted(arr)"},
    ]
    input_ids, labels = tokenizer.encode_chat(messages)
    assert len(input_ids) == len(labels)
    # User tokens should be masked (-100)
    assert -100 in labels


def test_tokenizer_fim():
    tokenizer = UniversalCodingTokenizer()
    fim_str = tokenizer.format_fim("def foo():", "return 42", "    pass\n")
    assert "<|fim_prefix|>" in fim_str
    assert "<|fim_suffix|>" in fim_str
    assert "<|fim_middle|>" in fim_str


def test_model_forward_and_loss():
    config = UniversalCodingConfig.from_preset("educational_tiny", vocab_size=512)
    model = UniversalCodingForCausalLM(config)

    input_ids = torch.tensor([[1, 25, 50, 75, 100]], dtype=torch.long)
    labels = torch.tensor([[25, 50, 75, 100, 2]], dtype=torch.long)

    outputs = model(input_ids=input_ids, labels=labels)
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["loss"] is not None
    assert outputs["logits"].shape == (1, 5, 512)


def test_model_generation():
    config = UniversalCodingConfig.from_preset("educational_tiny", vocab_size=512)
    model = UniversalCodingForCausalLM(config)
    input_ids = torch.tensor([[1, 10, 20]], dtype=torch.long)
    gen = model.generate(input_ids, max_new_tokens=5, temperature=0.0)
    assert gen.shape[1] == 8


def test_lora_adaptation():
    config = UniversalCodingConfig.from_preset("educational_tiny", vocab_size=512)
    model = UniversalCodingForCausalLM(config)
    before_params = count_parameters(model)

    adapted = apply_lora_to_model(model, rank=8)
    assert adapted > 0
    after_params = count_parameters(model)
    assert after_params["trainable_parameters"] < before_params["trainable_parameters"]


def test_sandbox_security_boundary():
    sandbox = SandboxManager("sandbox/test_workspace")
    sandbox.write_file("test.txt", "safe content")
    assert sandbox.read_file("test.txt") == "safe content"

    # Directory traversal should raise PermissionError
    with pytest.raises(PermissionError):
        sandbox.write_file("../../unsafe.txt", "malicious")
