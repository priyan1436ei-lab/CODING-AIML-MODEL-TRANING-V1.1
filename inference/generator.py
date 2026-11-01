"""
Local Inference Engine and Code Generation Pipeline.
Provides streaming generation, FIM code completion, conversational coding chat,
and algorithmic problem solving.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Tuple

import torch

from model.architecture import UniversalCodingForCausalLM
from model.modeling import load_checkpoint
from tokenizer.tokenizer import UniversalCodingTokenizer


class CodeGenerator:
    """Production Inference Manager for Universal Coding Model."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/sft/final",
        device: str = "cpu",
    ):
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Auto-detect checkpoint if default does not exist
        if not Path(checkpoint_dir).exists():
            for cand in ["checkpoints/dpo/final", "checkpoints/agentic/final", "checkpoints/pretrain/final"]:
                if Path(cand).exists():
                    self.checkpoint_dir = cand
                    break

        print(f"Loading Inference Model from: {self.checkpoint_dir} on {self.device.upper()}")
        self.model, self.config, self.meta = load_checkpoint(self.checkpoint_dir, device=self.device)
        self.model.eval()

        self.tokenizer = UniversalCodingTokenizer.from_pretrained(self.checkpoint_dir)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generates completion text for a prompt."""
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        prompt_len = input_ids.shape[1]

        gen_tokens = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        new_tokens = gen_tokens[0, prompt_len:].tolist()
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Generator[str, None, None]:
        """Streams generated tokens one by one as they are produced."""
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        past_key_values = None
        curr_input_ids = input_ids
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=curr_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]

                if repetition_penalty != 1.0:
                    for b in range(generated.shape[0]):
                        for prev_t in set(generated[b].tolist()):
                            if logits[b, prev_t] > 0:
                                logits[b, prev_t] /= repetition_penalty
                            else:
                                logits[b, prev_t] *= repetition_penalty

                if temperature > 0:
                    logits = logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float("Inf")
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                tok_id = next_token.item()
                if tok_id == self.tokenizer.eos_token_id:
                    break

                generated = torch.cat([generated, next_token], dim=-1)
                curr_input_ids = next_token

                yield self.tokenizer.decode([tok_id], skip_special_tokens=True)

    def complete_fim(
        self,
        prefix: str,
        suffix: str,
        max_new_tokens: int = 128,
        mode: str = "PSM",
    ) -> str:
        """Fill-In-The-Middle code completion."""
        fim_prompt = self.tokenizer.format_fim(prefix, suffix, "", mode=mode)
        # Strip trailing eos from prompt to let model fill middle
        if fim_prompt.endswith("<|eos|>"):
            fim_prompt = fim_prompt[:-7]
        return self.generate(fim_prompt, max_new_tokens=max_new_tokens, temperature=0.2)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.5,
    ) -> str:
        """Multi-turn conversational code generation using ChatML template."""
        input_ids, _ = self.tokenizer.encode_chat(messages, add_generation_prompt=True)
        inp_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        prompt_len = inp_tensor.shape[1]

        gen_tokens = self.model.generate(
            input_ids=inp_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        new_tokens = gen_tokens[0, prompt_len:].tolist()
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

