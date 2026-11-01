"""
PyTorch Decoder-Only Causal Transformer Architecture.
Features:
- Rotary Position Embeddings (RoPE)
- Root Mean Square Normalization (RMSNorm)
- Grouped-Query Attention (GQA) & Scaled Dot-Product Attention (SDPA)
- SwiGLU Activation Feed-Forward Network
- Pre-Norm Residual Connections
- Autoregressive Causal Masking & KV Caching
- Tied / Untied LM Output Head
"""

import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.configuration import UniversalCodingConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., RoFormer)."""

    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Inverse frequency calculation
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(self.max_position_embeddings)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len, :].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len, :].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary embeddings to query and key tensors."""
    # cos, sin shape: [seq_len, dim] -> broadcast to [batch, heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key and value heads for Grouped Query Attention (GQA)."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class UniversalAttention(nn.Module):
    """Grouped-Query Attention with RoPE and SDPA Flash-Attention optimization."""

    def __init__(self, config: UniversalCodingConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV Caching for autoregressive generation
        if past_key_value is not None:
            past_k, past_v = past_key_value
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for Grouped Query Attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Scaled Dot-Product Attention (SDPA with causal masking)
        is_causal = attention_mask is None and past_key_value is None and seq_len > 1
        dropout_p = self.attention_dropout if self.training else 0.0

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, present_key_value


class UniversalMLP(nn.Module):
    """SwiGLU feed-forward layer (Shazeer, 2020)."""

    def __init__(self, config: UniversalCodingConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class UniversalDecoderLayer(nn.Module):
    """Single Transformer Decoder block with Pre-RMSNorm and Residual Connections."""

    def __init__(self, config: UniversalCodingConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = UniversalAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = UniversalMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-Norm Attention Residual
        residual = hidden_states
        normed_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            normed_states,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # Pre-Norm MLP Residual
        residual = hidden_states
        normed_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_states)
        hidden_states = residual + mlp_output

        return hidden_states, present_key_value


class UniversalCodingModel(nn.Module):
    """Core Transformer decoder stack."""

    def __init__(self, config: UniversalCodingConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList([UniversalDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        # Compute RoPE frequencies for current sequence length (offset by cache length if present)
        cache_offset = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        total_len = cache_offset + seq_len
        cos, sin = self.rotary_emb(hidden_states, total_len)
        cos = cos[cache_offset:total_len]
        sin = sin[cache_offset:total_len]

        next_decoder_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache=False)
                    return custom_forward

                hidden_states, _ = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    (cos, sin),
                    attention_mask,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    rotary_emb=(cos, sin),
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
                if use_cache:
                    next_decoder_cache.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache


class UniversalCodingForCausalLM(nn.Module):
    """Universal Coding Model with Autoregressive Language Modeling Head."""

    def __init__(self, config: UniversalCodingConfig):
        super().__init__()
        self.config = config
        self.model = UniversalCodingModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        """Initializes weights using truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=self.config.initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.config.initializer_range)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten tokens for cross-entropy
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_cache,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching, top-k/top-p sampling, and repetition penalty."""
        self.eval()
        eos_token_id = eos_token_id or self.config.eos_token_id
        pad_token_id = pad_token_id or self.config.pad_token_id

        past_key_values = None
        generated = input_ids.clone()
        curr_input_ids = input_ids

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs["logits"][:, -1, :]  # [batch, vocab_size]
            past_key_values = outputs["past_key_values"]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    for prev_token in set(generated[b].tolist()):
                        if logits[b, prev_token] > 0:
                            logits[b, prev_token] /= repetition_penalty
                        else:
                            logits[b, prev_token] *= repetition_penalty

            # Temperature scaling & sampling
            if temperature > 0:
                logits = logits / temperature

                # Top-K filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                # Top-P (Nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    for b in range(logits.shape[0]):
                        indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                        logits[b, indices_to_remove] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            curr_input_ids = next_token

            # Check if all batches produced EOS
            if (next_token == eos_token_id).all():
                break

        return generated
