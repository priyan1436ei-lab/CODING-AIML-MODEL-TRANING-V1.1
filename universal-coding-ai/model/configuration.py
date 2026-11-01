"""
Universal Coding Model Configuration.
Provides comprehensive hyperparameters, architectural dimensions,
and built-in presets for both custom architectures and pretrained model fine-tuning.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class UniversalCodingConfig:
    """Configuration class for Universal Coding AI Transformer Model."""

    # Architectural Hyperparameters
    vocab_size: int = 32000
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 4  # Grouped Query Attention (GQA)
    intermediate_size: int = 1376  # SwiGLU multiplier ~ 8/3 * hidden_size
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    use_cache: bool = True

    # Special Token IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    fim_prefix_id: int = 3
    fim_suffix_id: int = 4
    fim_middle_id: int = 5

    # Strategy & Provenance
    model_strategy: str = "custom_transformer"  # 'custom_transformer' or 'pretrained_adapter'
    pretrained_model_name_or_path: Optional[str] = None
    torch_dtype: str = "float32"
    model_version: str = "1.0.0"

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )
        if self.intermediate_size is None:
            # Standard SwiGLU 8/3 approximation
            self.intermediate_size = int(2 * (4 * self.hidden_size) / 3)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_pretrained(self, save_directory: str):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        config_file = path / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalCodingConfig":
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "UniversalCodingConfig":
        config_file = Path(save_directory) / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "UniversalCodingConfig":
        """Load a predefined configuration profile."""
        presets = {
            # Educational & Smoke Test (~12M params, runs instantly on any CPU)
            "educational_tiny": cls(
                vocab_size=8192,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=512,
                max_position_embeddings=512,
                tie_word_embeddings=True,
            ),
            # Micro Coding Model (~32M params, fast training on CPU & entry GPU)
            "code_micro": cls(
                vocab_size=16384,
                hidden_size=384,
                num_hidden_layers=6,
                num_attention_heads=6,
                num_key_value_heads=2,
                intermediate_size=1024,
                max_position_embeddings=1024,
                tie_word_embeddings=True,
            ),
            # Small Coding Foundation Model (~125M params, fits 6GB VRAM or multi-core CPU)
            "code_125m": cls(
                vocab_size=32000,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                num_key_value_heads=4,
                intermediate_size=2048,
                max_position_embeddings=2048,
                tie_word_embeddings=True,
            ),
            # Medium Coding Model (~350M params)
            "code_350m": cls(
                vocab_size=32000,
                hidden_size=1024,
                num_hidden_layers=16,
                num_attention_heads=16,
                num_key_value_heads=4,
                intermediate_size=2816,
                max_position_embeddings=2048,
                tie_word_embeddings=True,
            ),
            # 1B Coding Assistant (Chinchilla-optimal for production deployment)
            "code_1b": cls(
                vocab_size=32000,
                hidden_size=2048,
                num_hidden_layers=24,
                num_attention_heads=16,
                num_key_value_heads=4,
                intermediate_size=5632,
                max_position_embeddings=4096,
                tie_word_embeddings=False,
            ),
        }

        if preset_name not in presets:
            raise KeyError(f"Unknown preset '{preset_name}'. Choose from: {list(presets.keys())}")

        cfg = presets[preset_name]
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
