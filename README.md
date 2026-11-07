# Universal Coding AI Model (Decoder-Only Transformer LLM)
## Advanced AI/ML Programming Foundation Model & Autonomous Engineering Runtime

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python)](https://python.org/)

The **Universal Coding AI Model** is a programming-specialized Artificial Intelligence foundation model and autonomous software engineering runtime. Built from foundational PyTorch primitives, it features a modern decoder-only causal Transformer architecture with multi-stage training: Causal Pretraining, Instruction-Tuning (SFT), Direct Preference Optimization (DPO), and Autonomous Agentic trajectory training.

---

## 1. Architectural Specifications

The custom Transformer architecture implements modern innovations found in frontier coding models (Llama 3, Qwen 2.5, DeepSeek V2/V3):

```
Input Tokens: [x_1, x_2, ..., x_t]
     │
     ▼
Token Embeddings (vocab_size -> hidden_size)
     │
     ▼
┌───────────────────────────────────────────────┐
│ Transformer Decoder Layer (x N Layers)        │
│                                               │
│  x ──► RMSNorm ──► GQA Attention (RoPE + SDPA)│
│  │                     │                      │
│  └──────────────────(+) Residual              │
│                        │                      │
│  x' ─► RMSNorm ──► SwiGLU MLP (Gate * Up)     │
│  │                     │                      │
│  └──────────────────(+) Residual              │
└───────────────────────────────────────────────┘
     │
     ▼
Final RMSNorm
     │
     ▼
LM Head (hidden_size -> vocab_size) [Tied Embeddings]
     │
     ▼
Next-Token Logits / Cross-Entropy Loss
```

### Core Architecture Components:
- **Rotary Position Embeddings (RoPE)**: Dynamic frequency precomputation supporting extended context windows with zero position-degradation.
- **Root Mean Square Normalization (RMSNorm)**: Pre-layer normalization with numerical stabilization ($\epsilon=10^{-6}$).
- **Grouped-Query Attention (GQA)**: Configurable query heads to key-value heads ratio for high inference throughput and reduced KV-cache memory consumption.
- **Scaled Dot-Product Attention (SDPA)**: Kernel-level accelerated causal masking.
- **SwiGLU Activation**: Gated feed-forward network ($\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}) \otimes xW_{\text{up}}) W_{\text{down}}$).
- **KV Caching**: Fast autoregressive decoding caching past keys and values.
- **Weight Tying**: Shares token embeddings with the output projection head to maximize parameter efficiency.

### Architectural Presets:
| Preset Name | Hidden Size | Layers | Heads | KV Heads | Intermediate | Context | Approx Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `educational_tiny` | 256 | 4 | 4 | 2 | 512 | 512 | 2.5M |
| `code_micro` | 384 | 6 | 6 | 2 | 1024 | 1024 | 32M |
| `code_125m` | 768 | 12 | 12 | 4 | 2048 | 2048 | 125M |
| `code_350m` | 1024 | 16 | 16 | 4 | 2816 | 2048 | 350M |
| `code_1b` | 2048 | 24 | 16 | 4 | 5632 | 4096 | 1.1B |

---

## 2. Directory Structure

```
universal-coding-ai/
│
├── configs/
│   ├── hardware_profile.json          # Inspected hardware profile & auto-tuning
│   ├── pretraining_config.json        # Stage 1: Causal pretraining config
│   ├── sft_config.json                # Stage 2: Supervised fine-tuning config
│   ├── dpo_config.json                # Stage 3: Direct preference optimization config
│   ├── agentic_config.json            # Stage 4: Agentic trajectory config
│   └── cloud_gpu_scaling.json         # A100/H100/RTX4090 cluster scaling presets
│
├── data/
│   ├── loader.py                      # Streaming data loaders & PyTorch datasets
│   └── validator.py                   # Schema, license, duplicate, and quality audit
│
├── tokenizer/
│   └── tokenizer.py                   # Programming BPE, FIM, ChatML, and keyword coverage
│
├── model/
│   ├── architecture.py                # PyTorch RoPE, RMSNorm, GQA, SwiGLU Causal Transformer
│   ├── configuration.py               # Config dataclass, presets, and serialization
│   └── modeling.py                   # Checkpoint I/O, Safetensors, and LoRA adapters
│
├── training/
│   ├── trainer.py                     # AdamW, Cosine warmup scheduler, PPL, metrics
│   ├── pretraining.py                 # Causal next-token prediction pretraining loop
│   ├── sft_training.py                # SFT with user-token loss masking
│   ├── dpo_training.py                # Direct Preference Optimization (Rafailov et al.)
│   └── agentic_training.py            # Autonomous software engineering trajectories
│
├── evaluation/
│   └── evaluator.py                   # HumanEval, MBPP, AST syntax check, Pass@1
│
├── checkpoints/                       # Model weights, config, and optimizer checkpoints
├── inference/
│   └── generator.py                   # Streaming, FIM completion, conversational chat
│
├── agent/
│   └── autonomous_engineer.py         # Full software engineering lifecycle agent
│
├── tools/
│   └── workspace_tools.py             # FileSystem, Terminal, TestRunner, and Git tools
│
├── sandbox/
│   ├── sandbox_manager.py             # Security boundary & traversal prevention
│   └── workspace/                     # Isolated runtime execution workspace
│
├── api/
│   └── server.py                      # FastAPI server with OpenAI-compatible endpoints
│
├── reports/                           # Data validation & evaluation benchmark reports
├── tests/                             # Unit tests for architecture, tokenizer, sandbox
│
├── train.py                           # Master training & inspection orchestrator
├── evaluate.py                        # Benchmark evaluation CLI
├── inference.py                       # Interactive terminal & one-shot inference
├── serve.py                           # Production FastAPI server launcher
├── requirements.txt                   # Production dependencies
└── README.md                          # Documentation
```

---

## 3. Execution Commands

All commands can be executed directly from the project directory:

```bash
# 1. Hardware Audit & Environment Inspection
python train.py --check-hardware

# 2. Dataset Validation & Quality Audit
python train.py --validate-data

# 3. Model Weight Initialization & Smoke Test
python train.py --initialize-model

# 4. Comprehensive End-to-End Pipeline Smoke Test
python train.py --smoke-test

# 5. Stage 1: Causal Pretraining
python train.py --stage pretrain --epochs 2 --grad-accum 2

# 6. Stage 2: Supervised Fine-Tuning (SFT)
python train.py --stage sft --epochs 3 --grad-accum 2

# 7. Stage 3: Direct Preference Optimization (DPO)
python train.py --stage dpo --epochs 2

# 8. Stage 4: Autonomous Agentic Trajectory Training
python train.py --stage agentic --epochs 2

# 9. Independent Benchmark Evaluation
python evaluate.py --all --num-samples 20

# 10. Interactive Coding Inference
python inference.py

# 11. Production OpenAI-Compatible API Server
python serve.py --host 0.0.0.0 --port 8000
```

---

## 4. The 4-Stage Progressive Training Pipeline

### Stage 1: Causal Language Model Pretraining
- **Objective**: Next-token prediction on multi-language code and documentation.
- **Loss**:
  $$\mathcal{L}_{\text{pretrain}} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$
- **Features**: Sequence packing, gradient accumulation, cosine learning rate warmup, gradient clipping ($1.0$).

### Stage 2: Supervised Fine-Tuning (SFT)
- **Objective**: Instruction alignment for requirements-to-code, debugging, and translation.
- **Chat Template**: ChatML format (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`).
- **Prompt Token Masking**: Calculates loss exclusively on assistant response tokens (user tokens set to label `-100`).
- **PEFT / LoRA**: Optional Low-Rank Adapters ($W + \frac{\alpha}{r}BA$) applied to attention projection matrices.

### Stage 3: Direct Preference Optimization (DPO)
- **Objective**: Preference alignment to favor clean, secure, idiomatic solutions over buggy or insecure code.
- **Loss Formulation**:
  $$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$
- **Implicit Reward**: Dynamically measures and maximizes the margin $r(x, y_w) - r(x, y_l)$.

### Stage 4: Autonomous Agentic Coding
- **Objective**: Autonomous multi-step software development with tool invocation.
- **Structured Tool Tokens**: `<|tool_call|>{"name": ..., "arguments": ...}<|tool_result|>`.
- **Reasoning Format**: `Thought -> Tool Call -> Observation -> Reflection -> Code Repair -> Final Solution`.

---

## 5. Autonomous Software Engineer Workflow

The model powers an autonomous agent operating inside a secure sandbox runtime:

```
User Requirement
     │
     ▼
[Step 1] Requirement Analysis & Architecture Design
     │
     ▼
[Step 2] Multi-file Code Generation
     │
     ▼
[Step 3] Sandboxed File Creation (`write_file`)
     │
     ▼
[Step 4] Automated Test Suite Creation (`test_app.py`)
     │
     ▼
[Step 5] Execution & Error Detection (`run_tests`)
     │
     ├──► [Test Passes] ──────────────────────────┐
     │                                            │
     └──► [Error Detected]                        ▼
               │                            [Step 6] Delivery
               ▼                            Verified Application
          Self-Repair & Debugging Loop
```

---

## 6. Production API Endpoints

The model exposes OpenAI-compatible REST and streaming endpoints:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/health` | `GET` | Health status, active checkpoint, device, and parameter count |
| `/v1/models` | `GET` | OpenAI-compatible model listing |
| `/v1/chat/completions` | `POST` | Chat completions with optional Server-Sent Events (SSE) streaming |
| `/v1/completions` | `POST` | Text completion and Fill-In-The-Middle (FIM) |
| `/generate` | `POST` | Low-latency direct generation endpoint |

---

## 7. Cloud GPU Scaling Configurations

Preconfigured configurations are available in `configs/cloud_gpu_scaling.json` for scaling up to multi-billion parameter models on NVIDIA A100, H100, and RTX 4090 GPUs.

# CODING-AIML-MODEL-TRANING-V1.1
# Universal-Coding-AI-Model-Decoder-Only-Transformer-LLM-
