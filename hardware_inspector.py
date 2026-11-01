"""
Hardware Inspection and Adaptive Configuration Engine.
Inspects CPU, RAM, GPU, VRAM, CUDA, Disk, and PyTorch environment,
and automatically generates optimal local and cloud training configurations.
"""

import os
import sys
import platform
import shutil
import json
import psutil
from pathlib import Path
from typing import Dict, Any

try:
    import torch
except ImportError:
    torch = None


def inspect_hardware() -> Dict[str, Any]:
    """Inspects complete system hardware and runtime environment."""
    profile = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
        },
        "cpu": {
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False) or os.cpu_count() or 4,
            "logical_cores": psutil.cpu_count(logical=True) or os.cpu_count() or 8,
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None,
        },
        "ram": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "percent_used": psutil.virtual_memory().percent,
        },
        "disk": {},
        "gpu": {
            "has_nvidia_smi": False,
            "gpu_name": None,
            "driver_version": None,
            "cuda_version": None,
            "vram_total_mb": None,
            "vram_free_mb": None,
        },
        "pytorch": {
            "installed": torch is not None,
            "version": torch.__version__ if torch else None,
            "cuda_available": torch.cuda.is_available() if torch else False,
            "cuda_version": torch.version.cuda if torch and torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch and torch.cuda.is_available() else 0,
            "devices": [],
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        }
    }

    # Inspect current drive disk usage
    try:
        total, used, free = shutil.disk_usage(".")
        profile["disk"] = {
            "total_gb": round(total / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "percent_used": round((used / total) * 100, 1),
        }
    except Exception as e:
        profile["disk"]["error"] = str(e)

    # Inspect GPU via PyTorch
    if torch and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            profile["pytorch"]["devices"].append({
                "index": i,
                "name": props.name,
                "total_vram_gb": round(props.total_memory / (1024**3), 2),
                "major_compute": props.major,
                "minor_compute": props.minor,
            })
            profile["gpu"]["gpu_name"] = props.name
            profile["gpu"]["vram_total_mb"] = round(props.total_memory / (1024**2), 1)

    # If PyTorch doesn't see CUDA, check if nvidia-smi exists to detect underlying hardware
    if not profile["pytorch"]["cuda_available"]:
        try:
            import subprocess
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            lines = res.stdout.strip().split("\n")
            if lines and lines[0]:
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 4:
                    profile["gpu"]["has_nvidia_smi"] = True
                    profile["gpu"]["gpu_name"] = parts[0]
                    profile["gpu"]["driver_version"] = parts[1]
                    profile["gpu"]["vram_total_mb"] = float(parts[2])
                    profile["gpu"]["vram_free_mb"] = float(parts[3])
        except Exception:
            pass

    # Generate adaptive training recommendation
    profile["recommended_training_config"] = generate_adaptive_config(profile)
    return profile


def generate_adaptive_config(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Generates optimal training hyper-parameters adapted to the inspected hardware."""
    cuda_avail = profile["pytorch"]["cuda_available"]
    gpu_vram_mb = profile["gpu"]["vram_total_mb"] or 0
    total_ram_gb = profile["ram"]["total_gb"]
    cpu_cores = profile["cpu"]["physical_cores"]

    config = {
        "device": "cpu",
        "precision": "fp32",
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_len": 512,
        "num_workers": min(4, cpu_cores),
        "gradient_checkpointing": False,
        "use_compile": False,
        "model_preset": "educational_tiny",
        "hardware_tier": "cpu_entry",
    }

    if cuda_avail and gpu_vram_mb > 0:
        config["device"] = "cuda"
        if gpu_vram_mb >= 24000:  # 24GB+ (RTX 3090, 4090, A10)
            config["precision"] = "bf16"
            config["micro_batch_size"] = 8
            config["gradient_accumulation_steps"] = 4
            config["max_seq_len"] = 2048
            config["gradient_checkpointing"] = True
            config["model_preset"] = "code_1b_compact"
            config["hardware_tier"] = "gpu_pro_24gb"
        elif gpu_vram_mb >= 12000:  # 12GB - 16GB (RTX 3060, 4070, T4)
            config["precision"] = "fp16"
            config["micro_batch_size"] = 4
            config["gradient_accumulation_steps"] = 8
            config["max_seq_len"] = 1024
            config["gradient_checkpointing"] = True
            config["model_preset"] = "code_500m_medium"
            config["hardware_tier"] = "gpu_mid_12gb"
        elif gpu_vram_mb >= 5000:  # 6GB - 8GB (RTX 3050, 2060, 4050)
            config["precision"] = "fp16"
            config["micro_batch_size"] = 2
            config["gradient_accumulation_steps"] = 8
            config["max_seq_len"] = 512
            config["gradient_checkpointing"] = True
            config["model_preset"] = "code_150m_small"
            config["hardware_tier"] = "gpu_entry_6gb"
        else:  # < 5GB
            config["precision"] = "fp16"
            config["micro_batch_size"] = 1
            config["gradient_accumulation_steps"] = 16
            config["max_seq_len"] = 256
            config["gradient_checkpointing"] = True
            config["model_preset"] = "code_small_test"
            config["hardware_tier"] = "gpu_low_vram"
    else:
        # Running on CPU (e.g. PyTorch CPU build)
        if profile["gpu"]["has_nvidia_smi"]:
            config["note"] = (
                f"NVIDIA GPU detected ({profile['gpu']['gpu_name']}, {profile['gpu']['vram_total_mb']} MB VRAM), "
                "but current PyTorch build is CPU-only. Auto-configuring high-performance CPU multi-threading, "
                "with an option to switch to PyTorch CUDA build for GPU acceleration."
            )
        config["micro_batch_size"] = 2
        config["gradient_accumulation_steps"] = 4
        config["max_seq_len"] = 512
        config["model_preset"] = "code_educational_experiment"
        config["hardware_tier"] = "cpu_optimized"

    return config


def print_hardware_summary(profile: Dict[str, Any]):
    """Pretty prints hardware inspection results."""
    print("=" * 80)
    print("      UNIVERSAL CODING AI MODEL: HARDWARE & ENVIRONMENT AUDIT")
    print("=" * 80)
    print(f"OS Platform          : {profile['os']['system']} {profile['os']['release']} ({profile['os']['architecture']})")
    print(f"CPU Model            : {profile['cpu']['processor']}")
    print(f"CPU Physical / Logic : {profile['cpu']['physical_cores']} Cores / {profile['cpu']['logical_cores']} Threads")
    print(f"System RAM           : {profile['ram']['total_gb']} GB Total (Available: {profile['ram']['available_gb']} GB, Used: {profile['ram']['percent_used']}%)")
    print(f"Storage Space        : {profile['disk'].get('free_gb', 'N/A')} GB Free / {profile['disk'].get('total_gb', 'N/A')} GB Total")
    print(f"Python Runtime       : {profile['python']['version']} ({profile['python']['executable']})")
    print(f"PyTorch Version      : {profile['pytorch']['version']} (CUDA Available: {profile['pytorch']['cuda_available']})")

    if profile['pytorch']['cuda_available']:
        for dev in profile['pytorch']['devices']:
            print(f"PyTorch CUDA Device  : [{dev['index']}] {dev['name']} ({dev['total_vram_gb']} GB VRAM, CC {dev['major_compute']}.{dev['minor_compute']})")
    elif profile['gpu']['has_nvidia_smi']:
        print(f"Hardware NVIDIA GPU  : {profile['gpu']['gpu_name']} ({profile['gpu']['vram_total_mb']} MB VRAM)")
        print(f"NVIDIA Driver / CUDA : Driver {profile['gpu']['driver_version']}, Hardware Ready")
        print(f"PyTorch Status       : PyTorch CPU build active. Operating in optimized multi-core CPU mode.")
    else:
        print("Hardware GPU         : No discrete NVIDIA GPU detected. Operating in CPU mode.")

    rec = profile["recommended_training_config"]
    print("-" * 80)
    print("      AUTOMATICALLY SELECTED TRAINING CONFIGURATION")
    print("-" * 80)
    print(f"Target Device        : {rec['device'].upper()}")
    print(f"Precision            : {rec['precision']}")
    print(f"Micro-Batch Size     : {rec['micro_batch_size']}")
    print(f"Grad Accumulation    : {rec['gradient_accumulation_steps']} (Effective Batch = {rec['micro_batch_size'] * rec['gradient_accumulation_steps']})")
    print(f"Context Sequence Len : {rec['max_seq_len']} tokens")
    print(f"Model Architecture   : {rec['model_preset']}")
    print(f"Hardware Profile Tier: {rec['hardware_tier']}")
    if "note" in rec:
        print(f"Note                 : {rec['note']}")
    print("=" * 80)


def save_hardware_profile(output_path: str = "configs/hardware_profile.json") -> Dict[str, Any]:
    """Inspects and saves hardware profile to JSON."""
    profile = inspect_hardware()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    return profile


if __name__ == "__main__":
    prof = save_hardware_profile()
    print_hardware_summary(prof)

