"""
Universal Coding AI Model - Production API Server Launcher.

Usage:
  python serve.py
  python serve.py --host 0.0.0.0 --port 8000 --checkpoint checkpoints/sft/final
"""

import argparse
import uvicorn
from api.server import app, generator
from inference.generator import CodeGenerator
import api.server as server_module


def main():
    parser = argparse.ArgumentParser(description="Universal Coding Model Production API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft/final", help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Target device (cpu or cuda)")

    args = parser.parse_args()

    # Pre-initialize generator
    try:
        server_module.generator = CodeGenerator(checkpoint_dir=args.checkpoint, device=args.device)
        print(f"API Server preloaded model from: {args.checkpoint}")
    except Exception as e:
        print(f"Notice: Model will lazy-load on first request ({e})")

    print("=" * 80)
    print(f"      STARTING UNIVERSAL CODING MODEL API SERVER")
    print(f"      Listening on: http://{args.host}:{args.port}")
    print(f"      Interactive API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 80)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

