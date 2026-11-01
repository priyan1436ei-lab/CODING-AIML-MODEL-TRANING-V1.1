"""
Universal Coding AI Model - Interactive Inference CLI.

Usage:
  python inference.py
  python inference.py --prompt "def quick_sort(arr):"
  python inference.py --fim-prefix "def is_even(n: int) -> bool:\n" --fim-suffix "\n    return res"
"""

import argparse
import sys
from pathlib import Path

from inference.generator import CodeGenerator


def run_interactive_session(generator: CodeGenerator):
    """Runs interactive command line coding session."""
    print("=" * 80)
    print("      UNIVERSAL CODING AI MODEL: INTERACTIVE PROGRAMMING TERMINAL")
    print("=" * 80)
    print("Commands:")
    print("  /fim   - Switch to Fill-In-The-Middle mode")
    print("  /chat  - Multi-turn Conversational Coding")
    print("  /exit  - Quit terminal\n")

    mode = "chat"
    history = []

    while True:
        try:
            user_input = input(f"[{mode.upper()}] >>> ").strip()
            if not user_input:
                continue

            if user_input in ["/exit", "exit", "quit"]:
                print("Exiting Universal Coding AI terminal. Happy coding!")
                break

            elif user_input == "/fim":
                print("\n--- Fill-In-The-Middle Mode ---")
                prefix = input("Enter Code Prefix:\n")
                suffix = input("Enter Code Suffix:\n")
                print("\nGenerating middle completion...")
                res = generator.complete_fim(prefix, suffix)
                print("-" * 40)
                print(f"{prefix}\n[FILLED MIDDLE]:\n{res}\n{suffix}")
                print("-" * 40 + "\n")
                continue

            elif user_input == "/chat":
                mode = "chat"
                history = []
                print("Switched to multi-turn coding chat. Conversation reset.\n")
                continue

            # Standard chat/completion
            history.append({"role": "user", "content": user_input})
            print("\nUniversal Model Thinking & Generating:\n" + "-" * 50)
            sys.stdout.flush()

            # Stream response to stdout
            full_response = []
            for chunk in generator.stream_generate(user_input, max_new_tokens=256, temperature=0.7):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                full_response.append(chunk)

            print("\n" + "-" * 50 + "\n")
            history.append({"role": "assistant", "content": "".join(full_response)})

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"Error during generation: {e}")


def main():
    parser = argparse.ArgumentParser(description="Universal Coding Model Inference CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft/final", help="Path to checkpoint directory")
    parser.add_argument("--prompt", type=str, default=None, help="One-shot prompt to generate code for")
    parser.add_argument("--fim-prefix", type=str, default=None, help="Prefix code for Fill-In-The-Middle")
    parser.add_argument("--fim-suffix", type=str, default=None, help="Suffix code for Fill-In-The-Middle")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cpu", help="Target device (cpu or cuda)")

    args = parser.parse_args()

    generator = CodeGenerator(checkpoint_dir=args.checkpoint, device=args.device)

    if args.fim_prefix and args.fim_suffix:
        print(f"Executing Fill-In-The-Middle Completion...")
        middle = generator.complete_fim(args.fim_prefix, args.fim_suffix, max_new_tokens=args.max_tokens)
        print("Generated Middle Code:")
        print(middle)

    elif args.prompt:
        print(f"Generating for prompt: {args.prompt}\n" + "-" * 50)
        for chunk in generator.stream_generate(args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print("\n" + "-" * 50)

    else:
        run_interactive_session(generator)


if __name__ == "__main__":
    main()
