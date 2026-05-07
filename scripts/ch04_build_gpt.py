"""Chapter 4: build a GPT model from reusable components."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt2_from_scratch.runtime import configure_runtime

configure_runtime()

import tiktoken
import torch

from gpt2_from_scratch.config import GPT_CONFIG_124M, TINY_GPT_CONFIG
from gpt2_from_scratch.generation import text_to_token_ids
from gpt2_from_scratch.model import GPTModel, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["tiny", "124m"], default="tiny")
    parser.add_argument("--prompt", default="Every effort moves you")
    args = parser.parse_args()

    config = TINY_GPT_CONFIG if args.config == "tiny" else GPT_CONFIG_124M
    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)
    model = GPTModel(config)
    token_ids = text_to_token_ids(args.prompt, tokenizer)
    logits = model(token_ids)

    print("Config:", args.config)
    print("Input shape:", tuple(token_ids.shape))
    print("Output shape:", tuple(logits.shape))
    print(f"Parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    main()
