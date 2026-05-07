"""Chapter 3: causal multi-head attention demo."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt2_from_scratch.runtime import configure_runtime

configure_runtime()

import torch

from gpt2_from_scratch.attention import MultiHeadAttention


def main() -> None:
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ]
    )
    batch = torch.stack((inputs, inputs), dim=0)

    torch.manual_seed(123)
    attention = MultiHeadAttention(
        d_in=3,
        d_out=4,
        context_length=batch.shape[1],
        num_heads=2,
        dropout=0.0,
    )
    context = attention(batch)
    print("Context vectors:\n", context)
    print("\nContext shape:", tuple(context.shape))


if __name__ == "__main__":
    main()
