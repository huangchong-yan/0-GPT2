"""Chapter 2: tokenization and sliding-window data loading."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt2_from_scratch.runtime import configure_runtime

configure_runtime()

from gpt2_from_scratch.data import create_dataloader_v1, read_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", default="the-verdict.txt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    text = read_text(args.text_file)
    dataloader = create_dataloader_v1(
        text,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=False,
    )

    input_batch, target_batch = next(iter(dataloader))
    print("Input IDs:\n", input_batch)
    print("\nTarget IDs:\n", target_batch)
    print("\nInput shape:", tuple(input_batch.shape))


if __name__ == "__main__":
    main()
