"""Chapter 5: train a small GPT or load OpenAI GPT-2 weights."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt2_from_scratch.runtime import configure_runtime

configure_runtime()

import tiktoken
import torch

from gpt2_from_scratch.config import TINY_GPT_CONFIG, get_model_config, model_size_from_name
from gpt2_from_scratch.data import create_dataloader_v1, read_text
from gpt2_from_scratch.generation import generate, text_to_token_ids, token_ids_to_text
from gpt2_from_scratch.gpt2_weights import download_and_load_gpt2, load_weights_into_gpt
from gpt2_from_scratch.model import GPTModel
from gpt2_from_scratch.training import train_language_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Every effort moves you")
    parser.add_argument("--text-file", default="the-verdict.txt")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--model-name", default="gpt2-small (124M)")
    parser.add_argument("--max-new-tokens", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    if args.pretrained:
        config = get_model_config(args.model_name, drop_rate=0.0, qkv_bias=True)
        settings, params = download_and_load_gpt2(
            model_size_from_name(args.model_name),
            models_dir=ROOT / "gpt2",
        )
        print("Loaded GPT-2 settings:", settings)
        model = GPTModel(config)
        load_weights_into_gpt(model, params)
    else:
        config = TINY_GPT_CONFIG
        model = GPTModel(config)

    model.to(device)

    if args.train:
        text = read_text(args.text_file)
        split_idx = int(0.9 * len(text))
        train_data, val_data = text[:split_idx], text[split_idx:]
        train_loader = create_dataloader_v1(
            train_data,
            batch_size=2,
            max_length=config["context_length"],
            stride=config["context_length"],
            shuffle=True,
        )
        val_loader = create_dataloader_v1(
            val_data,
            batch_size=2,
            max_length=config["context_length"],
            stride=config["context_length"],
            shuffle=False,
            drop_last=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
        train_language_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=1,
            eval_freq=5,
            eval_iter=5,
            start_context=args.prompt,
            tokenizer=tokenizer,
        )

    model.eval()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(args.prompt, tokenizer).to(device),
        max_new_tokens=args.max_new_tokens,
        context_size=config["context_length"],
        temperature=0.7 if args.pretrained else 0.0,
        top_k=50 if args.pretrained else None,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()
