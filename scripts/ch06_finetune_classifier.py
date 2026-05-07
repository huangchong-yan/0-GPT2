"""Chapter 6: fine-tune GPT-2 for SMS spam classification."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt2_from_scratch.runtime import configure_runtime

configure_runtime()

import tiktoken
import torch

from gpt2_from_scratch.classification import (
    calc_accuracy_loader,
    classify_sms,
    configure_classifier_head,
    create_spam_dataloaders,
    prepare_spam_csvs,
    train_classifier,
)
from gpt2_from_scratch.checkpointing import load_model_state_compatible
from gpt2_from_scratch.config import get_model_config, model_size_from_name
from gpt2_from_scratch.gpt2_weights import download_and_load_gpt2, load_weights_into_gpt
from gpt2_from_scratch.model import GPTModel


def build_classifier_model(
    model_name: str,
    checkpoint_path: Path | None,
    device: torch.device,
    *,
    load_pretrained: bool,
):
    config = get_model_config(model_name, drop_rate=0.0, qkv_bias=True)
    model = GPTModel(config)
    model = configure_classifier_head(model, emb_dim=config["emb_dim"])

    if checkpoint_path and checkpoint_path.exists() and not load_pretrained:
        remapped = load_model_state_compatible(model, checkpoint_path, device)
        model.to(device)
        print(f"Loaded classifier checkpoint: {checkpoint_path}")
        if remapped:
            print(f"Remapped {remapped} legacy attention checkpoint keys.")
        return model, config

    settings, params = download_and_load_gpt2(
        model_size_from_name(model_name),
        models_dir=ROOT / "gpt2",
    )
    print("Loaded GPT-2 settings:", settings)

    base_model = GPTModel(config)
    load_weights_into_gpt(base_model, params)
    model = configure_classifier_head(base_model, emb_dim=config["emb_dim"])
    model.to(device)
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "train", "infer"], default="prepare")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-name", default="gpt2-small (124M)")
    parser.add_argument("--checkpoint", default="review_classifier.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--text", default="You are a winner selected to receive cash.")
    args = parser.parse_args()

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "prepare":
        train_path, val_path, test_path = prepare_spam_csvs(args.data_dir)
        print("Prepared:", train_path, val_path, test_path)
        return

    checkpoint_path = Path(args.checkpoint)
    if args.mode == "infer" and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run --mode train first."
        )

    train_loader, val_loader, test_loader, max_length = create_spam_dataloaders(
        tokenizer,
        data_dir=args.data_dir,
    )
    model, _ = build_classifier_model(
        args.model_name,
        checkpoint_path,
        device,
        load_pretrained=args.mode == "train",
    )

    if args.mode == "train":
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
        train_classifier(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=args.epochs,
            eval_freq=50,
            eval_iter=5,
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved classifier checkpoint to {checkpoint_path}")
    elif not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run --mode train first."
        )

    train_acc = calc_accuracy_loader(train_loader, model, device)
    val_acc = calc_accuracy_loader(val_loader, model, device)
    test_acc = calc_accuracy_loader(test_loader, model, device)
    print(f"Training accuracy: {train_acc * 100:.2f}%")
    print(f"Validation accuracy: {val_acc * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("Prediction:", classify_sms(args.text, model, tokenizer, device, max_length))


if __name__ == "__main__":
    main()
