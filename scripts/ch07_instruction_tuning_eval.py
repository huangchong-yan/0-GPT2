"""Chapter 7: instruction fine-tuning and Ollama-based evaluation."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt2_from_scratch.runtime import configure_runtime

configure_runtime()

import tiktoken
import torch
from tqdm import tqdm

from gpt2_from_scratch.checkpointing import load_model_state_compatible
from gpt2_from_scratch.config import get_model_config, model_size_from_name
from gpt2_from_scratch.evaluation import check_if_running, score_responses
from gpt2_from_scratch.generation import generate, text_to_token_ids, token_ids_to_text
from gpt2_from_scratch.gpt2_weights import download_and_load_gpt2, load_weights_into_gpt
from gpt2_from_scratch.instruction_tuning import (
    create_instruction_dataloaders,
    download_and_load_instruction_data,
    extract_response,
    format_input,
    split_instruction_data,
)
from gpt2_from_scratch.model import GPTModel
from gpt2_from_scratch.training import calc_loss_loader, train_language_model


def load_model(model_name: str, checkpoint: Path | None, device: torch.device):
    config = get_model_config(model_name, drop_rate=0.0, qkv_bias=True)
    model = GPTModel(config)

    if checkpoint and checkpoint.exists():
        remapped = load_model_state_compatible(model, checkpoint, device)
        model.to(device)
        print(f"Loaded SFT checkpoint: {checkpoint}")
        if remapped:
            print(f"Remapped {remapped} legacy attention checkpoint keys.")
        return model, config

    settings, params = download_and_load_gpt2(
        model_size_from_name(model_name),
        models_dir=ROOT / "gpt2",
    )
    print("Loaded GPT-2 settings:", settings)
    load_weights_into_gpt(model, params)
    model.to(device)
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["preview", "train", "generate", "generate-responses", "evaluate"],
        default="preview",
    )
    parser.add_argument("--data-file", default="instruction-data.json")
    parser.add_argument("--model-name", default="gpt2-small (124M)")
    parser.add_argument("--checkpoint", default="gpt2-small124M-sft.pth")
    parser.add_argument("--responses-file", default="instruction-data-with-response.json")
    parser.add_argument("--ollama-model", default="phi3")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    data = download_and_load_instruction_data(args.data_file)
    train_data, val_data, test_data = split_instruction_data(data)
    print(
        f"Dataset sizes: train={len(train_data)}, "
        f"val={len(val_data)}, test={len(test_data)}"
    )

    if args.mode == "preview":
        sample = data[50]
        print(format_input(sample) + f"\n\n### Response:\n{sample['output']}")
        return

    if args.mode == "evaluate":
        if not check_if_running("ollama"):
            raise RuntimeError("Ollama is not running. Start Ollama before evaluation.")

        responses_path = Path(args.responses_file)
        if not responses_path.exists():
            raise FileNotFoundError(
                f"Responses file not found: {responses_path}. "
                "Generate and save model responses first."
            )
        scored_data = json.loads(responses_path.read_text(encoding="utf-8"))
        scores = score_responses(
            scored_data,
            format_input,
            "model_response",
            args.ollama_model,
        )
        print(f"Number of scores: {len(scores)} of {len(scored_data)}")
        print(f"Average score: {sum(scores) / len(scores):.2f}")
        return

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    model, config = load_model(args.model_name, checkpoint, device)

    if args.mode == "train":
        train_loader, val_loader, _, splits = create_instruction_dataloaders(
            data,
            tokenizer,
            device,
            allowed_max_length=config["context_length"],
        )
        with torch.no_grad():
            train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        print(f"Initial train loss: {train_loss:.3f}")
        print(f"Initial validation loss: {val_loss:.3f}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
        _, val_split, _ = splits
        train_language_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=args.epochs,
            eval_freq=5,
            eval_iter=5,
            start_context=format_input(val_split[0]),
            tokenizer=tokenizer,
        )
        torch.save(model.state_dict(), checkpoint)
        print(f"Saved SFT checkpoint to {checkpoint}")
        return

    if args.mode == "generate":
        for entry in test_data[:3]:
            input_text = format_input(entry)
            token_ids = generate(
                model=model,
                idx=text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=256,
                context_size=config["context_length"],
                eos_id=50256,
                temperature=0.7,
                top_k=50,
            )
            generated_text = token_ids_to_text(token_ids, tokenizer)
            response_text = extract_response(generated_text, input_text)
            print(input_text)
            print(f"\nCorrect response:\n>> {entry['output']}")
            print(f"\nModel response:\n>> {response_text}")
            print("-------------------------------------")
        return

    if args.mode == "generate-responses":
        for index, entry in tqdm(enumerate(test_data), total=len(test_data)):
            input_text = format_input(entry)
            token_ids = generate(
                model=model,
                idx=text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=256,
                context_size=config["context_length"],
                eos_id=50256,
                temperature=0.7,
                top_k=50,
            )
            generated_text = token_ids_to_text(token_ids, tokenizer)
            test_data[index]["model_response"] = extract_response(
                generated_text,
                input_text,
            )

        responses_path = Path(args.responses_file)
        responses_path.write_text(
            json.dumps(test_data, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved model responses to {responses_path}")
        return


if __name__ == "__main__":
    main()
