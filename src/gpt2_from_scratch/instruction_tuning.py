"""Instruction fine-tuning data pipeline."""

import json
import urllib.request
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


INSTRUCTION_DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)


def download_and_load_instruction_data(
    file_path: str | Path = "instruction-data.json",
    url: str = INSTRUCTION_DATA_URL,
) -> list[dict]:
    file_path = Path(file_path)
    if not file_path.exists():
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        file_path.write_text(text_data, encoding="utf-8")
    return json.loads(file_path.read_text(encoding="utf-8"))


def format_input(entry: dict) -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def split_instruction_data(
    data: list[dict],
    train_frac: float = 0.85,
    test_frac: float = 0.10,
) -> tuple[list[dict], list[dict], list[dict]]:
    train_portion = int(len(data) * train_frac)
    test_portion = int(len(data) * test_frac)
    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]
    return train_data, val_data, test_data


class InstructionDataset(Dataset):
    """Instruction-response pairs encoded as causal language-model examples."""

    def __init__(self, data: list[dict], tokenizer) -> None:
        self.data = data
        self.encoded_texts = []
        for entry in data:
            full_text = format_input(entry) + f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index: int):
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: int | None = None,
    device: str | torch.device = "cpu",
):
    """Pad variable-length SFT examples and mask repeated padding targets."""
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)

        pad_indices = torch.nonzero(targets == pad_token_id, as_tuple=True)[0]
        if pad_indices.numel() > 1:
            targets[pad_indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)


def create_instruction_dataloaders(
    data: list[dict],
    tokenizer,
    device,
    batch_size: int = 8,
    allowed_max_length: int = 1024,
    num_workers: int = 0,
):
    train_data, val_data, test_data = split_instruction_data(data)
    collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=allowed_max_length,
    )

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        InstructionDataset(test_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, (train_data, val_data, test_data)


def extract_response(generated_text: str, input_text: str) -> str:
    return generated_text[len(input_text) :].replace("### Response:", "").strip()

