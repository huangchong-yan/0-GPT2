"""Text sampling datasets for language-model pretraining."""

from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    """Sliding-window next-token prediction dataset."""

    def __init__(self, text: str, tokenizer, max_length: int, stride: int) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        for start in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[start : start + max_length]
            target_chunk = token_ids[start + 1 : start + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a GPT-style language-model dataloader."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def read_text(path: str | Path) -> str:
    """Read a UTF-8 text file."""
    return Path(path).read_text(encoding="utf-8")

