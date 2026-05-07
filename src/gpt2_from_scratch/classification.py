"""SMS spam classification fine-tuning utilities."""

import os
import urllib.request
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


SPAM_DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_and_unzip_spam_data(data_dir: str | Path, url: str = SPAM_DATA_URL) -> Path:
    """Download and extract the UCI SMS Spam Collection dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "sms_spam_collection.zip"
    extracted_path = data_dir / "sms_spam_collection"
    data_file_path = extracted_path / "SMSSpamCollection.tsv"

    if data_file_path.exists():
        return data_file_path

    with urllib.request.urlopen(url) as response:
        zip_path.write_bytes(response.read())

    extracted_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = extracted_path / "SMSSpamCollection"
    if original_file_path.exists():
        os.rename(original_file_path, data_file_path)

    return data_file_path


def load_sms_dataframe(data_file_path: str | Path):
    import pandas as pd

    return pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])


def create_balanced_dataset(df, random_state: int = 123):
    import pandas as pd

    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=random_state)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]])


def random_split(df, train_frac: float, validation_frac: float, random_state: int = 123):
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]


def prepare_spam_csvs(
    data_dir: str | Path = "data",
    train_frac: float = 0.7,
    validation_frac: float = 0.1,
):
    """Create balanced train/validation/test CSV files and return their paths."""
    data_dir = Path(data_dir)
    data_file_path = download_and_unzip_spam_data(data_dir)
    df = load_sms_dataframe(data_file_path)
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(
        balanced_df, train_frac, validation_frac
    )

    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"
    train_df.to_csv(train_path, index=None)
    validation_df.to_csv(val_path, index=None)
    test_df.to_csv(test_path, index=None)
    return train_path, val_path, test_path


class SpamDataset(Dataset):
    """Tokenized SMS spam dataset."""

    def __init__(self, csv_file: str | Path, tokenizer, max_length=None, pad_token_id=50256):
        import pandas as pd

        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        self.max_length = max_length or self._longest_encoded_length()
        self.encoded_texts = [
            encoded_text[: self.max_length] for encoded_text in self.encoded_texts
        ]
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index: int):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        return max(len(encoded_text) for encoded_text in self.encoded_texts)


def create_spam_dataloaders(
    tokenizer,
    data_dir: str | Path = "data",
    batch_size: int = 8,
    num_workers: int = 0,
):
    train_path, val_path, test_path = prepare_spam_csvs(data_dir)
    train_dataset = SpamDataset(train_path, tokenizer)
    val_dataset = SpamDataset(val_path, tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset(test_path, tokenizer, max_length=train_dataset.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_dataset.max_length


def configure_classifier_head(
    model,
    emb_dim: int,
    num_classes: int = 2,
    train_last_n_blocks: int = 1,
):
    """Freeze GPT and add a small classification head."""
    for param in model.parameters():
        param.requires_grad = False

    model.out_head = torch.nn.Linear(emb_dim, num_classes)

    if train_last_n_blocks > 0:
        for block in model.trf_blocks[-train_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    for param in model.final_norm.parameters():
        param.requires_grad = True

    return model


def calc_accuracy_loader(data_loader, model, device, num_batches=None) -> float:
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)
        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(data_loader, model, device, num_batches=None) -> float:
    if len(data_loader) == 0:
        return float("nan")

    total_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()

    return total_loss / num_batches


def evaluate_classifier(model, train_loader, val_loader, device, eval_iter: int):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_classifier(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        train_accuracy = calc_accuracy_loader(train_loader, model, device, eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, eval_iter)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        print(
            f"Training accuracy: {train_accuracy * 100:.2f}% | "
            f"Validation accuracy: {val_accuracy * 100:.2f}%"
        )

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_sms(text, model, tokenizer, device, max_length=None, pad_token_id=50256) -> str:
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    if max_length is None:
        max_length = min(len(input_ids), supported_context_length)
    else:
        max_length = min(max_length, supported_context_length)

    input_ids = input_ids[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"

