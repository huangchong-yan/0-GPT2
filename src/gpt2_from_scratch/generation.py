"""Text generation helpers."""

import torch


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """Greedy decoding used in the early chapters."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Generate token IDs with optional top-k and temperature sampling."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_values = top_logits[:, [-1]]
            logits = torch.where(
                logits < min_values,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and torch.all(idx_next == eos_id):
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx

