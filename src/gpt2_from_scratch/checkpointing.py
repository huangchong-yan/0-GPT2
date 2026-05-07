"""Checkpoint loading helpers."""

from pathlib import Path

import torch


LEGACY_KEY_REPLACEMENTS = {
    ".att.W_query.": ".att.w_query.",
    ".att.W_key.": ".att.w_key.",
    ".att.W_value.": ".att.w_value.",
}


def remap_legacy_attention_keys(state_dict: dict) -> tuple[dict, int]:
    """Map old notebook-style attention parameter names to package names."""
    remapped = {}
    changed = 0

    for key, value in state_dict.items():
        new_key = key
        for old, new in LEGACY_KEY_REPLACEMENTS.items():
            if old in new_key:
                new_key = new_key.replace(old, new)
        if new_key != key:
            changed += 1
        remapped[new_key] = value

    return remapped, changed


def load_model_state_compatible(model, checkpoint_path: str | Path, device):
    """Load a model checkpoint, including legacy chapter-script checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    try:
        model.load_state_dict(state_dict)
        return 0
    except RuntimeError:
        remapped_state_dict, changed = remap_legacy_attention_keys(state_dict)
        if changed == 0:
            raise
        model.load_state_dict(remapped_state_dict)
        return changed

