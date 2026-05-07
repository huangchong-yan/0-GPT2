"""Model configuration helpers."""

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

TINY_GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

BASE_GPT2_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def get_model_config(
    model_name: str = "gpt2-small (124M)",
    *,
    context_length: int = 1024,
    drop_rate: float = 0.0,
    qkv_bias: bool = True,
) -> dict:
    """Return a GPT-2 compatible config for a named model size."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}")

    config = BASE_GPT2_CONFIG.copy()
    config.update(MODEL_CONFIGS[model_name])
    config.update(
        {
            "context_length": context_length,
            "drop_rate": drop_rate,
            "qkv_bias": qkv_bias,
        }
    )
    return config


def model_size_from_name(model_name: str) -> str:
    """Extract the OpenAI checkpoint size string, for example '124M'."""
    return model_name.split(" ")[-1].lstrip("(").rstrip(")")

