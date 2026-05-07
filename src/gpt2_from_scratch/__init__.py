"""Reusable GPT-2-from-scratch components."""

from .runtime import configure_runtime

configure_runtime()

from .config import GPT_CONFIG_124M, MODEL_CONFIGS, TINY_GPT_CONFIG, get_model_config
from .model import GPTModel

__all__ = [
    "GPTModel",
    "GPT_CONFIG_124M",
    "MODEL_CONFIGS",
    "TINY_GPT_CONFIG",
    "get_model_config",
]
