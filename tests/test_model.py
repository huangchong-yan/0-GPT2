import torch

from gpt2_from_scratch.config import TINY_GPT_CONFIG
from gpt2_from_scratch.generation import generate_text_simple
from gpt2_from_scratch.model import GPTModel


def test_gpt_model_forward_shape():
    config = TINY_GPT_CONFIG.copy()
    config.update({"context_length": 8, "emb_dim": 32, "n_heads": 4, "n_layers": 2})
    model = GPTModel(config)
    input_ids = torch.randint(0, config["vocab_size"], (2, 5))

    logits = model(input_ids)

    assert logits.shape == (2, 5, config["vocab_size"])


def test_generate_text_simple_appends_tokens():
    config = TINY_GPT_CONFIG.copy()
    config.update({"context_length": 8, "emb_dim": 32, "n_heads": 4, "n_layers": 1})
    model = GPTModel(config)
    input_ids = torch.randint(0, config["vocab_size"], (1, 3))

    output_ids = generate_text_simple(
        model,
        input_ids,
        max_new_tokens=2,
        context_size=config["context_length"],
    )

    assert output_ids.shape == (1, 5)

