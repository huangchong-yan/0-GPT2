import torch

from gpt2_from_scratch.instruction_tuning import custom_collate_fn, format_input


def test_format_input_includes_optional_input():
    entry = {
        "instruction": "Translate the sentence.",
        "input": "Hello",
        "output": "Bonjour",
    }

    formatted = format_input(entry)

    assert "### Instruction:" in formatted
    assert "Translate the sentence." in formatted
    assert "### Input:" in formatted
    assert "Hello" in formatted


def test_custom_collate_masks_repeated_padding_targets():
    batch = [[1, 2, 3], [4]]
    _, targets = custom_collate_fn(batch, pad_token_id=0, ignore_index=-100)

    assert targets.shape == (2, 3)
    assert torch.equal(targets[0], torch.tensor([2, 3, 0]))
    assert torch.equal(targets[1], torch.tensor([0, -100, -100]))
