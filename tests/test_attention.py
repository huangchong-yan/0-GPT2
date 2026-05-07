import torch

from gpt2_from_scratch.attention import MultiHeadAttention


def test_multi_head_attention_output_shape():
    torch.manual_seed(123)
    batch = torch.randn(2, 6, 8)
    attention = MultiHeadAttention(
        d_in=8,
        d_out=12,
        context_length=6,
        num_heads=3,
        dropout=0.0,
    )

    output = attention(batch)

    assert output.shape == (2, 6, 12)

