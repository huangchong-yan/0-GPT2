# Copyright (c) Sebastian Raschka under Apache License 2.0.
# Adapted from: https://github.com/rasbt/LLMs-from-scratch

"""Download OpenAI GPT-2 checkpoints and load them into the local GPT model."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


def download_and_load_gpt2(model_size: str, models_dir: str | Path):
    """Download GPT-2 TensorFlow checkpoint files and return settings/params."""
    import requests
    import tensorflow as tf
    from tqdm import tqdm

    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"model_size must be one of {allowed_sizes}")

    model_dir = Path(models_dir) / model_size
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        file_url = f"{base_url}/{model_size}/{filename}"
        backup_url = f"{backup_base_url}/{model_size}/{filename}"
        file_path = model_dir / filename
        _download_file(file_url, file_path, backup_url, requests, tqdm)

    tf_ckpt_path = tf.train.latest_checkpoint(str(model_dir))
    settings = json.loads((model_dir / "hparams.json").read_text(encoding="utf-8"))
    params = _load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings, tf)
    return settings, params


def _download_file(url, destination: Path, backup_url, requests, tqdm) -> None:
    def attempt(download_url: str) -> bool:
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        file_size = int(response.headers.get("Content-Length", 0))

        if destination.exists():
            local_size = destination.stat().st_size
            if file_size and file_size == local_size:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        with tqdm(
            total=file_size,
            unit="iB",
            unit_scale=True,
            desc=os.path.basename(download_url),
        ) as progress_bar:
            with destination.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if attempt(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"Primary URL failed. Trying backup URL: {backup_url}")
            if attempt(backup_url):
                return
        raise


def _load_gpt2_params_from_tf_ckpt(ckpt_path, settings: dict, tf) -> dict:
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})
        target_dict[variable_name_parts[-1]] = variable_array

    return params


def assign(left: torch.Tensor, right) -> nn.Parameter:
    """Return a parameter after validating source and target shapes."""
    right_tensor = torch.as_tensor(right, dtype=left.dtype)
    if tuple(left.shape) != tuple(right_tensor.shape):
        raise ValueError(f"Shape mismatch. Left: {left.shape}, right: {right_tensor.shape}")
    return nn.Parameter(right_tensor.clone().detach())


def load_weights_into_gpt(gpt, params: dict) -> None:
    """Load OpenAI GPT-2 checkpoint parameters into GPTModel."""
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for block_idx in range(len(params["blocks"])):
        block = gpt.trf_blocks[block_idx]
        checkpoint_block = params["blocks"][block_idx]

        q_w, k_w, v_w = np.split(checkpoint_block["attn"]["c_attn"]["w"], 3, axis=-1)
        block.att.w_query.weight = assign(block.att.w_query.weight, q_w.T)
        block.att.w_key.weight = assign(block.att.w_key.weight, k_w.T)
        block.att.w_value.weight = assign(block.att.w_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(checkpoint_block["attn"]["c_attn"]["b"], 3, axis=-1)
        block.att.w_query.bias = assign(block.att.w_query.bias, q_b)
        block.att.w_key.bias = assign(block.att.w_key.bias, k_b)
        block.att.w_value.bias = assign(block.att.w_value.bias, v_b)

        block.att.out_proj.weight = assign(
            block.att.out_proj.weight,
            checkpoint_block["attn"]["c_proj"]["w"].T,
        )
        block.att.out_proj.bias = assign(
            block.att.out_proj.bias,
            checkpoint_block["attn"]["c_proj"]["b"],
        )

        block.ff.layers[0].weight = assign(
            block.ff.layers[0].weight,
            checkpoint_block["mlp"]["c_fc"]["w"].T,
        )
        block.ff.layers[0].bias = assign(
            block.ff.layers[0].bias,
            checkpoint_block["mlp"]["c_fc"]["b"],
        )
        block.ff.layers[2].weight = assign(
            block.ff.layers[2].weight,
            checkpoint_block["mlp"]["c_proj"]["w"].T,
        )
        block.ff.layers[2].bias = assign(
            block.ff.layers[2].bias,
            checkpoint_block["mlp"]["c_proj"]["b"],
        )

        block.norm1.scale = assign(block.norm1.scale, checkpoint_block["ln_1"]["g"])
        block.norm1.shift = assign(block.norm1.shift, checkpoint_block["ln_1"]["b"])
        block.norm2.scale = assign(block.norm2.scale, checkpoint_block["ln_2"]["g"])
        block.norm2.shift = assign(block.norm2.shift, checkpoint_block["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

