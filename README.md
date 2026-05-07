<h1 align="center">GPT-2 From Scratch and Fine-tuning Workflows</h1>

<h4 align="center">
Build a GPT-2-style Transformer in PyTorch, then use it for text generation, classification fine-tuning, instruction tuning, and automated response evaluation.
</h4>

<h5 align="center">
If this project helps you understand GPT-style language models, please consider giving it a star ⭐.
</h5>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch" alt="PyTorch"></a>
  <a href="https://github.com/openai/tiktoken"><img src="https://img.shields.io/badge/Tokenizer-tiktoken-4b8bbe" alt="tiktoken"></a>
  <a href="https://github.com/huangchong-yan/0-GPT2/actions"><img src="https://img.shields.io/badge/CI-pytest-2ea44f?logo=githubactions" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"></a>
</p>

<p align="center">
  <a href="#-news">News</a> •
  <a href="#-highlights">Highlights</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-workflows">Workflows</a> •
  <a href="#-project-structure">Structure</a>
</p>

---

## 📰 News

* **[2026.05.07]** ✨ Public project README polished with a cleaner research-project style.
* **[2026.05.07]** 🚀 Added GitHub Actions CI with syntax checks and unit tests.
* **[2026.05.07]** ✅ Added unit tests for attention, GPT forward pass, generation, and instruction-tuning collation.
* **[2026.05.07]** 🧩 Added citation metadata and attribution notes.

## 😮 Highlights

This repository is a compact implementation of a GPT-2-style decoder-only Transformer, paired with practical workflows that show how the model can be used beyond a forward pass.

### 💡 Transparent GPT-2-style implementation

The model is implemented directly in PyTorch, with readable modules for token embeddings, positional embeddings, causal attention, multi-head attention, Transformer blocks, and the final language-model head.

### 🔥 Runnable generation and fine-tuning workflows

The scripts cover text generation, loading GPT-2 checkpoint weights, SMS spam classification fine-tuning, instruction tuning, response generation, and local automated evaluation.

### 🧪 Tested and CI-ready

The repository includes lightweight unit tests and a GitHub Actions workflow so future changes can be checked automatically.

## 🤗 Overview

The project focuses on four capabilities:

* **Model internals**: embeddings, causal self-attention, multi-head attention, layer normalization, GELU, feed-forward networks, Transformer blocks, and GPT-style output heads.
* **Text generation**: greedy decoding, top-k filtering, temperature sampling, and context-window cropping.
* **Fine-tuning**: adapting a GPT-style model for classification and supervised instruction tuning.
* **Evaluation**: scoring generated instruction responses with a local Ollama judge model.

## 🛠️ Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/huangchong-yan/0-GPT2.git
cd 0-GPT2

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For Linux or macOS:

```bash
source .venv/bin/activate
```

To load original OpenAI GPT-2 TensorFlow checkpoints, install TensorFlow as well:

```bash
pip install "tensorflow>=2.15"
```

## 🚀 Quickstart

### Inspect the data pipeline

```bash
python scripts/ch02_data_pipeline.py --text-file the-verdict.txt
```

### Run causal multi-head attention

```bash
python scripts/ch03_attention_demo.py
```

### Build a small GPT model

```bash
python scripts/ch04_build_gpt.py --config tiny
```

### Generate text

```bash
python scripts/ch05_pretrain_or_load_gpt2.py --prompt "Every effort moves you"
```

Generate with OpenAI GPT-2 124M weights:

```bash
python scripts/ch05_pretrain_or_load_gpt2.py --pretrained --prompt "Every effort moves you"
```

## 🧠 Workflows

### Fine-tune a spam classifier

Prepare the SMS spam dataset:

```bash
python scripts/ch06_finetune_classifier.py --mode prepare
```

Train the classifier:

```bash
python scripts/ch06_finetune_classifier.py --mode train --epochs 5
```

Run inference:

```bash
python scripts/ch06_finetune_classifier.py --mode infer --text "Hey, are we still meeting tonight?"
```

### Run instruction tuning and evaluation

Preview the instruction format:

```bash
python scripts/ch07_instruction_tuning_eval.py --mode preview
```

Train a supervised fine-tuned model:

```bash
python scripts/ch07_instruction_tuning_eval.py --mode train --epochs 2
```

Generate responses for the test split:

```bash
python scripts/ch07_instruction_tuning_eval.py --mode generate-responses
```

Score generated responses with a local Ollama model:

```bash
python scripts/ch07_instruction_tuning_eval.py --mode evaluate --ollama-model phi3
```

## 📁 Project Structure

```text
.
├── src/gpt2_from_scratch/
│   ├── attention.py             # Self-attention, causal attention, multi-head attention
│   ├── checkpointing.py         # Compatibility helpers for older checkpoints
│   ├── classification.py        # SMS spam classification fine-tuning
│   ├── config.py                # GPT and GPT-2 model configs
│   ├── data.py                  # Tokenization and sliding-window datasets
│   ├── evaluation.py            # Ollama-based automatic evaluation
│   ├── generation.py            # Greedy, top-k, and temperature decoding
│   ├── gpt2_weights.py          # OpenAI GPT-2 checkpoint loading
│   ├── instruction_tuning.py    # Instruction dataset and SFT collate function
│   ├── model.py                 # LayerNorm, GELU, TransformerBlock, GPTModel
│   ├── runtime.py               # Windows and UTF-8 runtime helpers
│   └── training.py              # Language-model training and loss evaluation
├── scripts/
│   ├── ch02_data_pipeline.py
│   ├── ch03_attention_demo.py
│   ├── ch04_build_gpt.py
│   ├── ch05_pretrain_or_load_gpt2.py
│   ├── ch06_finetune_classifier.py
│   └── ch07_instruction_tuning_eval.py
├── tests/
├── docs/
├── .github/workflows/ci.yml
├── CITATION.cff
├── NOTICE
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## 🧩 Main Modules

| Module | Purpose |
| --- | --- |
| `attention.py` | Self-attention, causal masking, and multi-head attention |
| `model.py` | GPT-style Transformer model implementation |
| `generation.py` | Text generation and sampling utilities |
| `training.py` | Language-model loss evaluation and training loop |
| `classification.py` | Dataset preparation, training, evaluation, and inference for spam classification |
| `instruction_tuning.py` | Instruction formatting, dataset construction, and custom collate function |
| `evaluation.py` | Local Ollama-based response scoring |
| `gpt2_weights.py` | Downloading and loading OpenAI GPT-2 checkpoints |

## 📦 Data and Checkpoints

The repository intentionally excludes generated data and large model files:

* `gpt2/`
* `data/`
* `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`
* `sms_spam_collection/`, `sms_spam_collection.zip`
* `train.csv`, `validation.csv`, `test.csv`
* `instruction-data-with-response.json`

If trained weights need to be shared, use GitHub Releases, Hugging Face Hub, or another model hosting service instead of committing them directly.

## ✅ Verification

Run the test suite:

```bash
pytest
```

Run a syntax check:

```bash
python -m compileall src scripts tests
```

The repository also includes a GitHub Actions workflow that runs these checks on pushes and pull requests to `main`.

## 👍 Acknowledgement

This project follows the model-building path popularized by Sebastian Raschka's **Build a Large Language Model From Scratch** and its companion open-source code. The implementation here is organized as a standalone PyTorch project with additional fine-tuning and evaluation workflows.

## 🔒 License

This repository is released under the Apache 2.0 License. See [`LICENSE`](LICENSE) for details.

The GPT-2 checkpoint download and loading helper is adapted from Sebastian Raschka's `LLMs-from-scratch` project and retains its Apache 2.0 license notice.

## ✏️ Citation

If this repository is useful for your learning or project, please cite the original book/project:

```BibTeX
@book{raschka2024buildllm,
  title = {Build a Large Language Model From Scratch},
  author = {Sebastian Raschka},
  year = {2024},
  publisher = {Manning Publications}
}
```

