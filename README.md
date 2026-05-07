<h1 align="center">Build GPT-2 From Scratch & LLM Fine-tuning Workflows</h1>

<h4 align="center">
A compact, runnable learning project for implementing GPT-2-style models, classification fine-tuning, instruction tuning, and automated evaluation.
</h4>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch" alt="PyTorch"></a>
  <a href="https://github.com/openai/tiktoken"><img src="https://img.shields.io/badge/Tokenizer-tiktoken-4b8bbe" alt="tiktoken"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-highlights">Highlights</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-running-examples">Examples</a> •
  <a href="#-project-structure">Structure</a>
</p>

---

## 📰 News

* **[2026.05.07]** Reorganized the original learning scripts into a GitHub-ready Python project.
* **[2026.05.07]** Added the reusable `src/gpt2_from_scratch/` package with separate modules for attention, model definition, generation, training, fine-tuning, and evaluation.
* **[2026.05.07]** Added chapter-based scripts for quick demos and workflow execution.
* **[2026.05.07]** Added legacy checkpoint compatibility for models saved by the original notebook-style scripts.
* **[2026.05.07]** Added tests, GitHub Actions CI, citation metadata, and attribution notes.

## 📌 Overview

This repository contains two connected learning projects.

### Project 1: Build GPT-2 From Scratch

Chapters 1-5 implement a GPT-2-style autoregressive language model from core components:

* Text loading, GPT-2 BPE tokenization, and sliding-window sampling
* Self-attention, causal attention, and multi-head attention
* LayerNorm, GELU, feed-forward layers, and Transformer blocks
* GPTModel forward pass, parameter counting, and text generation
* Language-model training loops and loss evaluation
* Greedy decoding, top-k sampling, and temperature sampling
* Loading original OpenAI GPT-2 checkpoint weights into the local model implementation

### Project 2: LLM Fine-tuning and Evaluation Workflows

Chapters 6-7 build practical fine-tuning and evaluation workflows on top of the GPT implementation:

* SMS spam classification fine-tuning
* Instruction dataset formatting and supervised fine-tuning data loaders
* Instruction-tuned response generation
* Automated response scoring with a local Ollama judge model

## 😮 Highlights

### 💡 Educational but runnable

The code keeps the step-by-step learning path while being organized as reusable modules. You can inspect individual files such as `attention.py` and `model.py`, or run chapter scripts directly from `scripts/`.

### 🔥 Two complete workflows

The repository goes beyond model definition. It covers data preparation, language-model training utilities, classification fine-tuning, instruction tuning, response generation, and automated evaluation.

### 🧩 GitHub-ready structure

Large checkpoints, generated data, caches, and original scratch scripts are excluded through `.gitignore`, keeping the public repository clean and reproducible.

## 🛠️ Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/huangchong-yan/0-GPT2.git
cd 0-GPT2

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

To load the original OpenAI GPT-2 TensorFlow checkpoints, install TensorFlow as well:

```bash
pip install "tensorflow>=2.15"
```

## 🚀 Running Examples

### Chapter 2: Data Pipeline

```bash
python scripts/ch02_data_pipeline.py --text-file the-verdict.txt
```

### Chapter 3: Causal Multi-head Attention

```bash
python scripts/ch03_attention_demo.py
```

### Chapter 4: Build GPT Model

```bash
python scripts/ch04_build_gpt.py --config tiny
```

### Chapter 5: Text Generation

Run a quick smoke test with the tiny GPT configuration:

```bash
python scripts/ch05_pretrain_or_load_gpt2.py --prompt "Every effort moves you"
```

Generate text with the OpenAI GPT-2 124M weights:

```bash
python scripts/ch05_pretrain_or_load_gpt2.py --pretrained --prompt "Every effort moves you"
```

### Chapter 6: Classification Fine-tuning

Prepare the SMS spam dataset:

```bash
python scripts/ch06_finetune_classifier.py --mode prepare
```

Train the classifier:

```bash
python scripts/ch06_finetune_classifier.py --mode train --epochs 5
```

Run inference with a trained checkpoint:

```bash
python scripts/ch06_finetune_classifier.py --mode infer --text "Hey, are we still meeting tonight?"
```

### Chapter 7: Instruction Fine-tuning and Evaluation

Preview the instruction format:

```bash
python scripts/ch07_instruction_tuning_eval.py --mode preview
```

Train an SFT model:

```bash
python scripts/ch07_instruction_tuning_eval.py --mode train --epochs 2
```

Generate responses for the test set:

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
│   ├── checkpointing.py         # Legacy checkpoint compatibility
│   ├── classification.py        # SMS spam classification fine-tuning
│   ├── config.py                # GPT and GPT-2 model configs
│   ├── data.py                  # Tokenization and sliding-window dataset
│   ├── evaluation.py            # Ollama-based automatic evaluation
│   ├── generation.py            # Greedy, top-k, and temperature decoding
│   ├── gpt2_weights.py          # Download and load OpenAI GPT-2 checkpoints
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
│   ├── test_attention.py
│   ├── test_instruction_tuning.py
│   └── test_model.py
├── docs/
│   ├── CHAPTERS.md
│   └── UPLOAD_TO_GITHUB.md
├── .github/workflows/ci.yml
├── CITATION.cff
├── NOTICE
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## 📚 Chapter Mapping

| Chapter | Topic | Entry |
| --- | --- | --- |
| 1-2 | Tokenization and data sampling | `scripts/ch02_data_pipeline.py` |
| 3 | Attention mechanism | `scripts/ch03_attention_demo.py` |
| 4 | GPT architecture | `scripts/ch04_build_gpt.py` |
| 5 | Pretraining and GPT-2 weights | `scripts/ch05_pretrain_or_load_gpt2.py` |
| 6 | Classification fine-tuning | `scripts/ch06_finetune_classifier.py` |
| 7 | Instruction tuning and evaluation | `scripts/ch07_instruction_tuning_eval.py` |

More details are available in [`docs/CHAPTERS.md`](docs/CHAPTERS.md).

## 🧾 Data and Checkpoints

The following files are generated locally or too large for normal GitHub commits, so they are ignored by `.gitignore`:

* `gpt2/`
* `data/`
* `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`
* `sms_spam_collection/`, `sms_spam_collection.zip`
* `train.csv`, `validation.csv`, `test.csv`
* `instruction-data-with-response.json`

If you want to publish trained weights, use GitHub Releases, Hugging Face Hub, or another model hosting service instead of committing them directly.

## ✅ Verification

The following commands are suitable for quick local checks:

```bash
python -m compileall src scripts
pytest
python scripts/ch02_data_pipeline.py --text-file the-verdict.txt --batch-size 2 --max-length 4 --stride 4
python scripts/ch03_attention_demo.py
python scripts/ch04_build_gpt.py --config tiny
python scripts/ch05_pretrain_or_load_gpt2.py --max-new-tokens 5
python scripts/ch07_instruction_tuning_eval.py --mode preview
python scripts/ch06_finetune_classifier.py --mode infer
```

## 👍 Acknowledgement

This project is based on my learning notes and code organization work around Sebastian Raschka's **Build a Large Language Model From Scratch**.

Thanks to the original educational material and open-source examples that make it easier to understand how GPT-style models are built layer by layer.

## 🔒 License

This repository is released under the Apache 2.0 License. See [`LICENSE`](LICENSE) for details.

The GPT-2 checkpoint download and loading helper is adapted from Sebastian Raschka's `LLMs-from-scratch` project and retains its Apache 2.0 license notice.

## ✏️ Citation

If this repository helps your learning or project, please consider citing the original book/project:

```BibTeX
@book{raschka2024buildllm,
  title = {Build a Large Language Model From Scratch},
  author = {Sebastian Raschka},
  year = {2024},
  publisher = {Manning Publications}
}
```

