<h1 align="center">Build GPT-2 From Scratch & LLM Fine-tuning Workflows</h1>

<h4 align="center">
从零构建类 GPT-2 大模型，并完成分类微调、指令微调与自动化评估工作流
</h4>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch" alt="PyTorch"></a>
  <a href="https://github.com/openai/tiktoken"><img src="https://img.shields.io/badge/Tokenizer-tiktoken-4b8bbe" alt="tiktoken"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"></a>
</p>

<p align="center">
  <a href="#-项目简介">项目简介</a> •
  <a href="#-highlights">Highlights</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-running-examples">Examples</a> •
  <a href="#-project-structure">Structure</a>
</p>

---

## 📰 News

* **[2026.05.07]** 重新整理项目结构，形成适合上传 GitHub 的模块化代码仓库。
* **[2026.05.07]** 新增 `src/gpt2_from_scratch/` 核心包，拆分 attention、model、training、generation、fine-tuning 与 evaluation 模块。
* **[2026.05.07]** 新增章节脚本入口，支持第 2-7 章的快速运行、微调与自动化评估流程。
* **[2026.05.07]** 新增旧版 checkpoint 兼容加载逻辑，可读取原学习脚本保存的 `review_classifier.pth`。

## 📌 项目简介

本项目包含两个相互衔接的 LLM 学习与实践工程：

### Project 1: Build GPT-2 From Scratch

对应第 1-5 章，目标是从基础组件开始实现一个类 GPT-2 自回归语言模型：

* 文本读取、BPE tokenization 与滑动窗口采样
* 自注意力、因果注意力、多头注意力
* LayerNorm、GELU、FeedForward、Transformer Block
* GPTModel 前向计算、参数统计与文本生成
* 语言模型训练循环、top-k / temperature sampling
* 加载 OpenAI GPT-2 checkpoint 到本地实现的模型结构中

### Project 2: LLM Fine-tuning & Evaluation Workflows

对应第 6-7 章，目标是在基础 GPT 模型上构建更完整的微调与评估流程：

* SMS Spam 二分类任务微调
* 指令数据集格式化与 SFT dataloader
* 指令微调训练与模型响应生成
* 使用本地 Ollama 模型进行自动化评分

## 😮 Highlights

### 💡 Educational but runnable

代码保留了“从零理解”的学习路径，同时整理成可复用模块。你可以单独阅读 `attention.py`、`model.py`，也可以直接运行 `scripts/` 中的章节入口。

### 🔥 Two complete workflows

仓库不是只实现一个模型结构，还覆盖了从数据准备、预训练式语言建模、分类微调、指令微调到自动化评估的完整路径。

### 🧩 GitHub-ready structure

原始草稿脚本、本地权重、生成数据和缓存文件已通过 `.gitignore` 排除。上传 GitHub 时主仓库保持轻量、清晰、可复现。

## 🛠️ Requirements and Installation

建议使用 Python 3.10 或更高版本。

```bash
git clone https://github.com/<your-name>/<your-repo>.git
cd <your-repo>

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果需要加载 OpenAI 原始 GPT-2 TensorFlow checkpoint，请额外安装：

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

默认使用 tiny GPT 配置，适合快速检查代码是否能运行：

```bash
python scripts/ch05_pretrain_or_load_gpt2.py --prompt "Every effort moves you"
```

加载 OpenAI GPT-2 124M 权重后生成文本：

```bash
python scripts/ch05_pretrain_or_load_gpt2.py --pretrained --prompt "Every effort moves you"
```

### Chapter 6: Classification Fine-tuning

准备 SMS spam 数据集：

```bash
python scripts/ch06_finetune_classifier.py --mode prepare
```

训练分类器：

```bash
python scripts/ch06_finetune_classifier.py --mode train --epochs 5
```

使用已训练 checkpoint 推理：

```bash
python scripts/ch06_finetune_classifier.py --mode infer --text "Hey, are we still meeting tonight?"
```

### Chapter 7: Instruction Fine-tuning and Evaluation

预览指令数据格式：

```bash
python scripts/ch07_instruction_tuning_eval.py --mode preview
```

训练 SFT 模型：

```bash
python scripts/ch07_instruction_tuning_eval.py --mode train --epochs 2
```

生成测试集响应：

```bash
python scripts/ch07_instruction_tuning_eval.py --mode generate-responses
```

使用本地 Ollama 模型自动评分：

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
│   ├── generation.py            # Greedy, top-k and temperature decoding
│   ├── gpt2_weights.py          # Download and load OpenAI GPT-2 checkpoints
│   ├── instruction_tuning.py    # Instruction dataset and SFT collate function
│   ├── model.py                 # LayerNorm, GELU, TransformerBlock, GPTModel
│   ├── runtime.py               # Windows/UTF-8 runtime helpers
│   └── training.py              # Language-model training and loss evaluation
├── scripts/
│   ├── ch02_data_pipeline.py
│   ├── ch03_attention_demo.py
│   ├── ch04_build_gpt.py
│   ├── ch05_pretrain_or_load_gpt2.py
│   ├── ch06_finetune_classifier.py
│   └── ch07_instruction_tuning_eval.py
├── docs/
│   ├── CHAPTERS.md
│   └── UPLOAD_TO_GITHUB.md
├── instruction-data.json
├── the-verdict.txt
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

## ✅ Verified Commands

The following commands have been tested in the local workspace:

```bash
python -m compileall src scripts
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

