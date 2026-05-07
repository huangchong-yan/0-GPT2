# 从零构建 GPT-2 与 LLM 微调工作流

这个仓库整理了两个学习项目：

1. **第 1-5 章：从零构建类 GPT-2 大模型**
   - 文本采样与 GPT-2 BPE tokenization
   - 自注意力、因果注意力、多头注意力
   - Transformer Block 与 GPTModel
   - 预训练循环、文本生成、top-k / temperature 采样
   - 加载 OpenAI GPT-2 权重

2. **第 6-7 章：LLM 指令微调与自动化评估工作流**
   - SMS spam 分类微调
   - 指令数据集格式化与 SFT 数据管线
   - 模型响应生成
   - 使用本地 Ollama 模型进行自动评分

原始学习脚本仍保留在本地，但已经通过 `.gitignore` 排除；GitHub 上建议上传这里整理后的 `src/`、`scripts/`、文档和配置文件。

## 项目结构

```text
.
├── src/gpt2_from_scratch/       # 可复用核心代码
│   ├── attention.py             # 自注意力、因果注意力、多头注意力
│   ├── classification.py        # 第 6 章 SMS 分类微调工具
│   ├── config.py                # GPT 配置
│   ├── data.py                  # 第 2 章文本采样数据集
│   ├── evaluation.py            # Ollama 自动评估
│   ├── generation.py            # greedy/top-k/temperature 生成
│   ├── gpt2_weights.py          # GPT-2 权重下载与加载
│   ├── instruction_tuning.py    # 第 7 章指令微调数据管线
│   ├── model.py                 # LayerNorm/GELU/Transformer/GPTModel
│   └── training.py              # 语言模型训练与评估工具
├── scripts/                     # 按章节运行的示例/工作流入口
├── docs/                        # 上传 GitHub 与使用说明
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果需要加载 OpenAI 原始 GPT-2 TensorFlow checkpoint，请额外安装：

```bash
pip install "tensorflow>=2.15"
```

## 运行示例

第 2 章：文本采样与 dataloader：

```bash
python scripts/ch02_data_pipeline.py --text-file the-verdict.txt
```

第 3 章：多头因果注意力：

```bash
python scripts/ch03_attention_demo.py
```

第 4 章：构建 GPT 模型：

```bash
python scripts/ch04_build_gpt.py
```

第 5 章：文本生成，默认使用 tiny 配置做快速 smoke run：

```bash
python scripts/ch05_pretrain_or_load_gpt2.py --prompt "Every effort moves you"
```

加载 OpenAI GPT-2 124M 权重：

```bash
python scripts/ch05_pretrain_or_load_gpt2.py --pretrained --prompt "Every effort moves you"
```

第 6 章：准备 SMS spam 分类数据：

```bash
python scripts/ch06_finetune_classifier.py --mode prepare
```

第 7 章：预览指令数据格式：

```bash
python scripts/ch07_instruction_tuning_eval.py --mode preview
```

更多章节对应关系见 `docs/CHAPTERS.md`；上传步骤见 `docs/UPLOAD_TO_GITHUB.md`。

## 数据与模型文件

以下文件会自动生成或体积较大，不建议上传到 GitHub：

- `gpt2/`
- `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`
- `data/`
- `sms_spam_collection/`, `sms_spam_collection.zip`
- `train.csv`, `validation.csv`, `test.csv`
- `instruction-data-with-response.json`

这些内容已经写入 `.gitignore`。如果未来要发布模型权重，建议使用 GitHub Releases、Hugging Face Hub 或其他模型托管服务。

## 代码来源说明

本项目学习与整理自 Sebastian Raschka 的《Build a Large Language Model From Scratch》相关章节代码，并在此基础上做了模块化、命名规范、CLI 工作流和 GitHub 仓库结构整理。`gpt2_weights.py` 中的 GPT-2 checkpoint 下载/读取逻辑保留了原项目 Apache 2.0 许可声明。
