# 章节与代码对应关系

## 项目一：从 0 构建类 GPT-2 大模型

| 章节 | 主题 | 主要文件 |
| --- | --- | --- |
| 第 1-2 章 | 文本读取、分词、滑动窗口采样 | `src/gpt2_from_scratch/data.py`, `scripts/ch02_data_pipeline.py` |
| 第 3 章 | 自注意力、因果注意力、多头注意力 | `src/gpt2_from_scratch/attention.py`, `scripts/ch03_attention_demo.py` |
| 第 4 章 | LayerNorm、GELU、Transformer Block、GPTModel | `src/gpt2_from_scratch/model.py`, `scripts/ch04_build_gpt.py` |
| 第 5 章 | 预训练、生成策略、加载 GPT-2 权重 | `src/gpt2_from_scratch/training.py`, `src/gpt2_from_scratch/gpt2_weights.py`, `scripts/ch05_pretrain_or_load_gpt2.py` |

## 项目二：LLM 指令微调与自动化评估工作流

| 章节 | 主题 | 主要文件 |
| --- | --- | --- |
| 第 6 章 | GPT-2 分类微调，SMS spam 任务 | `src/gpt2_from_scratch/classification.py`, `scripts/ch06_finetune_classifier.py` |
| 第 7 章 | 指令微调数据管线、SFT、模型响应生成、Ollama 自动评分 | `src/gpt2_from_scratch/instruction_tuning.py`, `src/gpt2_from_scratch/evaluation.py`, `scripts/ch07_instruction_tuning_eval.py` |

