# 上传到 GitHub

建议先确认 `.gitignore` 生效，尤其不要把 `gpt2/`、`.pth` 权重文件、生成数据和原始草稿脚本上传。

## 首次上传

```bash
git init
git add README.md LICENSE .gitignore pyproject.toml requirements.txt src scripts docs the-verdict.txt instruction-data.json
git commit -m "Organize GPT-2 from scratch and LLM fine-tuning workflows"
git branch -M main
git remote add origin https://github.com/huangchong-yan/0-GPT2.git
git push -u origin main
```

如果你不想上传 `instruction-data.json`，可以从 `git add` 命令里删掉它；第 7 章脚本会在需要时自动下载。

## 检查将要上传的文件

```bash
git status --short
git check-ignore -v gpt2/124M/model.ckpt.data-00000-of-00001
git check-ignore -v review_classifier.pth
```

如果 `git check-ignore` 能显示命中的 `.gitignore` 规则，就说明大文件会被排除。
