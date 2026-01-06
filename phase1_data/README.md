# Phase 1: Data Engineering (SFT 数据准备)

目标：把一个公开小数据集（问答/百科类）变成后续 SFT/LoRA 可直接训练的 `train.jsonl` / `val.jsonl`，并输出 token 长度统计，避免“数据都没准备好就开训”。

## 你将得到什么

- `data/raw/`：从 Hugging Face 下载的原始数据快照（jsonl）
- `data/processed/`：转换后的 SFT 数据（jsonl），字段统一为：
  - `id`: string
  - `messages`: OpenAI Chat 格式，`[{role: "system"|"user"|"assistant", content: "..."}]`
- `reports/token_stats.json`：token 长度分布、超长样本比例等
- `reports/samples_preview.jsonl`：抽样的可读预览，方便人工检查“语气/事实/格式”

## 快速开始（PowerShell + conda env: agent）

1) 安装依赖到你的 conda 环境（你已指定 `conda activate agent`）：

```powershell
Set-Location E:\CodeStore\Hope\Zoo\Model-Zoo
conda run -n agent python -m pip install -r requirements.txt
```

2) 下载一个中文数据集（推荐 `cmrc2018`，可直接 `load_dataset()`）：

```powershell
conda run -n agent python phase1_data\scripts\01_download_dataset.py `
  --dataset cmrc2018 `
  --out phase1_data\data\raw\cmrc2018.jsonl `
  --limit 2000
```

3) 生成 SFT 格式数据（含“毒舌口吻”的轻量前后缀；并将 `context` 拼进 user 提示）：

```powershell
conda run -n agent python phase1_data\scripts\02_build_sft_data.py `
  --in phase1_data\data\raw\cmrc2018.jsonl `
  --out_train phase1_data\data\processed\train.jsonl `
  --out_val phase1_data\data\processed\val.jsonl `
  --system_lang zh `
  --style sarcastic `
  --style_strength 0.8 `
  --max_context_chars 800 `
  --max_samples 500
```  

4) 统计 token（默认用 `Qwen/Qwen2.5-0.5B` 的 tokenizer）：

```powershell
conda run -n agent python phase1_data\scripts\03_token_stats.py `
  --model Qwen/Qwen2.5-0.5B `
  --in phase1_data\data\processed\train.jsonl `
  --out phase1_data\reports\token_stats.json
```

## 风格（重要）

这里的“毒舌/幽默”建议只做成**轻度讽刺/吐槽式的表达**，避免：
- 指向真实个人/群体的人身攻击
- 涉及受保护属性（种族、宗教、国籍等）的贬损
- 鼓励自残、暴力、违法等内容

后续 Phase 2 做 SFT/LoRA 时，我们也会加入**事实性与安全性**的简单约束（例如系统提示 + 过滤规则），避免模型“只学会嘴臭”。 



