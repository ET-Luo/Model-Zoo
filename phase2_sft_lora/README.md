# Phase 2: SFT + LoRA (Unsloth)

目标：用 **Unsloth** 在 `Qwen/Qwen2.5-1.5B` 上做 **LoRA 监督微调（SFT）**，把 Phase 1 的 `messages` 数据集训练成“中文 + 毒舌口吻”的百科问答助手。

> 你当前是笔记本：本地只需要 `--dry_run` 验证脚本/数据管线；真正训练建议在 Kaggle（Linux + GPU）上跑。

## 输入数据

默认使用 Phase 1 的产物：
- `../phase1_data/data/processed/train.jsonl`
- `../phase1_data/data/processed/val.jsonl`

每行格式（OpenAI chat messages）：
- `{"id": "...", "messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}`

## Kaggle（推荐：实际训练）

在 Kaggle Notebook 里：

1) 安装依赖（建议新开一个 cell）：

```bash
pip -q install -U -r phase2_sft_lora/requirements.kaggle.txt
```

2) 运行训练（会输出 LoRA adapter 到 `outputs/`）：

```bash
python phase2_sft_lora/scripts/10_sft_lora_unsloth.py \
  --model_name Qwen/Qwen2.5-1.5B \
  --train_jsonl phase1_data/data/processed/train.jsonl \
  --val_jsonl phase1_data/data/processed/val.jsonl \
  --output_dir outputs/qwen2.5-1.5b-lora-sft \
  --max_seq_length 1024 \
  --load_in_4bit true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1
```

3) 训练后做推理 sanity check：

```bash
python phase2_sft_lora/scripts/11_infer_adapter.py \
  --base_model Qwen/Qwen2.5-1.5B \
  --adapter_dir outputs/qwen2.5-1.5b-lora-sft \
  --system_lang zh \
  --prompt "青蒿素是从何提取而来的？"
```

## Windows 笔记本（只验证脚本：dry-run）

你本机不需要装 Unsloth，也不需要 GPU。只跑数据管线检查即可：

```powershell
Set-Location E:\CodeStore\Hope\Zoo\Model-Zoo
conda run -n agent python phase2_sft_lora\scripts\10_sft_lora_unsloth.py `
  --train_jsonl phase1_data\data\processed\train.jsonl `
  --val_jsonl phase1_data\data\processed\val.jsonl `
  --dry_run true
```

它会验证：
- JSONL 可读
- `messages` 能被正确渲染为训练文本
- 能生成一个 batch 的 tokenized 输出（但不会训练）

## RWTH HPC（H100：推荐参数）

H100 支持 BF16，且对 `Qwen/Qwen2.5-1.5B` 来说显存非常宽裕，建议：
- **优先用 `bf16=true`**
- **优先关闭 4bit**（`load_in_4bit=false`，通常更稳定、也更快）
- batch 可以适当放大（下面给一个保守但效率不错的起点）

### Slurm 提交示例

仓库已提供：`phase2_sft_lora/slurm/train_h100.sbatch`

```bash
sbatch phase2_sft_lora/slurm/train_h100.sbatch
```


