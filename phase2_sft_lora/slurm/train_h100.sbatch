#!/bin/bash
#SBATCH --job-name=qwen25_15b_lora_sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs outputs

# ---- Env (choose one) ----
# Option A: conda (if available)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate agent

# Option B: venv
# python -m venv .venv
# source .venv/bin/activate

python -m pip install -U -r phase2_sft_lora/requirements.kaggle.txt

# Recommended for H100: bf16 + (optionally) no-quant.
# For Qwen2.5-1.5B, H100 can easily run without 4bit, which usually improves stability.
python phase2_sft_lora/scripts/10_sft_lora_unsloth.py \
  --model_name Qwen/Qwen2.5-1.5B \
  --train_jsonl phase1_data/data/processed/train.jsonl \
  --val_jsonl phase1_data/data/processed/val.jsonl \
  --output_dir outputs/qwen2.5-1.5b-lora-sft \
  --max_seq_length 1024 \
  --load_in_4bit false \
  --bf16 true \
  --fp16 false \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --packing true


