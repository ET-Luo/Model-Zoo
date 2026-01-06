import sys
import argparse
import json
import os

# Suggested by Unsloth for specific CUDA environments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset


def _str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y"}


# Make Chinese logs readable on Windows terminals (best-effort).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _render_messages_zh(messages: List[Dict[str, Any]]) -> str:
    """
    MVP: render chat messages to a deterministic plain-text format.
    This avoids any tokenizer chat-template dependency in dry-run mode.
    """
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not role or not content:
            continue
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
        else:
            parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts).strip()


def _messages_to_text(example: Dict[str, Any], eos_token: str = "") -> str:
    msgs = example.get("messages")
    if not isinstance(msgs, list):
        raise ValueError("Example missing 'messages' list")
    return _render_messages_zh(msgs) + eos_token


@dataclass
class TrainConfig:
    model_name: str
    train_jsonl: str
    val_jsonl: Optional[str]
    output_dir: str
    max_seq_length: int
    load_in_4bit: bool
    bf16: bool
    fp16: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    warmup_steps: int
    logging_steps: int
    save_steps: int
    eval_steps: int
    seed: int
    packing: bool
    dry_run: bool


def _load_jsonl_as_dataset(train_jsonl: str, val_jsonl: Optional[str]) -> Dict[str, Any]:
    files: Dict[str, str] = {"train": train_jsonl}
    if val_jsonl:
        files["validation"] = val_jsonl
    return load_dataset("json", data_files=files)


def _dry_run_pipeline(cfg: TrainConfig) -> None:
    ds = _load_jsonl_as_dataset(cfg.train_jsonl, cfg.val_jsonl)
    train = ds["train"]
    ex0 = train[0]
    
    # Tokenize using transformers tokenizer only (no Unsloth dependency for Windows sanity checks).
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    text0 = _messages_to_text(ex0, eos_token=tok.eos_token)
    
    print("dry_run: loaded train examples =", len(train))
    print("dry_run: example0 id =", ex0.get("id"))
    print("dry_run: rendered_chars =", len(text0))
    print("dry_run: rendered_preview =", text0[:240].replace("\n", "\\n"))

    out = tok(text0, truncation=True, max_length=cfg.max_seq_length, add_special_tokens=True)
    print("dry_run: tokenized_len =", len(out["input_ids"]))


def _train_with_unsloth(cfg: TrainConfig) -> None:
    # Lazy imports so Windows users can run --dry_run without installing Unsloth.
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        use_rslora=False,
        loftq_config=None,
    )

    ds = _load_jsonl_as_dataset(cfg.train_jsonl, cfg.val_jsonl)

    def to_text(batch: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": _messages_to_text(batch, eos_token=tokenizer.eos_token)}

    train_ds = ds["train"].map(to_text, remove_columns=ds["train"].column_names)
    eval_ds = ds["validation"].map(to_text, remove_columns=ds["validation"].column_names) if "validation" in ds else None

    _ensure_parent_dir(os.path.join(cfg.output_dir, "placeholder.txt"))

    # transformers>=4.57 uses `eval_strategy` instead of `evaluation_strategy`.
    args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.eval_steps if eval_ds is not None else None,
        save_total_limit=2,
        seed=cfg.seed,
        report_to="tensorboard",
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        dataloader_num_workers=2,
        dataset_text_field="text",
        dataset_num_proc=2,
        max_length=cfg.max_seq_length,
        packing=cfg.packing,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Saved LoRA adapter ->", cfg.output_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--train_jsonl", type=str, default="phase1_data/data/processed/train.jsonl")
    p.add_argument("--val_jsonl", type=str, default="phase1_data/data/processed/val.jsonl")
    p.add_argument("--output_dir", type=str, default="outputs/qwen2.5-1.5b-lora-sft")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--load_in_4bit", type=str, default="true")
    p.add_argument("--bf16", type=str, default="true", help="Use bf16 (recommended on A100/H100).")
    p.add_argument("--fp16", type=str, default="false", help="Use fp16 (use only if bf16 unsupported).")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--packing", type=str, default="true")

    p.add_argument("--dry_run", type=str, default="false", help="If true, only validate data + tokenization.")
    args = p.parse_args()

    cfg = TrainConfig(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl if args.val_jsonl else None,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=_str2bool(args.load_in_4bit),
        bf16=_str2bool(args.bf16),
        fp16=_str2bool(args.fp16),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        packing=_str2bool(args.packing),
        dry_run=_str2bool(args.dry_run),
    )

    if cfg.dry_run:
        _dry_run_pipeline(cfg)
        return

    _train_with_unsloth(cfg)


if __name__ == "__main__":
    main()


