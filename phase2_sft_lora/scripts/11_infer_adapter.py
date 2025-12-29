import sys
import argparse
from typing import Any, Dict, List, Literal

SystemLang = Literal["zh", "en"]


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _render_messages(messages: List[Dict[str, Any]]) -> str:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--system_lang", type=str, choices=["zh", "en"], default="zh")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    args = p.parse_args()

    # This script is intended for Kaggle/Linux where Unsloth works.
    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = FastLanguageModel.for_inference(model)

    system_prompt = (
        "你是一个百科问答助手。请优先给出准确、可验证的事实，避免编造。"
        "当信息不确定时，明确说明不确定并建议如何核实。"
        "语气可以略带吐槽，但不要辱骂、不要歧视。"
        if args.system_lang == "zh"
        else "You are a factual encyclopedic QA assistant. Be accurate and avoid fabrication. Mild dry humor ok; no insults."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    text = _render_messages(messages) + "\n<|assistant|>\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded)


if __name__ == "__main__":
    main()


