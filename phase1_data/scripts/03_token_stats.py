import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from transformers import AutoTokenizer


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_text_for_tokenize(example: Dict) -> str:
    # For chat-style datasets: join messages in a deterministic way.
    msgs = example.get("messages")
    if isinstance(msgs, list):
        parts: List[str] = []
        for m in msgs:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if not role or not content:
                continue
            parts.append(f"{role.upper()}: {content}")
        return "\n".join(parts)

    # Fallback: try common fields
    for k in ["text", "prompt", "instruction", "input", "output", "response"]:
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return json.dumps(example, ensure_ascii=False)


@dataclass
class TokenReport:
    count: int
    min: int
    p50: int
    p90: int
    p95: int
    p99: int
    max: int
    mean: float
    over_1024: int
    over_2048: int
    over_4096: int


def _compute_report(lengths: List[int]) -> TokenReport:
    arr = np.array(lengths, dtype=np.int32)
    return TokenReport(
        count=int(arr.size),
        min=int(arr.min()) if arr.size else 0,
        p50=int(np.percentile(arr, 50)) if arr.size else 0,
        p90=int(np.percentile(arr, 90)) if arr.size else 0,
        p95=int(np.percentile(arr, 95)) if arr.size else 0,
        p99=int(np.percentile(arr, 99)) if arr.size else 0,
        max=int(arr.max()) if arr.size else 0,
        mean=float(arr.mean()) if arr.size else 0.0,
        over_1024=int((arr > 1024).sum()) if arr.size else 0,
        over_2048=int((arr > 2048).sum()) if arr.size else 0,
        over_4096=int((arr > 4096).sum()) if arr.size else 0,
    )


def token_stats(model: str, in_path: str, out_path: str, max_examples: int) -> Tuple[TokenReport, List[Dict]]:
    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    rows = _read_jsonl(in_path)
    if max_examples and max_examples > 0:
        rows = rows[:max_examples]

    lengths: List[int] = []
    # Keep a few longest samples for debugging/truncation decisions
    longest: List[Tuple[int, Dict]] = []

    for ex in rows:
        text = _extract_text_for_tokenize(ex)
        ids = tok(text, add_special_tokens=True, truncation=False)["input_ids"]
        n = int(len(ids))
        lengths.append(n)

        longest.append((n, {"id": ex.get("id"), "n_tokens": n, "preview": text[:500]}))
        longest.sort(key=lambda t: t[0], reverse=True)
        longest = longest[:10]

    rep = _compute_report(lengths)

    _ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model,
                "input": os.path.abspath(in_path),
                "report": rep.__dict__,
                "longest_examples": [x[1] for x in longest],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return rep, [x[1] for x in longest]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Tokenizer model id")
    p.add_argument("--in", dest="in_path", type=str, required=True, help="Input JSONL (processed)")
    p.add_argument("--out", dest="out_path", type=str, required=True, help="Output report json path")
    p.add_argument("--max_examples", type=int, default=2000)
    args = p.parse_args()

    rep, _ = token_stats(args.model, args.in_path, args.out_path, args.max_examples)
    print(
        "Token stats:",
        f"count={rep.count} mean={rep.mean:.1f} p95={rep.p95} max={rep.max} over_2048={rep.over_2048}",
    )


if __name__ == "__main__":
    main()



