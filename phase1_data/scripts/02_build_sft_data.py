import argparse
import json
import os
import random
from typing import Dict, List, Literal, Optional, Tuple

Style = Literal["neutral", "sarcastic", "witty"]
SystemLang = Literal["auto", "en", "zh"]


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


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_system_prompt(style: Style, style_strength: float, system_lang: SystemLang) -> str:
    """
    Keep prompt language consistent with dataset language (EN dataset -> EN prompt).
    """
    if system_lang == "zh":
        base = (
            "你是一个百科问答助手。请优先给出准确、可验证的事实，避免编造。"
            "当信息不确定时，明确说明不确定并建议如何核实。"
        )
        if style == "neutral" or style_strength <= 0:
            return base
        flavor = (
            "语气要求：可以略带吐槽/冷幽默，但不得针对任何具体个人或群体进行羞辱、歧视或人身攻击。"
            "不要使用脏话或仇恨表达。仍需保持答案清晰、结构化。"
        )
        if style_strength >= 0.8:
            flavor += "幽默强度：中等偏强，但不要喧宾夺主。"
        elif style_strength >= 0.4:
            flavor += "幽默强度：轻度点缀即可。"
        else:
            flavor += "幽默强度：非常克制，几乎中性。"
        return base + "\n" + flavor

    base = (
        "You are a factual encyclopedic QA assistant. Prioritize accurate, verifiable facts and do not fabricate."
        " If information is uncertain, say so explicitly and suggest how to verify it."
    )
    if style == "neutral" or style_strength <= 0:
        return base
    flavor = (
        "Tone: you may add mild dry humor/sarcasm, but do NOT insult or demean any individual or group."
        " Avoid profanity, hate, and harassment. Keep answers clear and well-structured."
    )
    if style_strength >= 0.8:
        flavor += " Humor strength: medium-high, but don't let it overshadow the facts."
    elif style_strength >= 0.4:
        flavor += " Humor strength: light touch."
    else:
        flavor += " Humor strength: very subtle (almost neutral)."
    return base + "\n" + flavor


def _apply_style(answer: str, style: Style, style_strength: float, seed: int, system_lang: SystemLang) -> str:
    """
    MVP：不用额外 LLM 做改写，避免引入不稳定/成本。
    做法：在不改动事实内容的前提下，添加极轻量的口吻前后缀。
    """
    a = answer.strip()
    if style == "neutral" or style_strength <= 0:
        return a

    rnd = random.Random(seed)

    if system_lang == "zh":
        if style == "sarcastic":
            prefixes = [
                "简单说：",
                "一句话概括：",
                "你要的答案在这：",
                "行吧，结论是：",
                "别急着下结论，先看事实：",
            ]
            suffixes = [
                "（就这点事儿。）",
                "（别急，事实先放这。）",
                "（不难，但别瞎猜。）",
                "（吐槽归吐槽，事实没跑。）",
            ]
        else:  # witty
            prefixes = ["结论先行：", "快速科普：", "讲人话版："]
            suffixes = ["（到此为止，收工。）", "（这下应该清楚了。）", "（如果你还想更深，我也可以继续。）"]
    else:
        if style == "sarcastic":
            prefixes = ["In short: ", "Quick answer: ", "Here's what you want: "]
            suffixes = [" (That's it.)", " (Facts first, drama later.)", " (Not hard—just don't guess.)"]
        else:  # witty
            prefixes = ["Bottom line: ", "Mini explainer: ", "Plain-English version: "]
            suffixes = [" (And we're done.)", " (That should clear it up.)", " (Want the deeper version too?)"]

    # strength -> 触发概率
    p = min(max(style_strength, 0.0), 1.0)
    if rnd.random() < p:
        a = rnd.choice(prefixes) + a
    if rnd.random() < p * 0.8:
        a = a + rnd.choice(suffixes)
    return a


def _to_messages(
    row: Dict,
    system_prompt: str,
    style: Style,
    style_strength: float,
    seed: int,
    system_lang: SystemLang,
    max_context_chars: int,
) -> Dict:
    q = (row.get("question") or row.get("instruction") or row.get("prompt") or "").strip()
    a = (row.get("answer") or row.get("output") or row.get("response") or "").strip()
    if not q or not a:
        raise ValueError("Row missing question/answer fields")

    ctx = (row.get("context") or "").strip()
    if ctx:
        if max_context_chars and max_context_chars > 0 and len(ctx) > max_context_chars:
            ctx = ctx[:max_context_chars] + "…"
        if system_lang == "zh":
            user_content = f"请根据以下资料回答问题。\n资料：{ctx}\n问题：{q}"
        else:
            user_content = f"Answer the question based on the context.\nContext: {ctx}\nQuestion: {q}"
    else:
        user_content = q

    styled_a = _apply_style(a, style=style, style_strength=style_strength, seed=seed, system_lang=system_lang)
    return {
        "id": str(row.get("id", "")),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": styled_a},
        ],
    }


def _detect_lang_from_text(text: str) -> SystemLang:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def _should_drop(q: str, a: str, drop_if_contains: Optional[List[str]]) -> bool:
    if not drop_if_contains:
        return False
    hay = (q + "\n" + a).lower()
    return any(k.strip().lower() in hay for k in drop_if_contains if k.strip())


def build_sft(
    in_path: str,
    out_train: str,
    out_val: str,
    style: Style,
    style_strength: float,
    max_samples: int,
    val_ratio: float,
    seed: int,
    system_lang: SystemLang,
    drop_if_contains: Optional[List[str]],
    max_context_chars: int,
) -> Tuple[int, int]:
    rows = _read_jsonl(in_path)
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]

    if system_lang == "auto":
        probe_q = (rows[0].get("question") or rows[0].get("instruction") or rows[0].get("prompt") or "").strip() if rows else ""
        system_lang = _detect_lang_from_text(probe_q)

    system_prompt = _build_system_prompt(style=style, style_strength=style_strength, system_lang=system_lang)

    sft_rows: List[Dict] = []
    for i, r in enumerate(rows):
        rid = str(r.get("id") or f"row_{i}")
        r = dict(r)
        r["id"] = rid
        q = (r.get("question") or r.get("instruction") or r.get("prompt") or "").strip()
        a = (r.get("answer") or r.get("output") or r.get("response") or "").strip()
        if _should_drop(q, a, drop_if_contains):
            continue
        sft_rows.append(
            _to_messages(
                r,
                system_prompt,
                style,
                style_strength,
                seed=seed + i,
                system_lang=system_lang,
                max_context_chars=max_context_chars,
            )
        )

    n_val = max(1, int(len(sft_rows) * val_ratio)) if len(sft_rows) >= 10 else max(0, int(len(sft_rows) * val_ratio))
    val_rows = sft_rows[:n_val]
    train_rows = sft_rows[n_val:]

    _write_jsonl(out_train, train_rows)
    _write_jsonl(out_val, val_rows)

    # Preview samples (small, human-readable)
    preview_path = os.path.join(os.path.dirname(os.path.abspath(out_train)), "..", "..", "reports", "samples_preview.jsonl")
    preview_path = os.path.normpath(preview_path)
    _ensure_parent_dir(preview_path)
    preview = train_rows[:5] + val_rows[:5]
    _write_jsonl(preview_path, preview)

    return len(train_rows), len(val_rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=str, required=True, help="Input raw JSONL")
    p.add_argument("--out_train", type=str, required=True, help="Output train.jsonl")
    p.add_argument("--out_val", type=str, required=True, help="Output val.jsonl")
    p.add_argument("--style", type=str, default="sarcastic", choices=["neutral", "sarcastic", "witty"])
    p.add_argument("--style_strength", type=float, default=0.5, help="0..1, controls how often to add tone markers")
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--system_lang",
        type=str,
        default="auto",
        choices=["auto", "en", "zh"],
        help="Keep prompt/style language consistent with dataset language. 'auto' infers from first example.",
    )
    p.add_argument(
        "--drop_if_contains",
        type=str,
        default="",
        help="Comma-separated keywords; if (question+answer) contains any, drop the sample.",
    )
    p.add_argument(
        "--max_context_chars",
        type=int,
        default=800,
        help="If context exists, truncate to this many chars (0 disables). Helps avoid very long samples.",
    )
    args = p.parse_args()

    drop_list = [x.strip() for x in args.drop_if_contains.split(",") if x.strip()] if args.drop_if_contains else None
    n_train, n_val = build_sft(
        in_path=args.in_path,
        out_train=args.out_train,
        out_val=args.out_val,
        style=args.style,  # type: ignore[arg-type]
        style_strength=args.style_strength,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        system_lang=args.system_lang,  # type: ignore[arg-type]
        drop_if_contains=drop_list,
        max_context_chars=args.max_context_chars,
    )
    print(f"Wrote train={n_train} val={n_val}")


if __name__ == "__main__":
    main()



