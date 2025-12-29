import argparse
import json
import os
from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset
from tqdm import tqdm


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _iter_wiki_qa(split_iter: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """
    Normalizes WikiQA-like rows to: {id, question, answer, label}
    WikiQA contains multiple candidate answers per question with label.
    We keep only rows where label == 1 (correct answer).
    """
    for i, row in enumerate(split_iter):
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        label = int(row.get("label", 0))
        if not question or not answer:
            continue
        yield {
            "id": str(row.get("question_id") or f"wiki_qa_{i}"),
            "question": question,
            "answer": answer,
            "label": label,
        }


def _iter_cmrc2018(split_iter: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """
    CMRC2018 is a Chinese machine reading comprehension dataset.
    Normalizes rows to: {id, question, answer, context}
    """
    for i, row in enumerate(split_iter):
        q = (row.get("question") or "").strip()
        ctx = (row.get("context") or "").strip()
        answers = row.get("answers") or {}
        answer_text = ""
        if isinstance(answers, dict):
            texts = answers.get("text")
            if isinstance(texts, list) and texts:
                answer_text = (texts[0] or "").strip()
        if not q or not ctx or not answer_text:
            continue
        yield {
            "id": str(row.get("id") or f"cmrc2018_{i}"),
            "question": q,
            "answer": answer_text,
            "context": ctx,
        }


def download_dataset(dataset: str, out_path: str, limit: Optional[int]) -> int:
    ds = load_dataset(dataset)

    # Prefer "train" split; fall back to first available split.
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    split = ds[split_name]

    if dataset == "wiki_qa":
        rows = _iter_wiki_qa(split)
    elif dataset == "cmrc2018":
        rows = _iter_cmrc2018(split)
    else:
        # Best-effort passthrough for other datasets; user can implement a mapper later.
        rows = ({"id": str(i), **dict(r)} for i, r in enumerate(split))

    _ensure_parent_dir(out_path)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc=f"Writing {dataset}:{split_name}"):
            if dataset == "wiki_qa":
                # Keep only positive QA pairs.
                if int(row.get("label", 0)) != 1:
                    continue
                row.pop("label", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if limit is not None and n >= limit:
                break
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="wiki_qa", help="HF dataset name, e.g. wiki_qa")
    p.add_argument("--out", type=str, required=True, help="Output JSONL path")
    p.add_argument("--limit", type=int, default=None, help="Max rows to write")
    args = p.parse_args()

    n = download_dataset(args.dataset, args.out, args.limit)
    print(f"Wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()



