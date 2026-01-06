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
    p.add_argument("--prompt", type=str, default=None, help="If provided, run once and exit. Otherwise enter interactive mode.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.1)
    args = p.parse_args()

    # This script is intended for Kaggle/Linux where Unsloth works.
    from unsloth import FastLanguageModel
    from peft import PeftModel
    from transformers import TextStreamer

    print(f"\n[1/3] 正在加载基础模型: {args.base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    print(f"[2/3] 正在挂载 Adapter: {args.adapter_dir}...")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    
    print(f"[3/3] 切换至推理模式...")
    model = FastLanguageModel.for_inference(model)
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    system_prompt = (
        "你是一个百科问答助手。请优先给出准确、可验证的事实，避免编造。"
        "当信息不确定时，明确说明不确定并建议如何核实。\n"
        "语气要求：可以略带吐槽/冷幽默，但不得针对任何具体个人或群体进行羞辱、歧视或人身攻击。"
        "不要使用脏话或仇恨表达。仍需保持答案清晰、结构化。"
        if args.system_lang == "zh"
        else "You are a factual encyclopedic QA assistant. Prioritize accurate, verifiable facts and do not fabricate."
        " If information is uncertain, say so explicitly and suggest how to verify it.\n"
        "Tone: you may add mild dry humor/sarcasm, but do NOT insult or demean any individual or group."
        " Avoid profanity, hate, and harassment. Keep answers clear and well-structured."
    )
    if args.system_lang == "zh":
        system_prompt += "幽默强度：中等偏强，但不要喧宾夺主。"
    else:
        system_prompt += " Humor strength: medium-high, but don't let it overshadow the facts."

    def run_inference(user_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = _render_messages(messages) + "\n<|assistant|>\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        print("\n" + "="*20 + " AI 回复 " + "="*20)
        _ = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        print("="*49 + "\n")

    if args.prompt:
        run_inference(args.prompt)
    else:
        print("\n" + "*"*40)
        print("已进入【极速交互模式】")
        print("- 输入 'exit' 或 'q' 退出")
        print("- 直接输入问题即可快速获得回复")
        print("*"*40 + "\n")
        
        while True:
            try:
                user_input = input("测试 Prompt >>> ")
            except EOFError:
                break
                
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input.strip():
                continue
            
            run_inference(user_input)


if __name__ == "__main__":
    main()
