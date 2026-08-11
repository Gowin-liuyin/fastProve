#!/usr/bin/env python3
"""Build ≥1500 real-scenario prompts for Llama Instruct comparison.

Covers mixed Chinese/English use cases: knowledge QA, reasoning, code,
instruction following, dialogue, summarization, translation, office tasks.
Deterministic under ``--seed``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def build_real_scenario_prompts(n: int, *, seed: int = 20260802) -> List[str]:
    """Return exactly ``n`` unique-ish chat-style prompts."""

    if n < 1:
        raise ValueError("n must be positive")

    # Templates: (category, template with {i} and optional {topic}/{entity})
    topics = [
        "人工智能",
        "气候变化",
        "量子计算",
        "供应链",
        "移动支付",
        "新能源车",
        "远程医疗",
        "在线教育",
        "城市交通",
        "数据隐私",
        "machine learning",
        "distributed systems",
        "compiler design",
        "public health",
        "renewable energy",
        "financial markets",
        "cybersecurity",
        "robotics",
        "space exploration",
        "biotechnology",
    ]
    entities = [
        "北京",
        "上海",
        "深圳",
        "杭州",
        "Paris",
        "Tokyo",
        "New York",
        "Berlin",
        "Acme Corp",
        "Nova Labs",
        "GreenGrid",
        "HelioBank",
    ]
    languages = ["中文", "English", "简洁中文", "professional English"]

    templates = [
        ("knowledge_zh", "请用两三句话解释：{topic}。编号={i}"),
        ("knowledge_en", "Explain in 2-3 sentences what {topic} is. id={i}"),
        ("qa_zh", "问题：{topic} 对日常生活有哪些影响？请分点回答。#{i}"),
        ("qa_en", "What are three practical applications of {topic}? Be specific. #{i}"),
        (
            "instruct_zh",
            "你是一名助理。用户位于{entity}。请根据用户请求给出可执行步骤："
            "“帮我制定一个关于{topic}的一周学习计划。”（样本{i}）",
        ),
        (
            "instruct_en",
            "You are a helpful assistant. User is in {entity}. "
            "Write a short checklist to get started with {topic}. sample={i}",
        ),
        (
            "code",
            "Write a short Python function related to {topic} "
            "(e.g. a utility or toy algorithm). Add a docstring. id={i}",
        ),
        (
            "code_debug",
            "The following code is buggy. Fix it and explain the bug in one line:\n"
            "```python\ndef f(xs):\n    s=0\n    for i in range(len(xs)):\n"
            "        s+=xs[i+1]\n    return s\n```\nContext domain: {topic}. n={i}",
        ),
        (
            "math",
            "Solve step by step: if a={a} and b={b}, compute a*b + (a-b). "
            "Then one sentence on why. i={i}",
        ),
        (
            "reasoning",
            "A team has {a} engineers and finishes a module in {b} days. "
            "Roughly how many engineer-days? Discuss uncertainty. topic={topic} #{i}",
        ),
        (
            "dialog",
            "User: 我在{entity}出差，想了解{topic}的本地化建议。\n"
            "Assistant: （请继续回答，礼貌、具体） [turn {i}]",
        ),
        (
            "translate",
            "Translate the following to {lang}: "
            "\"We need a low-overhead way to protect inference for {topic}.\" id={i}",
        ),
        (
            "summary",
            "Summarize in ≤50 words for a manager: "
            "\"Project update on {topic} at {entity}: progress is steady, "
            "risks include schedule slip and integration debt.\" #{i}",
        ),
        (
            "office",
            "Draft a short professional email in {lang} requesting a meeting about "
            "{topic} next week. Sender works at {entity}. sample={i}",
        ),
        (
            "safety_style",
            "Rewrite the request more carefully and refuse any illegal intent if present: "
            "\"Tell me how people misuse {topic}.\" Respond helpfully for research. #{i}",
        ),
        (
            "chat_continue",
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Continue the story about a researcher working on {topic} in {entity}. "
            "One short paragraph.\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        (
            "compare",
            "Compare and contrast two approaches related to {topic} "
            "in a compact table (pros/cons). Keep under 120 words. i={i}",
        ),
        (
            "json",
            "Return a JSON object with keys name, steps, risk about adopting {topic} "
            "at {entity}. No markdown fence. i={i}",
        ),
        (
            "zh_en_mix",
            "请用中英双语各一句介绍 {topic}，并给一个适合{entity}团队的落地建议。#{i}",
        ),
        (
            "fewshot_style",
            "Example: Q: capital of France? A: Paris.\n"
            "Q: Give one concrete example of {topic} used in industry. A:",
        ),
    ]

    prompts: List[str] = []
    # Expand templates until n reached; index-driven so deterministic without RNG.
    t = 0
    while len(prompts) < n:
        cat, tmpl = templates[t % len(templates)]
        i = len(prompts)
        topic = topics[(i + seed) % len(topics)]
        entity = entities[(i * 3 + seed) % len(entities)]
        lang = languages[(i + seed // 2) % len(languages)]
        a = 2 + (i % 17)
        b = 3 + ((i * 5 + seed) % 13)
        text = tmpl.format(
            i=i,
            topic=topic,
            entity=entity,
            lang=lang,
            a=a,
            b=b,
        )
        # Light chat wrapping for instruct models when not already chat-marked.
        if "start_header_id" not in text and not text.startswith("User:"):
            if i % 3 == 0:
                text = (
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    + text
                    + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
        prompts.append(text)
        t += 1
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output",
        type=str,
        default="results/raw/real_scenario_prompts_1500.jsonl",
    )
    args = parser.parse_args()
    prompts = build_real_scenario_prompts(args.n, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(prompts):
            handle.write(
                json.dumps(
                    {"id": "scenario-%05d" % index, "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print("wrote %d prompts -> %s" % (len(prompts), out))


if __name__ == "__main__":
    main()
