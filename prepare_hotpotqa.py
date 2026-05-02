"""
prepare_hotpotqa.py

将 HotpotQA 转换为 coconut 训练格式：
    {"question": ..., "steps": [...], "answer": ...}

question = 完整 context（含干扰段落）+ 问题
steps    = supporting_facts 对应的关键句子（按原顺序）
answer   = 答案字符串

用法：
    python prepare_hotpotqa.py [--stats_only]
"""

import re
import json
import random
import argparse
import os
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer


def build_context_str(titles_and_sents):
    """
    将 [(title, [sent0, sent1, ...]), ...] 拼成单行字符串（段落间用 '. ' 分隔，去除多余连续句号）。

    输出示例：
        Scott Derrickson: Scott Derrickson (born 1966)... Ed Wood: Edward Davis Wood Jr....
    """
    parts = []
    for title, sents in titles_and_sents:
        body = " ".join(s.strip() for s in sents if s.strip())
        parts.append(f"{title}: {body}".rstrip(".").rstrip())
    text = ". ".join(parts)
    text = re.sub(r"\.(\s*\.)+", ".", text)
    return text


def select_paragraphs(context, supporting_titles, n_distractor=2, rng=None):
    """
    从 context 中选取段落：
      - supporting 相关的段落全部保留
      - 从剩余干扰段落中随机采样 n_distractor 篇
      - 打乱顺序后返回 [(title, sents), ...]

    context 格式（HuggingFace dict）：
        {'title': [...], 'sentences': [[...], ...]}
    supporting_titles: set of titles that are relevant
    """
    if rng is None:
        rng = random

    support_paras    = []
    distractor_paras = []
    for title, sents in zip(context["title"], context["sentences"]):
        if title in supporting_titles:
            support_paras.append((title, sents))
        else:
            distractor_paras.append((title, sents))

    # 随机采样干扰段落
    n = min(n_distractor, len(distractor_paras))
    sampled_distractors = rng.sample(distractor_paras, n)

    # 合并并打乱顺序
    all_paras = support_paras + sampled_distractors
    rng.shuffle(all_paras)
    return all_paras


def convert_sample(sample, n_distractor=2, rng=None):
    """
    转换单条 HotpotQA 样本。

    输入示例（HuggingFace 格式）：
    {
      "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
      "answer":   "yes",
      "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
      "context": {
        "title":     ["Scott Derrickson", "Ed Wood", "Marvel Comics", ...],
        "sentences": [["Scott Derrickson (born 1966)...", ...],
                      ["Edward Davis Wood Jr....", ...],
                      ["Marvel Comics is...", ...], ...]
      }
    }

    输出示例：
    {
      "question": "[Context]\nEd Wood: ...\nMarvel Comics: ...\nScott Derrickson: ...\n[Question] Were Scott ...",
      "steps":    ["Scott Derrickson (born 1966) is an American director, screenwriter and producer.",
                   "Edward Davis Wood Jr. (October 10, 1924) was an American filmmaker."],
      "answer":   "yes"
    }

    context 包含：2 篇 supporting 段落（全文）+ n_distractor 篇随机干扰段落（顺序打乱）
    """
    if rng is None:
        rng = random

    question = sample["question"].strip()
    answer   = sample["answer"].strip()
    if not question or not answer:
        return None

    sf_titles   = sample["supporting_facts"]["title"]
    sf_sent_ids = sample["supporting_facts"]["sent_id"]
    supporting_title_set = set(sf_titles)

    # 选取段落：supporting 全保留 + 随机 n_distractor 篇干扰，顺序打乱
    selected_paras = select_paragraphs(
        sample["context"], supporting_title_set, n_distractor=n_distractor, rng=rng
    )

    context_str  = build_context_str(selected_paras)
    question_str = re.sub(r"\.(\s*\.)+", ".", f"{context_str}. {question}")

    # 提取 supporting_facts 对应句子作为 steps（保持原始顺序）
    ctx_dict = {}
    for title, sents in zip(sample["context"]["title"], sample["context"]["sentences"]):
        if title not in ctx_dict:
            ctx_dict[title] = sents

    steps = []
    seen  = set()
    for title, sent_idx in zip(sf_titles, sf_sent_ids):
        key = (title, sent_idx)
        if key in seen:
            continue
        seen.add(key)
        sents = ctx_dict.get(title, [])
        if sent_idx < len(sents) and sents[sent_idx].strip():
            steps.append(sents[sent_idx].strip())

    if not steps:
        return None

    return {
        "question": question_str,
        "steps":    steps,
        "answer":   answer,
    }


def run_stats(data, tokenizer, label=""):
    """统计 question 字段的 token 长度分布。"""
    lengths = [len(tokenizer.encode(s["question"])) for s in data]
    over_512  = sum(1 for l in lengths if l > 512)
    over_768  = sum(1 for l in lengths if l > 768)
    over_1024 = sum(1 for l in lengths if l > 1024)
    total     = len(lengths)

    print(f"\n[{label}] 共 {total} 条")
    print(f"  token 长度  min={min(lengths)}  max={max(lengths)}  avg={sum(lengths)/total:.0f}")
    print(f"  > 512  : {over_512:5d} 条  ({over_512/total*100:.1f}%)")
    print(f"  > 768  : {over_768:5d} 条  ({over_768/total*100:.1f}%)")
    print(f"  > 1024 : {over_1024:5d} 条  ({over_1024/total*100:.1f}%)")
    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  default="data")
    parser.add_argument("--val_ratio",   type=float, default=0.5)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--model_id",    default="openai-community/gpt2",
                        help="用于统计 token 长度的 tokenizer")
    parser.add_argument("--n_distractor", type=int, default=2,
                        help="每条样本加入的干扰段落数量（默认 2）")
    parser.add_argument("--stats_only",  action="store_true",
                        help="只统计，不保存文件")
    args = parser.parse_args()

    random.seed(args.seed)

    print("正在下载 HotpotQA (distractor)...")
    dataset = load_dataset("hotpot_qa", "distractor")

    print(f"加载 tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- 转换 ----
    rng = random.Random(args.seed)

    print(f"\n处理 train split（{len(dataset['train'])} 条，n_distractor={args.n_distractor}）...")
    train_data, skipped = [], 0
    for s in dataset["train"]:
        r = convert_sample(s, n_distractor=args.n_distractor, rng=rng)
        if r: train_data.append(r)
        else: skipped += 1
    print(f"  有效 {len(train_data)} 条，跳过 {skipped} 条")

    print(f"\n处理 validation split（{len(dataset['validation'])} 条，n_distractor={args.n_distractor}）...")
    dev_data, skipped = [], 0
    for s in dataset["validation"]:
        r = convert_sample(s, n_distractor=args.n_distractor, rng=rng)
        if r: dev_data.append(r)
        else: skipped += 1
    print(f"  有效 {len(dev_data)} 条，跳过 {skipped} 条")

    random.shuffle(dev_data)
    cut       = int(len(dev_data) * args.val_ratio)
    val_data  = dev_data[:cut]
    test_data = dev_data[cut:]

    # ---- 统计 ----
    train_lens = run_stats(train_data, tokenizer, "train")
    val_lens   = run_stats(val_data,   tokenizer, "val")
    test_lens  = run_stats(test_data,  tokenizer, "test")

    all_lens = train_lens + val_lens + test_lens
    over_1024_total = sum(1 for l in all_lens if l > 1024)
    print(f"\n全部合计：{over_1024_total} / {len(all_lens)} 条超过 1024 tokens "
          f"({over_1024_total/len(all_lens)*100:.1f}%)")

    # ---- 保存 ----
    if not args.stats_only:
        os.makedirs(args.output_dir, exist_ok=True)
        for name, data in [("hotpot_train", train_data),
                            ("hotpot_valid", val_data),
                            ("hotpot_test",  test_data)]:
            path = os.path.join(args.output_dir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已保存 {path}")

    # ---- 样本预览 ----
    print("\n--- 样本预览 ---")
    for i, s in enumerate(train_data[:2]):
        print(f"\n[{i}]")
        print(f"  question (前200字): {s['question'][:200]}...")
        for j, step in enumerate(s["steps"]):
            print(f"  step{j+1}: {step}")
        print(f"  answer: {s['answer']}")


if __name__ == "__main__":
    main()
