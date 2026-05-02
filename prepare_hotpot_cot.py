"""
prepare_hotpot_cot.py

将 fsiddiqui2/hotpot-qa-cot-reasoning 转换为 coconut 训练格式：
    {"question": ..., "steps": [...], "answer": ...}

question = [Context]\n{supporting 段落 + n_distractor 篇干扰段落（随机打乱）}\n[Question] {问题}
steps    = reasoning_trace 按 "Step N:" 拆分后的各步骤（去掉前缀标签）
answer   = 答案字符串

相比 prepare_hotpotqa.py 的改进：
  - steps 来自 LLM 生成的完整推理链（reasoning_trace），质量更高
  - COT 包含推理过程而非仅原始支撑句

用法：
    python prepare_hotpot_cot.py
    python prepare_hotpot_cot.py --n_distractor 1 --stats_only
"""

import re
import json
import random
import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# 段落拼接与筛选（与 prepare_hotpotqa.py 逻辑相同）
# ---------------------------------------------------------------------------

def build_context_str(titles_and_sents):
    """[(title, [sent0, sent1, ...]), ...] → 单行字符串（段落间用 '. ' 分隔，去除多余连续句号）"""
    parts = []
    for title, sents in titles_and_sents:
        body = " ".join(s.strip() for s in sents if s.strip())
        # 去掉段落末尾已有的句号，统一由 join 添加
        parts.append(f"{title}: {body}".rstrip(".").rstrip())
    text = ". ".join(parts)
    # 折叠连续句号（含中间有空格的情况），如 ".."/". ." → "."
    text = re.sub(r"\.(\s*\.)+", ".", text)
    return text


def select_paragraphs(context, supporting_titles, n_distractor=2, rng=None):
    """
    保留所有 supporting 段落，随机抽取 n_distractor 篇干扰段落，顺序打乱。
    context 格式：{'title': [...], 'sentences': [[...], ...]}
    """
    if rng is None:
        rng = random

    support_paras, distractor_paras = [], []
    for title, sents in zip(context["title"], context["sentences"]):
        if title in supporting_titles:
            support_paras.append((title, sents))
        else:
            distractor_paras.append((title, sents))

    n = min(n_distractor, len(distractor_paras))
    sampled = rng.sample(distractor_paras, n)
    all_paras = support_paras + sampled
    rng.shuffle(all_paras)
    return all_paras


# ---------------------------------------------------------------------------
# reasoning_trace 解析
# ---------------------------------------------------------------------------

_MARKDOWN = re.compile(r"\*{1,2}([^*]+)\*{1,2}")


def parse_reasoning_trace(trace):
    """
    将 reasoning_trace 按 "Step N:" 拆分，清洗后返回步骤列表。

    清洗：
      1. 折叠换行/连续空白为单个空格（修复 Therefore 嵌在最后一步的换行问题）
      2. 去掉 markdown *...* / **...** 标记
    """
    parts = re.split(r"Step\s+\d+\s*:\s*", trace, flags=re.IGNORECASE)
    steps = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()          # 折叠换行/多余空白
        p = _MARKDOWN.sub(r"\1", p)                  # 去掉 markdown 标记
        if p:
            steps.append(p)
    return steps


# ---------------------------------------------------------------------------
# 单样本转换
# ---------------------------------------------------------------------------

def convert_sample(sample, n_distractor=2, rng=None):
    if rng is None:
        rng = random

    question = sample["question"].strip()
    answer   = sample["answer"].strip()
    if not question or not answer:
        return None

    supporting_title_set = set(sample["supporting_facts"]["title"])

    selected_paras = select_paragraphs(
        sample["context"], supporting_title_set,
        n_distractor=n_distractor, rng=rng,
    )
    context_str  = build_context_str(selected_paras)
    question_str = re.sub(r"\.(\s*\.)+", ".", f"{context_str}. {question}")

    steps = parse_reasoning_trace(sample["reasoning_trace"])
    if not steps:
        return None

    return {"question": question_str, "steps": steps, "answer": answer}


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def run_stats(data, tokenizer, label=""):
    lengths   = [len(tokenizer.encode(s["question"])) for s in data]
    step_lens = [sum(len(tokenizer.encode(st)) for st in s["steps"]) for s in data]
    over_512  = sum(1 for l in lengths if l > 512)
    over_768  = sum(1 for l in lengths if l > 768)
    over_1024 = sum(1 for l in lengths if l > 1024)
    total     = len(lengths)

    print(f"\n[{label}] 共 {total} 条")
    print(f"  question token 长度  min={min(lengths)}  max={max(lengths)}  avg={sum(lengths)/total:.0f}")
    print(f"  steps   token 长度   min={min(step_lens)}  max={max(step_lens)}  avg={sum(step_lens)/total:.0f}")
    print(f"  question > 512  : {over_512:5d} 条  ({over_512/total*100:.1f}%)")
    print(f"  question > 768  : {over_768:5d} 条  ({over_768/total*100:.1f}%)")
    print(f"  question > 1024 : {over_1024:5d} 条  ({over_1024/total*100:.1f}%)")
    return lengths


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",    default="data")
    parser.add_argument("--val_ratio",     type=float, default=0.05,
                        help="验证集占总数据比例（默认 5%%）")
    parser.add_argument("--test_ratio",    type=float, default=0.05,
                        help="测试集占总数据比例（默认 5%%）")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--model_id",      default="openai-community/gpt2",
                        help="用于 token 统计的 tokenizer")
    parser.add_argument("--n_distractor",  type=int,   default=2,
                        help="每条样本加入的干扰段落数（默认 2）")
    parser.add_argument("--stats_only",    action="store_true",
                        help="只统计，不保存文件")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("正在下载 fsiddiqui2/hotpot-qa-cot-reasoning ...")
    dataset  = load_dataset("fsiddiqui2/hotpot-qa-cot-reasoning")
    raw_data = dataset["train"]
    print(f"原始样本数: {len(raw_data)}")

    print(f"\n加载 tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- 转换 ----
    print(f"\n转换样本（n_distractor={args.n_distractor}）...")
    all_data, skipped = [], 0
    for s in raw_data:
        r = convert_sample(s, n_distractor=args.n_distractor, rng=rng)
        if r:
            all_data.append(r)
        else:
            skipped += 1
    print(f"有效 {len(all_data)} 条，跳过 {skipped} 条")

    # ---- 过滤 question token 长度 > 768 的样本 ----
    before = len(all_data)
    all_data = [
        s for s in all_data
        if len(tokenizer.encode(s["question"])) <= 768
    ]
    print(f"过滤 question>768 tokens：{before - len(all_data)} 条删除，剩余 {len(all_data)} 条")

    # ---- 划分 train / val / test ----
    rng.shuffle(all_data)
    n_total = len(all_data)
    n_val   = int(n_total * args.val_ratio)
    n_test  = int(n_total * args.test_ratio)

    val_data   = all_data[:n_val]
    test_data  = all_data[n_val : n_val + n_test]
    train_data = all_data[n_val + n_test :]

    print(f"\n划分结果：train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

    # ---- 统计 ----
    run_stats(train_data, tokenizer, "train")
    run_stats(val_data,   tokenizer, "val")
    run_stats(test_data,  tokenizer, "test")

    # ---- 保存 ----
    if not args.stats_only:
        os.makedirs(args.output_dir, exist_ok=True)
        for name, data in [
            ("hotpot_cot_train", train_data),
            ("hotpot_cot_valid", val_data),
            ("hotpot_cot_test",  test_data),
        ]:
            path = os.path.join(args.output_dir, f"{name}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已保存 {path}（{len(data)} 条）")

    # ---- 样本预览 ----
    print("\n--- 样本预览 ---")
    for i, s in enumerate(train_data[:3]):
        print(f"\n[{i}]")
        print(f"  question (前250字): {s['question'][:250]}...")
        for j, step in enumerate(s["steps"]):
            print(f"  step{j+1}: {step[:120]}")
        print(f"  answer: {s['answer']}")


if __name__ == "__main__":
    main()
