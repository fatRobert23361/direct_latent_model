"""
分析 prosqa_train.json 中每条数据每个 reasoning step 的 token 数统计。
使用与训练相同的 GPT-2 tokenizer。
"""

import json
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict

DATA_PATH = "data/prosqa_train.json"
MODEL_ID = "openai-community/gpt2"

print(f"加载 tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"加载数据: {DATA_PATH}")
with open(DATA_PATH, "r") as f:
    data = json.load(f)

print(f"共 {len(data)} 条数据\n")

# step_idx -> list of token counts
step_token_counts = defaultdict(list)
# 所有 step 的 token 数（不区分位置）
all_token_counts = []

for item in data:
    steps = item.get("steps", [])
    for i, step in enumerate(steps):
        n_tokens = len(tokenizer.encode(step))
        step_token_counts[i].append(n_tokens)
        all_token_counts.append(n_tokens)

# ── 按 step 位置统计 ──────────────────────────────────────────────
print("=" * 60)
print("各 step 位置的 token 数统计（step_idx 从 0 开始）")
print("=" * 60)
print(f"{'Step':>6}  {'样本数':>8}  {'最小':>6}  {'最大':>6}  {'平均':>8}  {'中位数':>8}")
print("-" * 60)

max_step = max(step_token_counts.keys())
for i in range(max_step + 1):
    counts = step_token_counts[i]
    if not counts:
        continue
    print(
        f"{i:>6}  {len(counts):>8}  {min(counts):>6}  {max(counts):>6}"
        f"  {np.mean(counts):>8.2f}  {np.median(counts):>8.2f}"
    )

# ── 所有 step 汇总统计 ────────────────────────────────────────────
print("=" * 60)
print("全部 reasoning step 汇总统计")
print("=" * 60)
arr = np.array(all_token_counts)
print(f"  总 step 数:  {len(arr)}")
print(f"  最小 token: {arr.min()}")
print(f"  最大 token: {arr.max()}")
print(f"  平均 token: {arr.mean():.4f}")
print(f"  中位数:     {np.median(arr):.1f}")
print(f"  标准差:     {arr.std():.4f}")

# ── 每条数据的 step 数分布 ────────────────────────────────────────
step_lengths = [len(item.get("steps", [])) for item in data]
sl = np.array(step_lengths)
print()
print("=" * 60)
print("每条数据的 step 数量分布")
print("=" * 60)
print(f"  最小: {sl.min()}")
print(f"  最大: {sl.max()}")
print(f"  平均: {sl.mean():.4f}")
print(f"  中位数: {np.median(sl):.1f}")

# 分布直方图（文字版）
from collections import Counter
cnt = Counter(step_lengths)
print()
print(f"  {'step数':>6}  {'数据条数':>10}  {'占比':>8}")
for k in sorted(cnt):
    print(f"  {k:>6}  {cnt[k]:>10}  {cnt[k]/len(data)*100:>7.2f}%")
