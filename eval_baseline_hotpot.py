"""
eval_baseline_hotpot.py

用原始 GPT-2（无任何微调）在 HotpotQA 测试集上做 baseline 评估。

输入格式：
    {question}\n###
    （question 字段已含完整 context，末尾接 ### 提示模型输出答案）

评估指标（HotpotQA 官方标准）：
    - Exact Match (EM)：预测答案与 ground truth 完全一致（忽略大小写/标点）
    - Token-level F1：预测与 ground truth 的词级 F1

用法：
    python eval_baseline_hotpot.py
    python eval_baseline_hotpot.py --test_path data/hotpot_test.json --num_samples 500
"""

import argparse
import json
import re
import string
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel


# ---------------------------------------------------------------------------
# 答案规范化与评估指标（来自 HotpotQA 官方评估代码）
# ---------------------------------------------------------------------------

def normalize_answer(s):
    """lowercase、去冠词、去标点、合并空格。"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred, gold):
    pred_tokens  = normalize_answer(pred).split()
    gold_tokens  = normalize_answer(gold).split()
    common       = Counter(pred_tokens) & Counter(gold_tokens)
    num_same     = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 从生成文本中提取答案
# ---------------------------------------------------------------------------

def extract_answer(generated_text):
    """
    取 ### 之后、第一个换行之前的内容作为预测答案。

    生成文本示例：
        "### Arthur's Magazine\n\n..."  →  "Arthur's Magazine"
        "### yes\n"                     →  "yes"
    """
    if "###" in generated_text:
        after = generated_text.split("###")[-1]
    else:
        after = generated_text
    # 取第一行，去首尾空白
    answer = after.split("\n")[0].strip()
    return answer


# ---------------------------------------------------------------------------
# 主评估逻辑
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 加载模型 ----
    print(f"加载 GPT-2: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"   # 生成时左 padding

    model = GPT2LMHeadModel.from_pretrained(args.model_id).to(device)
    model.eval()
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ---- 加载测试集 ----
    print(f"加载测试集: {args.test_path}")
    raw = json.load(open(args.test_path, encoding="utf-8"))
    if args.num_samples:
        raw = raw[:args.num_samples]
    print(f"共 {len(raw)} 条样本")

    em_scores, f1_scores = [], []
    preview_shown = 0

    for sample in tqdm(raw, desc="Evaluating"):
        gold   = sample["answer"].strip()
        # 拼接 prompt：question 字段已含 context，末尾加 "\n### " 引导答案
        prompt = sample["question"].rstrip() + "\n### "

        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(device)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,          # 贪心解码，与训练后评估保持一致
            pad_token_id=tokenizer.eos_token_id,
        )

        # 只解码新生成的 token
        new_ids   = output_ids[0, input_ids.shape[1]:]
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)
        pred      = extract_answer(generated)

        em  = exact_match(pred, gold)
        f1  = token_f1(pred, gold)
        em_scores.append(em)
        f1_scores.append(f1)

        # 打印前 N 条样本详情
        if preview_shown < args.num_preview:
            print(f"\n--- 样本 {preview_shown + 1} ---")
            print(f"  Context+Q (前150字): {prompt[:150]}...")
            print(f"  Gold   : {gold}")
            print(f"  Pred   : {pred!r}")
            print(f"  EM={em}  F1={f1:.3f}")
            preview_shown += 1

    # ---- 汇总 ----
    avg_em = sum(em_scores) / len(em_scores) * 100
    avg_f1 = sum(f1_scores) / len(f1_scores) * 100

    print("\n" + "=" * 50)
    print(f"GPT-2 Baseline on HotpotQA")
    print(f"  测试样本数 : {len(raw)}")
    print(f"  Exact Match: {avg_em:.2f}%")
    print(f"  Token F1   : {avg_f1:.2f}%")
    print("=" * 50)

    if args.output_json:
        result = {
            "model":        args.model_id,
            "test_path":    args.test_path,
            "num_samples":  len(raw),
            "exact_match":  avg_em,
            "token_f1":     avg_f1,
        }
        import os
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存: {args.output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",         default="openai-community/gpt2")
    parser.add_argument("--test_path",        default="data/hotpot_test.json")
    parser.add_argument("--num_samples",      type=int, default=None,
                        help="只跑前 N 条，不设则跑全部")
    parser.add_argument("--max_input_tokens", type=int, default=900,
                        help="输入截断长度（GPT-2 上限 1024，留余量给生成）")
    parser.add_argument("--max_new_tokens",   type=int, default=30,
                        help="答案生成最大 token 数（HotpotQA 答案通常 <10 tokens）")
    parser.add_argument("--num_preview",      type=int, default=3,
                        help="打印前 N 条样本的详细预测")
    parser.add_argument("--output_json",      default="results/baseline_hotpot.json")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
