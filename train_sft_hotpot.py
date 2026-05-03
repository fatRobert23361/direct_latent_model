"""
train_sft_hotpot.py

在 HotpotQA-COT 上对预训练 GPT-2 做带 COT 的监督微调（SFT）。

训练格式：
    {question}\\n{step1}\\n{step2}\\n...### {answer}<eos>
    labels: question 部分 -100，step + answer 部分计算 loss

评估格式：
    prompt = {question}\\n → 模型自由生成 COT + answer
    → 取最后一个 ### 之后的内容作为预测答案

每 epoch 评估：
    val / test 的 Exact Match (EM) 和 Token F1
    val / test 的 loss（完整序列）
    前 n_samples_upload 条测试样本的生成结果上传 wandb Table

配置文件：args/sft_hotpot.yaml

用法：
    python train_sft_hotpot.py
    python train_sft_hotpot.py --config args/sft_hotpot.yaml
"""

import argparse
import itertools
import json
import os
import re
import string
from collections import Counter

import torch
import wandb
import yaml
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel

from dataset import get_dataset
from utils import set_seed


# ---------------------------------------------------------------------------
# EM / F1（与 eval_baseline_hotpot.py 一致）
# ---------------------------------------------------------------------------

def normalize_answer(s):
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
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common      = Counter(pred_tokens) & Counter(gold_tokens)
    num_same    = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_answer(text):
    """取最后一个 ### 之后、第一个换行之前的内容作为预测答案。"""
    after = text.split("###")[-1] if "###" in text else text
    return after.split("\n")[0].strip()


# ---------------------------------------------------------------------------
# Dataset 构建
# ---------------------------------------------------------------------------

def build_train_features(raw_dataset, max_seq_len=1024):
    """
    将 get_dataset() 返回的 Dataset 转为 list of dict：
        input_ids = question_tokens + cot_tokens + answer_tokens
        labels    = [-100]*len(question_tokens) + cot_tokens + answer_tokens
    超过 max_seq_len 的样本直接丢弃（GPT-2 位置嵌入上限）。
    """
    features, skipped = [], 0
    for sample in raw_dataset:
        q   = sample["question_tokenized"]
        cot = list(itertools.chain.from_iterable(sample["steps_tokenized"]))
        ans = sample["answer_tokenized"]

        ids = q + cot + ans
        if len(ids) > max_seq_len:
            skipped += 1
            continue

        labels = [-100] * len(q) + cot + ans
        features.append({
            "input_ids":      ids,
            "labels":         labels,
            "attention_mask": [1] * len(ids),
        })
    if skipped:
        print(f"  [build_train_features] skipped {skipped} samples exceeding {max_seq_len} tokens")
    return features


def build_eval_prompts(raw_dataset):
    """评估 prompt：只含 question_tokens（让模型自由生成 COT + answer）。"""
    return [
        {
            "input_ids":      sample["question_tokenized"],
            "attention_mask": [1] * len(sample["question_tokenized"]),
        }
        for sample in raw_dataset
    ]


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------

def collate_train(batch, tokenizer, label_pad=-100):
    """右 padding，用于训练。"""
    max_len = max(len(f["input_ids"]) for f in batch)
    pad_id  = tokenizer.pad_token_id
    ids, lbls, masks = [], [], []
    for f in batch:
        n = max_len - len(f["input_ids"])
        ids.append(f["input_ids"]      + [pad_id]    * n)
        lbls.append(f["labels"]        + [label_pad] * n)
        masks.append(f["attention_mask"] + [0]        * n)
    return {
        "input_ids":      torch.tensor(ids,   dtype=torch.long),
        "labels":         torch.tensor(lbls,  dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def collate_eval(batch, tokenizer):
    """左 padding，用于 batch 生成。"""
    max_len = max(len(f["input_ids"]) for f in batch)
    pad_id  = tokenizer.pad_token_id
    ids, masks = [], []
    for f in batch:
        n = max_len - len(f["input_ids"])
        ids.append([pad_id] * n + f["input_ids"])
        masks.append([0]    * n + f["attention_mask"])
    return {
        "input_ids":      torch.tensor(ids,   dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Loss 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(model, raw_dataset, tokenizer, device, cfg):
    model.eval()
    features = build_train_features(raw_dataset)
    loader   = DataLoader(
        features,
        batch_size=cfg.get("batch_size_eval", 8),
        shuffle=False,
        collate_fn=lambda b: collate_train(b, tokenizer),
    )
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="  loss", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item()
        n     += 1
    return total / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# 准确率评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(model, raw_dataset, answers_list, tokenizer,
                      device, cfg, collect_samples=False):
    model.eval()
    n_upload = cfg.get("n_samples_upload", 50)
    prompts  = build_eval_prompts(raw_dataset)

    table = (
        wandb.Table(columns=[
            "idx", "question", "gt_cot", "generated_text",
            "gt_answer", "pred_answer", "em", "f1",
        ])
        if collect_samples else None
    )

    em_list, f1_list = [], []

    for i in tqdm(range(len(raw_dataset)), desc="  gen", leave=False):
        prompt    = prompts[i]
        gold      = answers_list[i]
        input_ids = torch.tensor([prompt["input_ids"]],      dtype=torch.long).to(device)
        attn_mask = torch.tensor([prompt["attention_mask"]], dtype=torch.long).to(device)

        gen_ids  = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=cfg.get("max_new_tokens_eval", 200),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids  = gen_ids[0, input_ids.shape[1]:]
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        pred     = extract_answer(gen_text)

        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)
        em_list.append(em)
        f1_list.append(f1)

        if collect_samples and i < n_upload:
            q_text = tokenizer.decode(prompt["input_ids"], skip_special_tokens=True)
            gt_cot = "\n".join(str(s) for s in raw_dataset[i]["steps"])
            table.add_data(i, q_text, gt_cot, gen_text, gold, pred, em, round(f1, 4))

    avg_em = sum(em_list) / len(em_list) if em_list else 0.0
    avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    return avg_em, avg_f1, table


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, val_em, save_path, name):
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                epoch,
            "val_em":               val_em,
        },
        os.path.join(save_path, f"{name}_best.pt"),
    )


# ---------------------------------------------------------------------------
# 训练主流程
# ---------------------------------------------------------------------------

def train(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Model ----
    model = GPT2LMHeadModel.from_pretrained(cfg["model_id"]).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ---- Resume ----
    resume_ckpt = None
    if cfg.get("resume_from_checkpoint"):
        print(f"Resuming from {cfg['resume_from_checkpoint']}")
        resume_ckpt = torch.load(cfg["resume_from_checkpoint"], map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])

    # ---- Datasets ----
    print("Loading datasets ...")
    raw_train = get_dataset(cfg["train_path"], tokenizer)
    raw_val   = get_dataset(cfg["val_path"],   tokenizer)
    raw_test  = get_dataset(cfg["test_path"],  tokenizer)

    val_answers  = [d["answer"].strip() for d in json.load(open(cfg["val_path"]))]
    test_answers = [d["answer"].strip() for d in json.load(open(cfg["test_path"]))]

    print("Building training features ...")
    train_features = build_train_features(raw_train)
    print(f"  train={len(train_features)}  val={len(raw_val)}  test={len(raw_test)}")

    train_loader = DataLoader(
        train_features,
        batch_size=cfg["batch_size_training"],
        shuffle=True,
        collate_fn=lambda b: collate_train(b, tokenizer),
    )

    # ---- Optimizer / Scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    if resume_ckpt and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])

    warmup_steps = cfg.get("warmup_steps", 200)
    scheduler    = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)),
    )

    # ---- WandB ----
    debug = cfg.get("debug", False)
    if not debug:
        wandb.init(project=cfg["project"], name=cfg["name"], config=cfg)

    grad_accum  = cfg.get("gradient_accumulation_steps", 4)
    num_epochs  = cfg.get("num_epochs", 10)
    start_epoch = (resume_ckpt.get("epoch", -1) + 1) if resume_ckpt else 0
    global_step = 0
    best_val_em = 0.0

    eval_kw = dict(tokenizer=tokenizer, device=device, cfg=cfg)

    # ---- Training loop ----
    for epoch in range(start_epoch, num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs - 1}")

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            loss  = out.loss / grad_accum
            loss.backward()

            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % 10 == 0 and not debug:
                wandb.log({
                    "train/loss": out.loss.item(),
                    "meta/epoch": epoch,
                    "meta/step":  global_step,
                    "meta/lr":    optimizer.param_groups[0]["lr"],
                })

            pbar.set_postfix({"loss": f"{out.loss.item():.4f}"})
            global_step += 1

        # ================================================================
        # Per-epoch evaluation
        # ================================================================
        print(f"\n[Epoch {epoch}] Evaluating ...")

        print("  Val loss ...")
        val_loss = evaluate_loss(model, raw_val, **eval_kw)
        print("  Val EM/F1 ...")
        val_em, val_f1, _ = evaluate_accuracy(
            model, raw_val, val_answers, **eval_kw, collect_samples=False,
        )

        print("  Test loss ...")
        test_loss = evaluate_loss(model, raw_test, **eval_kw)
        print("  Test EM/F1 + samples ...")
        test_em, test_f1, test_table = evaluate_accuracy(
            model, raw_test, test_answers, **eval_kw,
            collect_samples=not debug,
        )

        print(
            f"[Epoch {epoch}]  "
            f"val_em={val_em*100:.2f}%  val_f1={val_f1*100:.2f}%  val_loss={val_loss:.4f}  |  "
            f"test_em={test_em*100:.2f}%  test_f1={test_f1*100:.2f}%  test_loss={test_loss:.4f}"
        )

        if not debug:
            log_dict = {
                "eval/val_em":    val_em,
                "eval/val_f1":    val_f1,
                "eval/val_loss":  val_loss,
                "eval/test_em":   test_em,
                "eval/test_f1":   test_f1,
                "eval/test_loss": test_loss,
                "meta/epoch":     epoch,
            }
            if test_table is not None:
                log_dict["eval/test_samples"] = test_table
            wandb.log(log_dict)

        if val_em > best_val_em:
            best_val_em = val_em
            save_checkpoint(model, optimizer, epoch, val_em,
                            cfg["save_path"], cfg["name"])
            print(f"  → New best val EM {val_em*100:.2f}% — checkpoint saved.")

    if not debug:
        wandb.finish()

    print(f"\nTraining complete. Best val EM: {best_val_em*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="args/sft_hotpot.yaml")
    args = parser.parse_args()
    train(args.config)
