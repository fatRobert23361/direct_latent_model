"""
train_direct_latent.py

训练 DirectLatentModel：固定数量 latent token + 端到端联合训练（无 detach）。
配置文件：args/direct_latent.yaml

每个 epoch 评估：
  - val / test 的 answer accuracy
  - val / test 的 coconut_loss + translator_loss
  - test 前 n_translations_upload 条样本的翻译内容（上传 wandb Table）

用法：
    python train_direct_latent.py
    python train_direct_latent.py --config args/direct_latent.yaml
"""

import argparse
import json
import os

import torch
import wandb
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel
import yaml

from dataset import get_dataset
from direct_latent_dataset import (
    DirectLatentCollator,
    get_direct_latent_eval_dataset,
    get_direct_latent_train_dataset,
)
from direct_latent_model import DirectLatentModel
from translator_v3 import CoconutTranslator


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_best_checkpoint(model, save_path, name, epoch, val_acc):
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
        },
        os.path.join(save_path, f"{name}_best.pt"),
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _greedy_from_forward(model, fwd_outputs, device, max_new_tokens):
    """从已有的 forward 输出贪心解码，避免重跑 backbone。"""
    inputs_embeds = fwd_outputs.inputs_embeds
    next_token = torch.argmax(fwd_outputs.logits[0, -1]).item()
    generated = [next_token]
    curr_embeds = inputs_embeds

    for _ in range(max_new_tokens - 1):
        emb = model.embedding(torch.tensor([[next_token]], device=device))
        curr_embeds = torch.cat([curr_embeds, emb], dim=1)
        out = model.base_causallm(inputs_embeds=curr_embeds)
        next_token = torch.argmax(out.logits[0, -1]).item()
        if next_token == model.eos_token_id:
            break
        generated.append(next_token)

    return torch.tensor([generated], device=device)


@torch.no_grad()
def evaluate_loss(model, raw_dataset, tokenizer, n_latent, latent_id,
                  start_id, end_id, device, cfg):
    """
    在完整序列（含答案和 translator_labels）上计算平均损失。
    返回 (avg_coconut_loss, avg_translator_loss)。
    """
    model.eval()
    num_eval = min(cfg.get("num_eval_samples", 300), len(raw_dataset))
    ds = get_direct_latent_train_dataset(
        base_dataset=raw_dataset.select(range(num_eval)),
        n_latent=n_latent,
        start_id=start_id, latent_id=latent_id, end_id=end_id,
        eos_id=tokenizer.eos_token_id,
        shuffle=False,
    )
    collator = DirectLatentCollator(tokenizer=tokenizer, latent_id=latent_id)
    loader = DataLoader(ds, batch_size=cfg["batch_size_training"],
                        shuffle=False, collate_fn=collator)

    total_coconut, total_translator, n_batches = 0.0, 0.0, 0
    for batch in tqdm(loader, desc="  loss", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        outputs = model(**batch)
        total_coconut    += outputs.coconut_loss.item()
        total_translator += outputs.translator_loss.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0
    return total_coconut / n_batches, total_translator / n_batches


@torch.no_grad()
def evaluate_accuracy(model, raw_dataset, answers_list, tokenizer,
                      n_latent, latent_id, start_id, end_id, device, cfg,
                      collect_translations=False):
    """
    逐样本生成答案，计算 answer accuracy。

    collect_translations=True 时，对前 n_translations_upload 条样本额外运行
    翻译器，将 (question, gt_chain, predicted_chain, gt_answer, pred_answer)
    记录到 wandb.Table 并返回。

    返回 (accuracy, table_or_None)。
    """
    model.eval()
    n_upload = cfg.get("n_translations_upload", 100)
    num_eval = min(cfg.get("num_eval_samples", 300), len(raw_dataset))
    eval_raw = raw_dataset.select(range(num_eval))

    eval_ds = get_direct_latent_eval_dataset(
        base_dataset=eval_raw,
        n_latent=n_latent, start_id=start_id, latent_id=latent_id, end_id=end_id,
    )
    for col in ["answer", "steps"]:
        if col in eval_ds.column_names:
            eval_ds = eval_ds.remove_columns(col)

    collator = DirectLatentCollator(tokenizer=tokenizer, latent_id=latent_id)
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collator)

    table = (
        wandb.Table(columns=[
            "idx", "question",
            "gt_chain", "predicted_chain",
            "gt_answer", "predicted_answer", "answer_match",
        ])
        if collect_translations else None
    )

    correct, total = 0, 0

    for i, batch in enumerate(tqdm(loader, desc="  gen", leave=False)):
        idx = batch["idx"][0].item()
        answer = answers_list[idx]
        input_ids = batch["input_ids"].to(device)
        max_new = cfg.get("max_new_tokens_eval", 150)

        if collect_translations and i < n_upload:
            # 单次 forward 取得 latent_states，再贪心解码
            latent_pos = (input_ids == latent_id).nonzero()
            first_lat = latent_pos[0, 1].item() if len(latent_pos) > 0 else input_ids.shape[1]
            context_ids = input_ids[:, :first_lat]

            fwd = model.forward(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids.clone(),
                position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
            )
            gen_ids = _greedy_from_forward(model, fwd, device, max_new)
            predicted_chain = model.translate_latents(fwd.latent_states, context_ids, tokenizer)
            predicted_chain_text = predicted_chain[0] if predicted_chain else ""

            gt_steps = eval_raw[i]["steps"]
            gt_chain_text = " ".join(str(s) for s in gt_steps)
            question_text = tokenizer.decode(context_ids[0], skip_special_tokens=True)
        else:
            gen_ids = model.generate(input_ids, tokenizer,
                                     max_new_tokens=max_new, show_thoughts=False)

        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred = text.split("#")[-1].replace(",", "").strip()
        is_correct = pred == answer
        correct += int(is_correct)
        total += 1

        if collect_translations and i < n_upload:
            table.add_data(
                idx, question_text,
                gt_chain_text, predicted_chain_text,
                answer, pred, is_correct,
            )

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, table


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    special_tokens = ["<latent>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    latent_id  = tokenizer.convert_tokens_to_ids("<latent>")
    start_id   = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id     = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    pad_id     = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    vocab_size = len(tokenizer)
    print(f"latent_id={latent_id}, start_id={start_id}, end_id={end_id}, vocab_size={vocab_size}")

    # ---- Base model ----
    base_model = GPT2LMHeadModel.from_pretrained(cfg["model_id"])
    base_model.resize_token_embeddings(vocab_size)

    # ---- Translator ----
    translator = CoconutTranslator(
        hidden_size=base_model.config.n_embd,
        vocab_size=vocab_size,
        start_id=start_id, end_id=end_id,
        pad_id=pad_id, eos_id=tokenizer.eos_token_id,
        mode="context_latent",
    )

    # ---- Model ----
    n_latent = cfg.get("n_latent", 6)
    model = DirectLatentModel(
        base_causallm=base_model,
        translator=translator,
        n_latent=n_latent,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        lambda_translator=cfg.get("lambda_translator", 0.5),
    ).to(device)

    # ---- Resume ----
    resume_ckpt = None
    if cfg.get("resume_from_checkpoint"):
        print(f"Resuming from {cfg['resume_from_checkpoint']}")
        resume_ckpt = torch.load(cfg["resume_from_checkpoint"], map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])

    # ---- Datasets ----
    print("Loading datasets...")
    raw_train = get_dataset(cfg["train_path"], tokenizer)
    raw_val   = get_dataset(cfg["val_path"],   tokenizer)
    raw_test  = get_dataset(cfg["test_path"],  tokenizer)

    # 预读答案列表（避免每 epoch 重复读文件）
    val_answers = [
        d["answer"].replace(",", "").strip()
        for d in json.load(open(cfg["val_path"]))
    ]
    test_answers = [
        d["answer"].replace(",", "").strip()
        for d in json.load(open(cfg["test_path"]))
    ]

    train_ds = get_direct_latent_train_dataset(
        base_dataset=raw_train,
        n_latent=n_latent,
        start_id=start_id, latent_id=latent_id, end_id=end_id,
        eos_id=tokenizer.eos_token_id,
        shuffle=True,
    )
    collator = DirectLatentCollator(tokenizer=tokenizer, latent_id=latent_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size_training"],
        shuffle=True,
        collate_fn=collator,
    )

    # ---- Optimizer / Scheduler ----
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if resume_ckpt and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])

    warmup_steps = cfg.get("warmup_steps", 100)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)),
    )

    # ---- WandB ----
    debug = cfg.get("debug", False)
    if not debug:
        wandb.init(project=cfg["project"], name=cfg["name"], config=cfg)

    global_step = 0
    best_val_acc = 0.0
    start_epoch = (resume_ckpt.get("epoch", -1) + 1) if resume_ckpt else 0
    num_epochs  = cfg.get("num_epochs", 50)
    grad_accum  = cfg.get("gradient_accumulation_steps", 4)

    eval_args = dict(
        tokenizer=tokenizer, n_latent=n_latent,
        latent_id=latent_id, start_id=start_id, end_id=end_id,
        device=device, cfg=cfg,
    )

    # ---- Training loop ----
    for epoch in range(start_epoch, num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs - 1}")

        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % 10 == 0 and not debug:
                wandb.log({
                    "train/total_loss":      outputs.loss.item(),
                    "train/coconut_loss":    outputs.coconut_loss.item(),
                    "train/translator_loss": outputs.translator_loss.item(),
                    "meta/epoch":            epoch,
                    "meta/step":             global_step,
                    "meta/lr":               optimizer.param_groups[0]["lr"],
                })

            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})
            global_step += 1

        # ================================================================
        # Per-epoch evaluation
        # ================================================================
        print(f"\n[Epoch {epoch}] Evaluating ...")

        # ---- Val loss ----
        print("  Val loss ...")
        val_coconut_loss, val_trans_loss = evaluate_loss(
            model, raw_val, **eval_args
        )

        # ---- Val accuracy ----
        print("  Val accuracy ...")
        val_acc, _ = evaluate_accuracy(
            model, raw_val, val_answers, **eval_args,
            collect_translations=False,
        )

        # ---- Test loss ----
        print("  Test loss ...")
        test_coconut_loss, test_trans_loss = evaluate_loss(
            model, raw_test, **eval_args
        )

        # ---- Test accuracy + translations ----
        print("  Test accuracy + translations ...")
        test_acc, test_table = evaluate_accuracy(
            model, raw_test, test_answers, **eval_args,
            collect_translations=not debug,
        )

        print(
            f"[Epoch {epoch}]  "
            f"val_acc={val_acc*100:.2f}%  test_acc={test_acc*100:.2f}%  |  "
            f"val_coconut={val_coconut_loss:.4f}  val_trans={val_trans_loss:.4f}  |  "
            f"test_coconut={test_coconut_loss:.4f}  test_trans={test_trans_loss:.4f}"
        )

        if not debug:
            log_dict = {
                "eval/val_answer_accuracy":    val_acc,
                "eval/val_coconut_loss":       val_coconut_loss,
                "eval/val_translator_loss":    val_trans_loss,
                "eval/test_answer_accuracy":   test_acc,
                "eval/test_coconut_loss":      test_coconut_loss,
                "eval/test_translator_loss":   test_trans_loss,
                "meta/epoch":                  epoch,
            }
            if test_table is not None:
                log_dict["eval/test_translations"] = test_table
            wandb.log(log_dict)

        # ---- Best checkpoint (based on val accuracy) ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_checkpoint(model, cfg["save_path"], cfg["name"], epoch, val_acc)
            print(f"  → New best val {val_acc * 100:.2f}% — checkpoint saved.")

    if not debug:
        wandb.finish()

    print(f"\nTraining complete. Best val accuracy: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/haoyang/haoyang/coconut/args/direct_latent.yaml",
    )
    args = parser.parse_args()
    train(args.config)
