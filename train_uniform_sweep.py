"""
train_uniform_sweep.py

在 ProsQA 上以不同 uniform_prob 值扫描训练 CoconutWithTranslator 模型。
训练逻辑与 train.py 完全一致，额外增加：
  - test 集每 epoch 评估
  - 只保存 test accuracy 最高的 checkpoint
  - 最后一个 stage 的 early stopping（至少跑 min_epochs 后才触发）

uniform_prob 取值：0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

用法：
    python train_uniform_sweep.py
    python train_uniform_sweep.py --probs 0.3 0.5
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, GPT2LMHeadModel

from mixed import CoconutWithTranslator
from mixed_dataset import get_cot_latent_dataset, MyCollator, get_question_latent_dataset
from dataset import get_dataset
from translator_v3 import CoconutTranslator

GLOBAL_SEED = 42

UNIFORM_PROBS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 与 mixed_coconut.yaml 对齐
BASE_CFG = {
    "project":                     "coconut-translator-uniform-sweep",
    "model_id":                    "openai-community/gpt2",
    "train_path":                  "data/prosqa_train.json",
    "val_path":                    "data/prosqa_valid.json",
    "test_path":                   "data/prosqa_test.json",
    "load_model_path":             "/home/haoyang/haoyang/coconut/models/nihaoyang2002-kth-royal-institute-of-technology/checkpoint_5",
    "load_translator_path":        None,
    "save_base_path":              "./models/uniform_sweep",

    "max_latent_stage":            6,
    "epochs_per_stage":            5,
    "epochs_for_final_stage":      20,
    "c_thought":                   1,
    "pad_latent_to_max":           True,
    "no_cot":                      False,

    "lambda_translator":           0.5,
    "translator_lr":               5e-4,
    "warmup_steps_per_stage":      100,

    "lr":                          1e-4,
    "weight_decay":                0.01,
    "batch_size_training":         8,
    "gradient_accumulation_steps": 4,

    "num_eval_samples":            500,

    # Early stopping（仅最后一个 stage 生效）
    "early_stopping_patience":     8,
    "early_stopping_min_epochs":   5,
}


def save_checkpoint(model, optimizer, stage, epoch, path, name):
    os.makedirs(path, exist_ok=True)
    save_dict = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage":                stage,
        "epoch":                epoch,
    }
    torch.save(save_dict, os.path.join(path, f"{name}_best.pt"))
    print(f"  [Checkpoint] Saved best → {os.path.join(path, f'{name}_best.pt')}")


@torch.no_grad()
def evaluate_and_log(model, raw_val, raw_test, tokenizer, stage, epoch,
                     device, cfg, latent_id, start_id, end_id):
    """val 和 test 各跑一遍，返回 (val_acc, test_acc)。逻辑与 train.py 完全一致。"""
    model.eval()

    def run_eval(raw_ds, tag):
        num_eval = min(cfg.get("num_eval_samples", 300), len(raw_ds))
        eval_raw = raw_ds.select(range(num_eval))

        # --- Loss ---
        eval_loss_ds = get_cot_latent_dataset(
            scheduled_stage=stage,
            base_dataset=eval_raw,
            configs=type('obj', (object,), {**cfg, 'uniform_prob': 0.0}),
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
            shuffle=False,
            eos_id=tokenizer.eos_token_id,
        )
        collator    = MyCollator(tokenizer=tokenizer, latent_id=latent_id)
        loss_loader = DataLoader(eval_loss_ds, batch_size=cfg["batch_size_training"],
                                 collate_fn=collator)

        total_loss, total_coco, total_trans = 0.0, 0.0, 0.0
        for batch in tqdm(loss_loader, desc=f"Stage {stage} {tag} Loss"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(**batch)
            total_loss  += outputs.loss.item()
            total_coco  += outputs.coconut_loss.item()
            total_trans += outputs.translator_loss.item()
        n = len(loss_loader)
        avg_loss, avg_coco, avg_trans = total_loss / n, total_coco / n, total_trans / n
        print(f"  {tag} Loss: {avg_loss:.4f} (Coconut: {avg_coco:.4f}, Translator: {avg_trans:.4f})")

        # --- Generation Accuracy ---
        eval_gen_ds = get_question_latent_dataset(
            scheduled_stage=stage,
            base_dataset=eval_raw,
            configs=type('obj', (object,), cfg),
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
        )

        table = wandb.Table(columns=[
            "Stage", "Epoch", "Question",
            "GT_Thoughts", "Decoded_Thoughts",
            "GT_Answer", "Generated_Answer", "Answer_Match"
        ])

        correct_answers, correct_thoughts, total_thoughts = 0, 0, 0

        for i in tqdm(range(num_eval), desc=f"Eval Gen Stage {stage} {tag}"):
            sample_gen = eval_gen_ds[i]
            raw_sample  = eval_raw[i]

            gt_answer   = str(raw_sample["answer"]).strip()
            gt_steps    = raw_sample["steps"]
            gt_thoughts = {idx: step.strip() for idx, step in enumerate(gt_steps[:stage])}

            input_ids = torch.tensor(sample_gen["input_ids"]).unsqueeze(0).to(device)

            first_latent_pos = (input_ids[0] == latent_id).nonzero()
            first_pos = first_latent_pos[0, 0].item() if len(first_latent_pos) > 0 else input_ids.shape[1]
            context_ids = input_ids[:, :first_pos]

            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids,
                position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0).to(device),
            )
            decoded_thoughts_list = model.translate_latents(
                outputs.latent_states, context_ids, tokenizer,
                c_thought=cfg.get("c_thought", 1)
            )
            decoded_thoughts = {idx: t.strip() for idx, t in enumerate(decoded_thoughts_list)}

            gen_ids     = model.generate(input_ids, tokenizer, max_new_tokens=150, show_thoughts=False)
            answer      = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            pure_answer = answer.split("#")[-1].replace(",", "").strip()
            question    = tokenizer.decode(context_ids[0], skip_special_tokens=True)

            answer_is_correct = (gt_answer == pure_answer)
            if answer_is_correct:
                correct_answers += 1

            for idx, gt_step in gt_thoughts.items():
                clean_gt  = gt_step.replace(" ", "").replace("\n", "").lower()
                clean_dec = decoded_thoughts.get(idx, "").replace(" ", "").replace("\n", "").lower()
                if clean_gt and clean_gt == clean_dec:
                    correct_thoughts += 1
                elif not clean_gt and not clean_dec:
                    correct_thoughts += 1
                total_thoughts += 1

            if i < 5:
                gt_str  = "\n".join([f"Thought {k+1}: {v}" for k, v in gt_thoughts.items()])
                dec_str = "\n".join([f"Thought {k+1}: {v}" for k, v in decoded_thoughts.items()])
                table.add_data(stage, epoch, question, gt_str, dec_str,
                               gt_answer, answer, answer_is_correct)

        ans_acc     = correct_answers / num_eval
        thought_acc = correct_thoughts / total_thoughts if total_thoughts > 0 else 0.0
        print(f"  [{tag}] Answer Acc: {ans_acc*100:.2f}%  Thought Acc: {thought_acc*100:.2f}%")
        return avg_loss, avg_coco, avg_trans, ans_acc, thought_acc, table

    val_loss,  val_coco,  val_trans,  val_acc,  val_thought,  val_table  = run_eval(raw_val,  "Val")
    test_loss, test_coco, test_trans, test_acc, test_thought, test_table = run_eval(raw_test, "Test")

    wandb.log({
        "eval/val_loss":              val_loss,
        "eval/val_coconut_loss":      val_coco,
        "eval/val_translator_loss":   val_trans,
        "eval/val_samples_table":     val_table,
        "eval/val_answer_accuracy":   val_acc,
        "eval/val_thought_accuracy":  val_thought,
        "eval/test_loss":             test_loss,
        "eval/test_coconut_loss":     test_coco,
        "eval/test_translator_loss":  test_trans,
        "eval/test_samples_table":    test_table,
        "eval/test_answer_accuracy":  test_acc,
        "eval/test_thought_accuracy": test_thought,
        "meta/stage":                 stage,
        "meta/epoch":                 epoch,
    })

    model.train()
    return val_acc, test_acc


def train_one_prob(uniform_prob, cfg, device):
    run_name = f"uniform_prob_{uniform_prob:.1f}".replace(".", "p")
    save_dir  = os.path.join(cfg["save_base_path"], run_name)

    print(f"\n{'='*60}")
    print(f"Training: uniform_prob = {uniform_prob}  (run: {run_name})")
    print(f"{'='*60}")

    # 固定随机种子（保证单一变量）
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # ---- Tokenizer（与 train.py 完全一致）----
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = ["<latent>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    pad_id    = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    run_cfg = {**cfg, "uniform_prob": uniform_prob, "name": run_name}

    # ---- Base Model（与 train.py 完全一致）----
    base_model = GPT2LMHeadModel.from_pretrained(cfg["model_id"])
    vocab_size  = len(tokenizer)

    if cfg.get("load_model_path"):
        coconut_ckpt    = torch.load(cfg["load_model_path"], map_location="cpu")
        state_dict      = coconut_ckpt.get("model_state_dict", coconut_ckpt)
        new_state_dict  = {k.replace("base_causallm.", ""): v for k, v in state_dict.items()}
        ckpt_vocab_size = new_state_dict["transformer.wte.weight"].size(0)
        base_model.resize_token_embeddings(ckpt_vocab_size)
        final_state_dict = {k: v for k, v in new_state_dict.items()
                            if k != "embedding.weight" and not k.startswith("embedding.")}
        base_model.load_state_dict(final_state_dict, strict=False)
        actual_vocab_size = len(tokenizer)
        if ckpt_vocab_size != actual_vocab_size:
            base_model.resize_token_embeddings(actual_vocab_size)
        vocab_size = actual_vocab_size
    else:
        base_model.resize_token_embeddings(vocab_size)

    # ---- Translator（与 train.py 完全一致）----
    translator = CoconutTranslator(
        hidden_size=base_model.config.n_embd,
        vocab_size=vocab_size,
        start_id=start_id,
        end_id=end_id,
        pad_id=pad_id,
        eos_id=tokenizer.eos_token_id,
        mode="context_latent",
    )

    if cfg.get("load_translator_path"):
        t_ckpt  = torch.load(cfg["load_translator_path"], map_location="cpu")
        t_state = t_ckpt.get("model_state_dict", t_ckpt)
        t_final = {k.replace("decoder.", ""): v for k, v in t_state.items()
                   if k != "embedding.weight" and not k.startswith("embedding.")}
        translator.decoder.resize_token_embeddings(vocab_size)
        translator.decoder.load_state_dict(t_final, strict=False)

    # ---- 组装模型（与 train.py 完全一致：只 .to(device)，不转 bf16）----
    model = CoconutWithTranslator(
        base_causallm=base_model,
        translator=translator,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        lambda_translator=cfg.get("lambda_translator", 0.5),
        c_thought=cfg.get("c_thought", 1),
    ).to(device)

    # ---- 数据集 ----
    raw_train = get_dataset(cfg["train_path"], tokenizer)
    raw_val   = get_dataset(cfg["val_path"],   tokenizer)
    raw_test  = get_dataset(cfg["test_path"],  tokenizer)

    # ---- 优化器（与 train.py 完全一致：单一参数组）----
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    wandb.init(
        project=cfg["project"],
        name=run_name,
        config=run_cfg,
        reinit=True,
    )

    global_step   = 0
    best_test_acc = -1.0
    patience      = cfg.get("early_stopping_patience", 8)
    min_epochs    = cfg.get("early_stopping_min_epochs", 5)
    warmup_steps  = cfg.get("warmup_steps_per_stage", 100)

    for stage in range(1, cfg["max_latent_stage"] + 1):
        print(f"\n>>> Stage {stage}  (uniform_prob={uniform_prob})")

        # 动态 lambda_translator（与 train.py 完全一致）
        current_lambda = cfg.get("lambda_translator", 0.5) * (stage / cfg["max_latent_stage"])
        model.lambda_translator = current_lambda
        print(f"  lambda_translator = {current_lambda:.3f}")

        # Stage 开始时重置 LR + warmup scheduler（与 train.py 完全一致）
        for pg in optimizer.param_groups:
            pg["lr"] = cfg["lr"]
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)),
        )

        target_epochs = (cfg["epochs_for_final_stage"]
                         if stage == cfg["max_latent_stage"]
                         else cfg["epochs_per_stage"])

        train_ds = get_cot_latent_dataset(
            scheduled_stage=stage,
            base_dataset=raw_train,
            configs=type('obj', (object,), run_cfg),
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
            shuffle=True,
            eos_id=tokenizer.eos_token_id,
        )
        collator     = MyCollator(tokenizer=tokenizer, latent_id=latent_id)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size_training"],
            shuffle=True,
            collate_fn=collator,
        )

        is_final_stage   = (stage == cfg["max_latent_stage"])
        stage_best_test  = -1.0
        stage_no_improve = 0

        for epoch in range(target_epochs):
            # ---- 训练（与 train.py 完全一致）----
            model.train()
            pbar = tqdm(train_loader,
                        desc=f"[prob={uniform_prob}] Stage {stage} Epoch {epoch}/{target_epochs-1}")

            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss / cfg["gradient_accumulation_steps"]
                loss.backward()

                if (global_step + 1) % cfg["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if global_step % 10 == 0:
                    wandb.log({
                        "train/total_loss":       outputs.loss.item(),
                        "train/coconut_loss":     outputs.coconut_loss.item(),
                        "train/translator_loss":  outputs.translator_loss.item(),
                        "meta/stage":             stage,
                        "meta/epoch":             epoch,
                        "meta/step":              global_step,
                        "meta/lr":                optimizer.param_groups[0]["lr"],
                        "meta/lambda_translator": model.lambda_translator,
                    })

                pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})
                global_step += 1

            # ---- 评估（val + test）----
            print(f"\nRunning Evaluation for Stage {stage}, Epoch {epoch}...")
            val_acc, test_acc = evaluate_and_log(
                model, raw_val, raw_test, tokenizer,
                stage, epoch, device, run_cfg,
                latent_id, start_id, end_id,
            )

            # ---- 只保存 test acc 最高的 checkpoint ----
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_checkpoint(model, optimizer, stage, epoch, save_dir, run_name)
                wandb.log({"eval/best_test_accuracy": best_test_acc,
                           "meta/stage": stage, "meta/epoch": epoch})
                print(f"  *** New best test acc: {best_test_acc*100:.2f}% ***")

            # ---- Early stopping（仅最后 stage，至少跑 min_epochs）----
            if test_acc > stage_best_test:
                stage_best_test  = test_acc
                stage_no_improve = 0
            else:
                stage_no_improve += 1

            wandb.log({
                "early_stopping/stage_no_improve": stage_no_improve,
                "early_stopping/stage_best_test":  stage_best_test,
                "meta/stage": stage, "meta/epoch": epoch,
            })

            should_stop = (is_final_stage
                           and stage_no_improve >= patience
                           and epoch + 1 >= min_epochs)
            if should_stop:
                print(f"  [Early Stop] Stage {stage} stopped at epoch {epoch} "
                      f"({stage_no_improve} epochs no improvement, "
                      f"best={stage_best_test*100:.2f}%)")
                wandb.log({"early_stopping/triggered_stage": stage,
                           "early_stopping/triggered_epoch": epoch,
                           "meta/stage": stage, "meta/epoch": epoch})
                break

    wandb.finish()
    print(f"\n[Done] uniform_prob={uniform_prob}  best_test_acc={best_test_acc*100:.2f}%")
    return best_test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs", nargs="+", type=float, default=UNIFORM_PROBS)
    parser.add_argument("--patience",   type=int, default=None)
    parser.add_argument("--min_epochs", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"uniform_prob sweep: {args.probs}")

    cfg = BASE_CFG.copy()
    if args.patience   is not None:
        cfg["early_stopping_patience"]  = args.patience
    if args.min_epochs is not None:
        cfg["early_stopping_min_epochs"] = args.min_epochs

    summary = {}
    for prob in args.probs:
        best_acc = train_one_prob(prob, cfg, device)
        summary[prob] = best_acc

    print("\n" + "="*50)
    print("Sweep Summary (best test accuracy):")
    for prob, acc in sorted(summary.items()):
        print(f"  uniform_prob={prob:.1f}  →  {acc*100:.2f}%")
    print("="*50)

    os.makedirs(cfg["save_base_path"], exist_ok=True)
    out_path = os.path.join(cfg["save_base_path"], "sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
