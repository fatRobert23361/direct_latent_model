"""
train_ca.py — 使用 Cross-Attention 翻译器（translator_v4）的联合训练脚本。

与 train.py 的唯一模型差异：
  translator_v3.CoconutTranslator   →  translator_v4.CoconutTranslatorCA
  latent 不再拼入 decoder 序列，而是通过专用 cross-attention 路径注入。

其余逻辑（课程学习、KV-cache、分布式、WandB、评估）与 train.py 完全一致。
配置文件：args/mixed_coconut_ca.yaml
"""

import os
import yaml
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
from translator_v4 import CoconutTranslatorCA   # ← 唯一改动


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, stage, epoch, path, name):
    os.makedirs(path, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stage": stage,
            "epoch": epoch,
        },
        os.path.join(path, f"{name}_stage{stage}_epoch{epoch}.pt"),
    )


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train():
    # 1. 加载配置
    config_path = os.path.join(
        os.path.dirname(__file__), "args", "mixed_coconut_ca.yaml"
    )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = ["<latent>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    pad_id    = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 3. 初始化 Base LLM
    print(f"Initializing Base Model: {cfg['model_id']}")
    base_model = GPT2LMHeadModel.from_pretrained(cfg["model_id"])
    vocab_size = len(tokenizer)

    # 4. 加载纯 Coconut 权重
    if cfg.get("load_model_path"):
        print(f"Loading Coconut weights from {cfg['load_model_path']}")
        coconut_ckpt = torch.load(cfg["load_model_path"], map_location="cpu")
        state_dict = coconut_ckpt.get("model_state_dict", coconut_ckpt)

        new_state_dict = {k.replace("base_causallm.", ""): v for k, v in state_dict.items()}
        ckpt_vocab_size = new_state_dict["transformer.wte.weight"].size(0)
        print(f"Detected vocab size from checkpoint: {ckpt_vocab_size}")

        base_model.resize_token_embeddings(ckpt_vocab_size)

        final_state_dict = {
            k: v for k, v in new_state_dict.items()
            if k != "embedding.weight" and not k.startswith("embedding.")
        }
        base_model.load_state_dict(final_state_dict, strict=False)
        print("Successfully loaded Coconut base weights.")

        actual_vocab_size = len(tokenizer)
        if ckpt_vocab_size != actual_vocab_size:
            print(f"Expanding model vocab from {ckpt_vocab_size} to {actual_vocab_size}.")
            base_model.resize_token_embeddings(actual_vocab_size)

        vocab_size = actual_vocab_size
    else:
        base_model.resize_token_embeddings(vocab_size)

    # 5. 初始化 Cross-Attention 翻译器
    print(f"Initializing CoconutTranslatorCA with vocab size: {vocab_size}")
    translator = CoconutTranslatorCA(
        hidden_size=base_model.config.n_embd,
        vocab_size=vocab_size,
        start_id=start_id,
        end_id=end_id,
        pad_id=pad_id,
        eos_id=tokenizer.eos_token_id,
    )

    # 6. 加载预训练翻译器权重（可选）
    if cfg.get("load_translator_path"):
        print(f"Loading Translator weights from {cfg['load_translator_path']}")
        t_ckpt = torch.load(cfg["load_translator_path"], map_location="cpu")
        t_state_dict = t_ckpt.get("model_state_dict", t_ckpt)
        t_final_dict = {
            k.replace("decoder.", ""): v for k, v in t_state_dict.items()
            if k != "embedding.weight" and not k.startswith("embedding.")
        }
        translator.decoder.resize_token_embeddings(vocab_size)
        translator.decoder.load_state_dict(t_final_dict, strict=False)

    # 7. 组装混合模型（CoconutWithTranslator 接口与 translator 版本无关）
    model = CoconutWithTranslator(
        base_causallm=base_model,
        translator=translator,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        lambda_translator=cfg.get("lambda_translator", 0.5),
        c_thought=cfg.get("c_thought", 1),
    ).to(device).to(torch.bfloat16)

    # 8. 恢复 checkpoint（仅加载一次）
    resume_ckpt = None
    if cfg.get("resume_from_checkpoint"):
        print(f"Resuming from checkpoint: {cfg['resume_from_checkpoint']}")
        resume_ckpt = torch.load(cfg["resume_from_checkpoint"], map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        print("Resumed model weights.")

    # 9. 数据集
    print("Loading datasets...")
    raw_train = get_dataset(cfg["train_path"], tokenizer)
    raw_val   = get_dataset(cfg["val_path"], tokenizer)

    # 10. 优化器：主干与翻译器分组，翻译器用更高 lr
    translator_lr = cfg.get("translator_lr", cfg["lr"] * 5)
    optimizer = AdamW(
        [
            {"params": list(model.base_causallm.parameters()), "lr": cfg["lr"]},
            {"params": list(model.translator.parameters()), "lr": translator_lr, "name": "translator"},
        ],
        weight_decay=cfg["weight_decay"],
    )

    if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        print("Resumed optimizer state.")

    # 11. WandB
    if not cfg.get("debug", False):
        wandb.init(project=cfg["project"], name=cfg["name"], config=cfg)

    global_step = 0
    start_stage = cfg.get("resume_stage", 1)
    start_epoch = cfg.get("resume_epoch", 0)
    warmup_steps = cfg.get("warmup_steps_per_stage", 100)

    # 12. 课程学习主循环
    for stage in range(start_stage, cfg["max_latent_stage"] + 1):
        print(f"\n>>> Stage {stage}")

        current_lambda = cfg.get("lambda_translator", 0.5) * (stage / cfg["max_latent_stage"])
        model.lambda_translator = current_lambda
        print(f"lambda_translator = {current_lambda:.3f}")

        # stage 切换时重置翻译器 Adam 动量（latent 分布突变，旧动量方向会误导更新）
        translator_param_set = set(model.translator.parameters())
        for p in list(translator_param_set):
            if p in optimizer.state:
                del optimizer.state[p]
        for pg in optimizer.param_groups:
            if pg.get("name") == "translator":
                pg["lr"] = translator_lr
            else:
                pg["lr"] = cfg["lr"]

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)),
        )

        if stage == cfg["max_latent_stage"] and cfg.get("epochs_for_final_stage"):
            target_epochs = cfg["epochs_for_final_stage"]
        else:
            target_epochs = cfg["epochs_per_stage"]

        train_ds = get_cot_latent_dataset(
            scheduled_stage=stage,
            base_dataset=raw_train,
            configs=type("obj", (object,), cfg),
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
            shuffle=True,
            eos_id=tokenizer.eos_token_id,
        )
        collator = MyCollator(tokenizer=tokenizer, latent_id=latent_id)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size_training"],
            shuffle=True,
            collate_fn=collator,
        )

        first_epoch = start_epoch if stage == start_stage else 0
        for epoch in range(first_epoch, target_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Stage {stage} | Epoch {epoch}/{target_epochs-1}")

            for batch in pbar:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)

                loss = outputs.loss / cfg["gradient_accumulation_steps"]
                loss.backward()

                if (global_step + 1) % cfg["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if global_step % 10 == 0 and not cfg.get("debug", False):
                    wandb.log(
                        {
                            "train/total_loss":       outputs.loss.item(),
                            "train/coconut_loss":     outputs.coconut_loss.item(),
                            "train/translator_loss":  outputs.translator_loss.item(),
                            "meta/stage":             stage,
                            "meta/epoch":             epoch,
                            "meta/step":              global_step,
                            "meta/lr":                optimizer.param_groups[0]["lr"],
                            "meta/lambda_translator": model.lambda_translator,
                        }
                    )

                pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})
                global_step += 1

            print(f"\nEvaluating Stage {stage}, Epoch {epoch}...")
            evaluate_and_log_wandb(
                model, raw_val, tokenizer, stage, epoch, device, cfg,
                latent_id, start_id, end_id,
            )
            if not cfg.get("debug", False):
                save_checkpoint(model, optimizer, stage, epoch, cfg["save_path"], cfg["name"])

    if not cfg.get("debug", False):
        wandb.finish()


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_and_log_wandb(model, raw_val, tokenizer, stage, epoch, device, cfg, latent_id, start_id, end_id):
    model.eval()

    num_eval = min(cfg.get("num_eval_samples", 300), len(raw_val))
    eval_raw = raw_val.select(range(num_eval))

    # --- Validation Loss ---
    eval_loss_ds = get_cot_latent_dataset(
        scheduled_stage=stage,
        base_dataset=eval_raw,
        configs=type("obj", (object,), {**cfg, "uniform_prob": 0.0}),
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        shuffle=False,
        eos_id=tokenizer.eos_token_id,
    )
    collator = MyCollator(tokenizer=tokenizer, latent_id=latent_id)
    val_loader = DataLoader(eval_loss_ds, batch_size=cfg["batch_size_training"], collate_fn=collator)

    total_val_loss, total_val_coconut, total_val_trans = 0.0, 0.0, 0.0
    for batch in tqdm(val_loader, desc=f"Stage {stage} Val Loss"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
        total_val_loss    += outputs.loss.item()
        total_val_coconut += outputs.coconut_loss.item()
        total_val_trans   += outputs.translator_loss.item()

    n = len(val_loader)
    avg_val_loss    = total_val_loss    / n
    avg_val_coconut = total_val_coconut / n
    avg_val_trans   = total_val_trans   / n
    print(f"Val Loss: {avg_val_loss:.4f} (Coconut: {avg_val_coconut:.4f}, Translator: {avg_val_trans:.4f})")

    # --- Generation Accuracy ---
    eval_gen_ds = get_question_latent_dataset(
        scheduled_stage=stage,
        base_dataset=eval_raw,
        configs=type("obj", (object,), cfg),
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
    )

    table = wandb.Table(columns=[
        "Stage", "Epoch", "Question",
        "GT_Thoughts", "Decoded_Thoughts",
        "GT_Answer", "Generated_Answer", "Answer_Match",
    ]) if not cfg.get("debug", False) else None

    correct_answers, correct_thoughts, total_thoughts = 0, 0, 0

    print(f"\nRunning Generation Eval ({num_eval} samples)...")
    for i in tqdm(range(num_eval), desc=f"Gen Eval Stage {stage}"):
        sample_gen = eval_gen_ds[i]
        raw_sample = eval_raw[i]

        gt_answer = str(raw_sample["answer"]).strip()
        gt_steps  = raw_sample["steps"]
        gt_thoughts_text = {idx: step.strip() for idx, step in enumerate(gt_steps[:stage])}

        input_ids = torch.tensor(sample_gen["input_ids"]).unsqueeze(0).to(device)

        first_latent_pos = (input_ids[0] == latent_id).nonzero()
        first_pos   = first_latent_pos[0, 0].item() if len(first_latent_pos) > 0 else input_ids.shape[1]
        context_ids = input_ids[:, :first_pos]

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids,
                position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0).to(device),
            )
            decoded_thoughts_list = model.translate_latents(
                outputs.latent_states, context_ids, tokenizer,
                c_thought=cfg.get("c_thought", 1),
            )
            gen_ids = model.generate(input_ids, tokenizer, max_new_tokens=150, show_thoughts=False)

        decoded_thoughts_text = {idx: text.strip() for idx, text in enumerate(decoded_thoughts_list)}
        answer      = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        pure_answer = answer.split("#")[-1].replace(",", "").strip()
        question    = tokenizer.decode(context_ids[0], skip_special_tokens=True)

        answer_is_correct = gt_answer == pure_answer
        if answer_is_correct:
            correct_answers += 1

        for idx, gt_step in gt_thoughts_text.items():
            clean_gt      = gt_step.replace(" ", "").replace("\n", "").lower()
            decoded_step  = decoded_thoughts_text.get(idx, "")
            clean_decoded = decoded_step.replace(" ", "").replace("\n", "").lower()
            if clean_gt == clean_decoded:
                correct_thoughts += 1
            total_thoughts += 1

        if table is not None and i < 5:
            gt_str      = "\n".join([f"Thought {k+1}: {v}" for k, v in gt_thoughts_text.items()])
            decoded_str = "\n".join([f"Thought {k+1}: {v}" for k, v in decoded_thoughts_text.items()])
            table.add_data(stage, epoch, question, gt_str, decoded_str, gt_answer, answer, answer_is_correct)

    ans_acc    = correct_answers / num_eval
    thought_acc = correct_thoughts / total_thoughts if total_thoughts > 0 else 0.0
    print(f"[Stage {stage} Epoch {epoch}] Answer Acc: {ans_acc*100:.2f}% | Thought Acc: {thought_acc*100:.2f}%")

    if not cfg.get("debug", False):
        log_dict = {
            "eval/loss":             avg_val_loss,
            "eval/coconut_loss":     avg_val_coconut,
            "eval/translator_loss":  avg_val_trans,
            "eval/answer_accuracy":  ans_acc,
            "eval/thought_accuracy": thought_acc,
            "meta/stage":            stage,
            "meta/epoch":            epoch,
        }
        if table is not None:
            log_dict["eval/samples_table"] = table
        wandb.log(log_dict)

    model.train()


if __name__ == "__main__":
    train()
