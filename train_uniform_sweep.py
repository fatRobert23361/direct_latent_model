"""
train_uniform_sweep.py

在 ProsQA 上以不同 uniform_prob 值扫描训练 CoconutWithTranslator 模型。
每个 uniform_prob 启动一个独立的 wandb run，每个 epoch 在 val 和 test
两个集合上评估，只保存 test accuracy 最高的 checkpoint。

uniform_prob 取值：0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

用法：
    python train_uniform_sweep.py
    python train_uniform_sweep.py --probs 0.3 0.5      # 只跑部分值
    python train_uniform_sweep.py --start_stage 3      # 从指定 stage 开始（接续训练）
"""

import argparse
import json
import os
import copy

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

# -----------------------------------------------------------------------
# 固定超参（从 mixed_coconut.yaml 中提取，不参与 sweep）
# -----------------------------------------------------------------------
BASE_CFG = {
    "project":                    "coconut-translator-uniform-sweep",
    "model_id":                   "openai-community/gpt2",
    "train_path":                 "data/prosqa_train.json",
    "val_path":                   "data/prosqa_valid.json",
    "test_path":                  "data/prosqa_test.json",
    "load_model_path":            "/home/haoyang/haoyang/coconut/models/nihaoyang2002-kth-royal-institute-of-technology/checkpoint_5",
    "load_translator_path":       None,
    "save_base_path":             "./models/uniform_sweep",

    # 训练阶段
    "max_latent_stage":           6,
    "epochs_per_stage":           5,
    "epochs_for_final_stage":     20,   # 最后 stage 多训
    "c_thought":                  1,
    "pad_latent_to_max":          True,
    "no_cot":                     False,

    # 翻译器
    "lambda_translator":          0.5,
    "translator_lr":              5e-4,
    "warmup_steps_per_stage":     100,

    # 优化器
    "lr":                         1e-4,
    "weight_decay":               0.01,
    "batch_size_training":        8,
    "gradient_accumulation_steps": 4,

    # 评估
    "num_eval_samples":           500,   # val/test 各评估多少条

    # Early stopping（仅在最后一个 stage 生效）
    # 监控指标：test accuracy（与 checkpoint 保存标准一致）
    "early_stopping_patience":    4,     # 连续多少个 epoch test acc 不改善则停止训练
    "early_stopping_min_epochs":  5,     # 最后 stage 至少跑多少 epoch 才允许触发
}

UNIFORM_PROBS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# -----------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------

def make_cfg_obj(d):
    """把 dict 包成有 getattr 和 get 的对象，兼容 dataset 函数的 configs 参数。"""
    class Cfg:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
        def get(self, k, default=None):
            return getattr(self, k, default)
    return Cfg(d)


def build_tokenizer_and_ids(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    special_tokens = ["<latent>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    return tokenizer, latent_id, start_id, end_id


def build_model(cfg, tokenizer, latent_id, start_id, end_id, device):
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id

    base_model = GPT2LMHeadModel.from_pretrained(cfg["model_id"])

    if cfg.get("load_model_path"):
        ckpt = torch.load(cfg["load_model_path"], map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        clean = {k.replace("base_causallm.", ""): v for k, v in state_dict.items()}
        ckpt_vocab = clean["transformer.wte.weight"].size(0)
        base_model.resize_token_embeddings(ckpt_vocab)
        clean = {k: v for k, v in clean.items()
                 if not k.startswith("embedding.")}
        base_model.load_state_dict(clean, strict=False)
        if ckpt_vocab != vocab_size:
            base_model.resize_token_embeddings(vocab_size)
    else:
        base_model.resize_token_embeddings(vocab_size)

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
        t_ckpt = torch.load(cfg["load_translator_path"], map_location="cpu")
        t_sd = t_ckpt.get("model_state_dict", t_ckpt)
        t_clean = {k.replace("decoder.", ""): v for k, v in t_sd.items()
                   if not k.startswith("embedding.")}
        translator.decoder.resize_token_embeddings(vocab_size)
        translator.decoder.load_state_dict(t_clean, strict=False)

    model = CoconutWithTranslator(
        base_causallm=base_model,
        translator=translator,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        lambda_translator=cfg.get("lambda_translator", 0.5),
        c_thought=cfg.get("c_thought", 1),
    )
    model.to(device).to(torch.bfloat16)
    return model


def build_optimizer(model, cfg, translator_lr):
    return AdamW([
        {"params": list(model.base_causallm.parameters()), "lr": cfg["lr"]},
        {"params": list(model.translator.parameters()), "lr": translator_lr, "name": "translator"},
    ], weight_decay=cfg["weight_decay"])


def reset_stage(model, optimizer, cfg, translator_lr, warmup_steps):
    """每个 stage 开始时：重置 translator Adam 状态，重置 LR，建新 scheduler。"""
    translator_params = set(model.translator.parameters())
    for p in translator_params:
        if p in optimizer.state:
            del optimizer.state[p]
    for pg in optimizer.param_groups:
        if pg.get("name") == "translator":
            pg["lr"] = translator_lr
        else:
            pg["lr"] = cfg["lr"]
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps))
    )
    return scheduler


@torch.no_grad()
def compute_accuracy(model, tokenizer, raw_dataset, cfg_obj, latent_id, start_id, end_id,
                     stage, device, num_samples, answers_gt):
    """
    在 raw_dataset 上运行生成评估，返回 (accuracy, thought_accuracy)。
    raw_dataset: HuggingFace Dataset，已含 tokenized 字段。
    answers_gt:  List[str]，按 idx 对应的 ground-truth 答案（已 strip & 去逗号）。
    """
    model.eval()
    num_samples = min(num_samples, len(raw_dataset))
    eval_ds = raw_dataset.select(range(num_samples))

    gen_ds = get_question_latent_dataset(
        scheduled_stage=stage,
        base_dataset=eval_ds,
        configs=cfg_obj,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
    )

    correct_answers, correct_thoughts, total_thoughts = 0, 0, 0
    c_thought = cfg_obj.get("c_thought", 1)

    for i in range(num_samples):
        sample_gen = gen_ds[i]
        raw_sample = eval_ds[i]
        gt_answer = str(raw_sample["answer"]).replace(",", "").strip()
        gt_steps  = raw_sample["steps"]
        gt_thoughts = {k: v.strip() for k, v in enumerate(gt_steps[:stage])}

        input_ids = torch.tensor(sample_gen["input_ids"]).unsqueeze(0).to(device)
        first_latent_pos = (input_ids[0] == latent_id).nonzero()
        first_pos = first_latent_pos[0, 0].item() if len(first_latent_pos) > 0 else input_ids.shape[1]
        context_ids = input_ids[:, :first_pos]

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids,
                position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0).to(device),
            )
            decoded_thoughts_list = model.translate_latents(
                outputs.latent_states, context_ids, tokenizer, c_thought=c_thought
            )
            gen_ids = model.generate(input_ids, tokenizer, max_new_tokens=150, show_thoughts=False)

        answer_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        pure_answer = answer_text.split("#")[-1].replace(",", "").strip()

        if pure_answer == gt_answer:
            correct_answers += 1

        decoded_map = {k: t.strip() for k, t in enumerate(decoded_thoughts_list)}
        for idx, gt_step in gt_thoughts.items():
            clean_gt = gt_step.replace(" ", "").replace("\n", "").lower()
            clean_dec = decoded_map.get(idx, "").replace(" ", "").replace("\n", "").lower()
            if clean_gt == clean_dec:
                correct_thoughts += 1
            total_thoughts += 1

    ans_acc     = correct_answers / num_samples
    thought_acc = correct_thoughts / total_thoughts if total_thoughts > 0 else 0.0
    return ans_acc, thought_acc


@torch.no_grad()
def compute_val_loss(model, raw_val, cfg, cfg_obj, latent_id, start_id, end_id,
                     stage, device, tokenizer):
    """计算 validation loss（使用完整 CoT+latent 序列，禁用 uniform_prob）。"""
    num_eval = min(cfg["num_eval_samples"], len(raw_val))
    eval_raw = raw_val.select(range(num_eval))

    # uniform_prob=0 保证 eval loss 可复现
    eval_cfg = make_cfg_obj({**cfg, "uniform_prob": 0.0})
    eval_ds = get_cot_latent_dataset(
        scheduled_stage=stage,
        base_dataset=eval_raw,
        configs=eval_cfg,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        shuffle=False,
        eos_id=tokenizer.eos_token_id,
    )
    collator = MyCollator(tokenizer=tokenizer, latent_id=latent_id)
    loader   = DataLoader(eval_ds, batch_size=cfg["batch_size_training"], collate_fn=collator)

    total_loss, total_coco, total_trans = 0.0, 0.0, 0.0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**batch)
        total_loss  += out.loss.item()
        total_coco  += out.coconut_loss.item()
        total_trans += out.translator_loss.item()

    n = len(loader)
    return total_loss / n, total_coco / n, total_trans / n


def save_best_checkpoint(model, optimizer, stage, epoch, save_dir, run_name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{run_name}_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage": stage,
        "epoch": epoch,
    }, path)
    print(f"  [Checkpoint] Saved best checkpoint → {path}")


# -----------------------------------------------------------------------
# 单次 uniform_prob 训练
# -----------------------------------------------------------------------

def train_one_prob(uniform_prob, cfg, raw_train, raw_val, raw_test,
                   answers_val, answers_test, device, start_stage=1):

    run_name = f"uniform_prob_{uniform_prob:.1f}".replace(".", "p")
    save_dir = os.path.join(cfg["save_base_path"], run_name)

    print(f"\n{'='*60}")
    print(f"Training: uniform_prob = {uniform_prob}  (run: {run_name})")
    print(f"{'='*60}")

    # 每次从头构建 tokenizer / model
    tokenizer, latent_id, start_id, end_id = build_tokenizer_and_ids(cfg["model_id"])

    run_cfg = {**cfg, "uniform_prob": uniform_prob, "name": run_name}
    model = build_model(run_cfg, tokenizer, latent_id, start_id, end_id, device)

    translator_lr = cfg["translator_lr"]
    optimizer = build_optimizer(model, cfg, translator_lr)

    # 评估时的 cfg 对象（uniform_prob 对 eval 无影响，只影响训练集采样）
    eval_cfg_obj = make_cfg_obj({**run_cfg, "uniform_prob": 0.0})

    wandb.init(
        project=cfg["project"],
        name=run_name,
        config={**run_cfg, "uniform_prob": uniform_prob},
        reinit=True,
    )

    global_step = 0
    best_test_acc = -1.0
    num_eval = cfg["num_eval_samples"]
    patience     = cfg.get("early_stopping_patience", 3)
    min_epochs   = cfg.get("early_stopping_min_epochs", 2)

    for stage in range(start_stage, cfg["max_latent_stage"] + 1):
        print(f"\n>>> Stage {stage}  (uniform_prob={uniform_prob})")

        # 动态 lambda_translator
        current_lambda = cfg["lambda_translator"] * (stage / cfg["max_latent_stage"])
        model.lambda_translator = current_lambda

        scheduler = reset_stage(model, optimizer, cfg, translator_lr,
                                 cfg["warmup_steps_per_stage"])

        # early stopping 计数器（仅最后 stage 有效，进入新 stage 时重置）
        stage_best_test_acc = -1.0
        stage_no_improve    = 0
        is_final_stage      = (stage == cfg["max_latent_stage"])

        target_epochs = (cfg["epochs_for_final_stage"]
                         if stage == cfg["max_latent_stage"]
                         else cfg["epochs_per_stage"])

        train_cfg_obj = make_cfg_obj(run_cfg)
        train_ds = get_cot_latent_dataset(
            scheduled_stage=stage,
            base_dataset=raw_train,
            configs=train_cfg_obj,
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
            shuffle=True,
            eos_id=tokenizer.eos_token_id,
        )
        collator    = MyCollator(tokenizer=tokenizer, latent_id=latent_id)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size_training"],
            shuffle=True,
            collate_fn=collator,
        )

        for epoch in range(target_epochs):
            # ---------- 训练 ----------
            model.train()
            pbar = tqdm(train_loader,
                        desc=f"[prob={uniform_prob}] Stage {stage} Epoch {epoch}/{target_epochs-1}")
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                        "train/total_loss":        outputs.loss.item(),
                        "train/coconut_loss":      outputs.coconut_loss.item(),
                        "train/translator_loss":   outputs.translator_loss.item(),
                        "meta/stage":              stage,
                        "meta/epoch":              epoch,
                        "meta/step":               global_step,
                        "meta/lr":                 optimizer.param_groups[0]["lr"],
                        "meta/lambda_translator":  model.lambda_translator,
                    })
                pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})
                global_step += 1

            # ---------- 验证 loss ----------
            print(f"\n  [Eval] Stage {stage} Epoch {epoch} — computing val loss ...")
            val_loss, val_coco, val_trans = compute_val_loss(
                model, raw_val, cfg, eval_cfg_obj, latent_id, start_id, end_id,
                stage, device, tokenizer
            )
            print(f"  val_loss={val_loss:.4f}  (coconut={val_coco:.4f}  translator={val_trans:.4f})")

            # ---------- Val accuracy ----------
            print(f"  [Eval] Stage {stage} Epoch {epoch} — val accuracy ...")
            val_acc, val_thought_acc = compute_accuracy(
                model, tokenizer, raw_val, eval_cfg_obj,
                latent_id, start_id, end_id,
                stage, device, num_eval, answers_val
            )
            print(f"  val_acc={val_acc*100:.2f}%  thought_acc={val_thought_acc*100:.2f}%")

            # ---------- Test accuracy ----------
            print(f"  [Eval] Stage {stage} Epoch {epoch} — test accuracy ...")
            test_acc, test_thought_acc = compute_accuracy(
                model, tokenizer, raw_test, eval_cfg_obj,
                latent_id, start_id, end_id,
                stage, device, num_eval, answers_test
            )
            print(f"  test_acc={test_acc*100:.2f}%  test_thought_acc={test_thought_acc*100:.2f}%")

            # ---------- 上传 wandb ----------
            wandb.log({
                "eval/val_loss":              val_loss,
                "eval/val_coconut_loss":      val_coco,
                "eval/val_translator_loss":   val_trans,
                "eval/val_answer_accuracy":   val_acc,
                "eval/val_thought_accuracy":  val_thought_acc,
                "eval/test_answer_accuracy":  test_acc,
                "eval/test_thought_accuracy": test_thought_acc,
                "meta/stage":                 stage,
                "meta/epoch":                 epoch,
            })

            # ---------- 只保存全局 test 最优 checkpoint ----------
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_best_checkpoint(model, optimizer, stage, epoch, save_dir, run_name)
                wandb.log({"eval/best_test_accuracy": best_test_acc,
                           "meta/stage": stage, "meta/epoch": epoch})
                print(f"  *** New best test acc: {best_test_acc*100:.2f}% ***")

            # ---------- Per-stage early stopping ----------
            if test_acc > stage_best_test_acc:
                stage_best_test_acc = test_acc
                stage_no_improve    = 0
            else:
                stage_no_improve += 1

            # Early stopping 仅在最后 stage 触发
            should_stop = (is_final_stage
                           and stage_no_improve >= patience
                           and epoch + 1 >= min_epochs)
            wandb.log({
                "early_stopping/stage_no_improve": stage_no_improve,
                "early_stopping/stage_best_test":  stage_best_test_acc,
                "meta/stage": stage, "meta/epoch": epoch,
            })

            if should_stop:
                print(f"  [Early Stop] Final stage {stage} stopped at epoch {epoch} "
                      f"({stage_no_improve} epochs without improvement, "
                      f"best={stage_best_test_acc*100:.2f}%)")
                wandb.log({"early_stopping/triggered_stage": stage,
                           "early_stopping/triggered_epoch": epoch,
                           "meta/stage": stage, "meta/epoch": epoch})
                break   # 结束训练

    wandb.finish()
    print(f"\n[Done] uniform_prob={uniform_prob}  best_test_acc={best_test_acc*100:.2f}%")
    return best_test_acc


# -----------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probs", nargs="+", type=float, default=UNIFORM_PROBS,
                        help="要扫描的 uniform_prob 值列表")
    parser.add_argument("--start_stage", type=int, default=1,
                        help="从第几个 stage 开始训练（续训时用）")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience（覆盖 BASE_CFG 默认值）")
    parser.add_argument("--min_epochs", type=int, default=None,
                        help="每 stage 最少跑多少 epoch 才允许 early stop（覆盖 BASE_CFG）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"uniform_prob sweep: {args.probs}")

    cfg = BASE_CFG.copy()
    if args.patience is not None:
        cfg["early_stopping_patience"] = args.patience
    if args.min_epochs is not None:
        cfg["early_stopping_min_epochs"] = args.min_epochs

    # 预加载数据集（tokenizer 只用来 tokenize，不同 prob 共享同一份数据）
    print("\nLoading datasets ...")
    tokenizer, _, _, _ = build_tokenizer_and_ids(cfg["model_id"])
    raw_train = get_dataset(cfg["train_path"], tokenizer)
    raw_val   = get_dataset(cfg["val_path"],   tokenizer)
    raw_test  = get_dataset(cfg["test_path"],  tokenizer)

    # ground-truth 答案（按顺序，idx 对应 raw_* 中的位置）
    def load_answers(path):
        data = json.load(open(path))
        return [d["answer"].replace(",", "").strip() for d in data]

    answers_val  = load_answers(cfg["val_path"])
    answers_test = load_answers(cfg["test_path"])

    summary = {}
    for prob in args.probs:
        best_acc = train_one_prob(
            uniform_prob=prob,
            cfg=cfg,
            raw_train=raw_train,
            raw_val=raw_val,
            raw_test=raw_test,
            answers_val=answers_val,
            answers_test=answers_test,
            device=device,
            start_stage=args.start_stage,
        )
        summary[prob] = best_acc

    print("\n" + "="*50)
    print("Sweep Summary (best test accuracy):")
    for prob, acc in sorted(summary.items()):
        print(f"  uniform_prob={prob:.1f}  →  {acc*100:.2f}%")
    print("="*50)

    # 保存 summary
    os.makedirs(cfg["save_base_path"], exist_ok=True)
    with open(os.path.join(cfg["save_base_path"], "sweep_summary.json"), "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    print(f"Summary saved to {cfg['save_base_path']}/sweep_summary.json")


if __name__ == "__main__":
    main()
