"""
eval_latent_replacement_direct.py

测试 DirectLatentModel 中替换单个 latent 位置对推理准确率的影响。

实验设计：
  baseline        — 不替换，正常推理
  replace_pos_k   — 只替换第 k 个 latent（k=0..n_latent-1），其余不变
                    替换方式分两种：random（随机正态）/ other（另一条样本的同位置向量）

输出：
  - results JSON：每个位置 × 两种替换方式的准确率
  - 折线图：x 轴 = 替换位置，y 轴 = 准确率，baseline 画水平参考线

用法:
    python eval_latent_replacement_direct.py \
        --checkpoint models/direct_latent/direct_latent_prosqa_best.pt \
        --val_path data/prosqa_test.json \
        --n_latent 6 \
        --output_json results/latent_direct/latent_replacement_direct.json \
        --output_plot  results/latent_direct/latent_replacement_direct.png
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel

from dataset import get_dataset
from direct_latent_dataset import get_direct_latent_eval_dataset, DirectLatentCollator
from direct_latent_model import DirectLatentModel
from translator_v3 import CoconutTranslator


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, n_latent, device):
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    special_tokens = ["<latent>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    latent_id  = tokenizer.convert_tokens_to_ids("<latent>")
    start_id   = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id     = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    vocab_size = len(tokenizer)

    base_model = GPT2LMHeadModel.from_pretrained(model_id)
    base_model.resize_token_embeddings(vocab_size)

    translator = CoconutTranslator(
        hidden_size=base_model.config.n_embd,
        vocab_size=vocab_size,
        start_id=start_id, end_id=end_id,
        pad_id=tokenizer.pad_token_id, eos_id=tokenizer.eos_token_id,
        mode="context_latent",
    )

    model = DirectLatentModel(
        base_causallm=base_model,
        translator=translator,
        n_latent=n_latent,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, tokenizer, latent_id, start_id, end_id


# ---------------------------------------------------------------------------
# 收集所有样本的 latent 向量池
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_latents(model, loader, device):
    """
    对 loader 中每条样本运行一次 forward，收集全部 n_latent 个 latent 向量。

    Returns:
        latent_pool: List[ Tensor(n_latent, hidden_size) | None ]
    """
    latent_pool = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        fwd = model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids.clone(),
            position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
        )
        vecs = [s[0, 0, :] for s in fwd.latent_states if s is not None]
        latent_pool.append(torch.stack(vecs) if vecs else None)  # (n_latent, H)
    return latent_pool


# ---------------------------------------------------------------------------
# 带单位置替换的前向 + 贪心解码
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_single_replacement(model, input_ids, replace_pos, replacement_vec, device,
                                     max_new_tokens=128):
    """
    只替换第 replace_pos 个 latent（0-based）的向量为 replacement_vec。
    其余 latent 照常由 hidden state 填充。

    replace_pos:     int，0-based，-1 表示不替换（baseline）
    replacement_vec: Tensor (hidden_size,) 或 None
    """
    latent_id      = model.latent_token_id
    latent_indices = (input_ids == latent_id).nonzero()

    if len(latent_indices) == 0 or replace_pos < 0 or replacement_vec is None:
        return model.generate(input_ids, None, max_new_tokens=max_new_tokens, show_thoughts=False)

    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents = max(len(l) for l in latent_lists)

    inputs_embeds  = model.embedding(input_ids)
    attention_mask = torch.ones_like(input_ids)
    position_ids   = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    next_range     = (0, latent_indices[:, 1].min().item())
    kv_cache       = None

    for pass_idx in range(max_n_latents):
        if kv_cache is None:
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, next_range[0]:next_range[1], :],
                attention_mask=attention_mask[:, next_range[0]:next_range[1]],
                position_ids=position_ids[:, next_range[0]:next_range[1]],
                output_hidden_states=True,
            )
            hs_offset = 0
        else:
            past_kv = [(k[:, :, :next_range[0], :], v[:, :, :next_range[0], :])
                       for k, v in kv_cache]
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, next_range[0]:next_range[1], :],
                attention_mask=attention_mask[:, :next_range[1]],
                position_ids=position_ids[:, next_range[0]:next_range[1]],
                past_key_values=past_kv,
                output_hidden_states=True,
            )
            hs_offset = next_range[0]

        hidden_states = outputs.hidden_states[-1]
        kv_cache      = outputs.past_key_values

        next_range = (
            next_range[1],
            input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_range[1] + 1,
        )

        filling = [(b, latent_lists[b][pass_idx])
                   for b in range(len(latent_lists)) if len(latent_lists[b]) > pass_idx]
        b_idx = torch.tensor([b for b, _ in filling], device=device)
        t_idx = torch.tensor([t for _, t in filling], device=device)

        if pass_idx == replace_pos:
            # 只替换这一轮的 latent 向量
            new_vals = replacement_vec.unsqueeze(0).expand(len(filling), -1)
        else:
            new_vals = torch.stack([
                hidden_states[b, t - 1 - hs_offset, :]
                for b, t in filling
            ])
        inputs_embeds = inputs_embeds.index_put((b_idx, t_idx), new_vals)

    # 最后一轮
    past_kv = [(k[:, :, :next_range[0], :], v[:, :, :next_range[0], :])
               for k, v in kv_cache] if kv_cache else None
    final_out = model.base_causallm(
        inputs_embeds=inputs_embeds[:, next_range[0]:next_range[1], :],
        attention_mask=attention_mask[:, :next_range[1]],
        position_ids=position_ids[:, next_range[0]:next_range[1]],
        past_key_values=past_kv,
    )

    next_token  = torch.argmax(final_out.logits[0, -1]).item()
    generated   = [next_token]
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


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_replace_pos(model, tokenizer, loader, answers, latent_pool,
                         replace_pos, mode, device):
    """
    replace_pos: 0-based，替换第几个 latent；-1 表示 baseline（不替换）
    mode: "random" 或 "other"（baseline 时忽略）
    """
    hidden_size = model.embedding.embedding_dim
    correct, total = 0, 0

    for i, batch in enumerate(loader):
        idx       = batch["idx"][0].item()
        answer    = answers[idx]
        input_ids = batch["input_ids"].to(device)

        if replace_pos < 0:
            replacement = None
        elif mode == "random":
            replacement = torch.randn(hidden_size, device=device)
        else:  # "other"
            candidates = [v[replace_pos] for j, v in enumerate(latent_pool)
                          if j != i and v is not None]
            replacement = random.choice(candidates).to(device) if candidates else \
                          torch.randn(hidden_size, device=device)

        gen_ids = generate_with_single_replacement(
            model, input_ids, replace_pos, replacement, device
        )
        text    = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred    = text.split("#")[-1].replace(",", "").strip()
        correct += int(pred == answer)
        total   += 1

    return correct / total if total > 0 else 0.0, correct, total


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_results(baseline_acc, results_random, results_other, n_latent, output_path):
    positions = list(range(1, n_latent + 1))   # 1-based for display

    accs_random = [results_random[k]["accuracy"] * 100 for k in range(n_latent)]
    accs_other  = [results_other[k]["accuracy"]  * 100 for k in range(n_latent)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(baseline_acc * 100, color="steelblue", linewidth=2,
               linestyle="--", label=f"baseline ({baseline_acc*100:.1f}%)")
    ax.plot(positions, accs_random, marker="o", linewidth=2, markersize=8,
            color="tomato",     label="replace w/ random")
    ax.plot(positions, accs_other,  marker="s", linewidth=2, markersize=8,
            color="darkorange", label="replace w/ other sample")

    for x, y in zip(positions, accs_random):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="tomato")
    for x, y in zip(positions, accs_other):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9, color="darkorange")

    ax.set_xlabel("Replaced Latent Position (1-based)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("DirectLatentModel: Effect of Single-Position Latent Replacement\n(ProsQA test set)",
                 fontsize=13)
    ax.set_xticks(positions)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="models/direct_latent/direct_latent_prosqa_best.pt")
    parser.add_argument("--val_path",    default="data/prosqa_test.json")
    parser.add_argument("--n_latent",    type=int, default=6)
    parser.add_argument("--output_json", default="results/latent_replacement_direct.json")
    parser.add_argument("--output_plot", default="results/latent_replacement_direct.png")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading model ...")
    model, tokenizer, latent_id, start_id, end_id = load_model(
        args.checkpoint, args.n_latent, device
    )

    print("Loading dataset ...")
    base_dataset = get_dataset(args.val_path, tokenizer)
    answers      = [d["answer"].replace(",", "").strip()
                    for d in json.load(open(args.val_path))]

    eval_ds = get_direct_latent_eval_dataset(
        base_dataset=base_dataset,
        n_latent=args.n_latent, start_id=start_id,
        latent_id=latent_id, end_id=end_id,
    )
    for col in ["answer", "steps"]:
        if col in eval_ds.column_names:
            eval_ds = eval_ds.remove_columns(col)

    collator = DirectLatentCollator(tokenizer=tokenizer, latent_id=latent_id, p_mask=1.0)
    loader   = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collator)

    # ---- Step 1: 收集 latent 向量池 ----
    print(f"\nCollecting latent states (n_latent={args.n_latent}) ...")
    latent_pool = collect_latents(model, loader, device)
    valid_count = sum(1 for v in latent_pool if v is not None)
    print(f"  Collected {valid_count}/{len(latent_pool)} valid latents")

    # ---- Step 2: baseline ----
    print("\n--- baseline (no replacement) ---")
    baseline_acc, bc, bt = evaluate_replace_pos(
        model, tokenizer, loader, answers, latent_pool,
        replace_pos=-1, mode="random", device=device,
    )
    print(f"  {bc}/{bt} = {baseline_acc*100:.2f}%")

    # ---- Step 3: 逐位置替换 ----
    results_random = {}
    results_other  = {}

    for pos in range(args.n_latent):
        for mode, store in [("random", results_random), ("other", results_other)]:
            print(f"\n--- replace pos={pos+1}/{args.n_latent}, mode={mode} ---")
            acc, correct, total = evaluate_replace_pos(
                model, tokenizer, loader, answers, latent_pool,
                replace_pos=pos, mode=mode, device=device,
            )
            store[pos] = {"accuracy": acc, "correct": correct, "total": total}
            drop = (baseline_acc - acc) * 100
            print(f"  {correct}/{total} = {acc*100:.2f}%  (drop: {drop:+.2f}pp)")

    # ---- 保存 ----
    output = {
        "checkpoint":    args.checkpoint,
        "n_latent":      args.n_latent,
        "baseline":      {"accuracy": baseline_acc, "correct": bc, "total": bt},
        "random":        {str(k+1): v for k, v in results_random.items()},
        "other":         {str(k+1): v for k, v in results_other.items()},
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    plot_results(baseline_acc, results_random, results_other, args.n_latent, args.output_plot)

    # ---- 汇总 ----
    print("\n=== Summary ===")
    print(f"  {'pos':>4}  {'random':>10}  {'drop':>8}  |  {'other':>10}  {'drop':>8}")
    print(f"  {'base':>4}  {baseline_acc*100:>9.2f}%  {'—':>8}  |  {baseline_acc*100:>9.2f}%  {'—':>8}")
    for pos in range(args.n_latent):
        ra = results_random[pos]["accuracy"]
        oa = results_other[pos]["accuracy"]
        print(f"  {pos+1:>4}  {ra*100:>9.2f}%  {(baseline_acc-ra)*100:>+7.2f}pp  |  "
              f"{oa*100:>9.2f}%  {(baseline_acc-oa)*100:>+7.2f}pp")


if __name__ == "__main__":
    main()
