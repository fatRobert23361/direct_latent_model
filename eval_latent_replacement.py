"""
eval_latent_replacement.py

测试 vanilla COCONUT 中替换单个 latent 位置对推理准确率的影响。

实验设计：
  baseline        — 不替换，正常推理
  replace_pos_k   — 只替换第 k 个 latent（k=0..stage-1），其余不变
                    替换方式分两种：random（随机正态）/ other（另一条样本的同位置向量）

输出：
  - results JSON：每个位置 × 两种替换方式的准确率
  - 折线图：x 轴 = 替换位置，y 轴 = 准确率，baseline 画水平参考线

用法:
    python eval_latent_replacement.py \
        --checkpoint models/prosqa-coconut/checkpoint_38 \
        --val_path data/prosqa_test.json \
        --stage 6 \
        --output_json results/latent_replacement.json \
        --output_plot  results/latent_replacement.png
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut import Coconut
from dataset import get_dataset, get_question_latent_dataset, MyCollator
from utils import Config


# ---------------------------------------------------------------------------
# 配置与模型加载
# ---------------------------------------------------------------------------

def build_config(val_path, stage):
    return Config({
        "coconut": True,
        "cot": False,
        "no_thoughts": False,
        "no_cot": False,
        "c_thought": 1,
        "max_latent_stage": stage,
        "pad_latent_to_max": True,
        "uniform_prob": 0.0,
        "epochs_per_stage": 5,
        "val_path": val_path,
        "debug": False,
    })


def load_model(checkpoint_path, device):
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")

    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    base_model.resize_token_embeddings(len(tokenizer))
    model = Coconut(base_model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model, tokenizer, latent_id, start_id, end_id


# ---------------------------------------------------------------------------
# 收集所有样本的全部 latent 向量池
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_all_latents(model, dataloader, n_latent, device):
    """
    遍历 dataloader，对每个样本运行一次 forward，
    收集全部 n_latent 个 latent 位置的 hidden state。

    Returns:
        latent_pool: List[ Tensor(n_latent, hidden_size) | None ]
    """
    latent_pool = []

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        position_ids   = torch.arange(input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        labels         = input_ids.clone()

        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

        # latent_states[k]: (n_samples_with_latent, 1, hidden_size) 或 None
        vecs = []
        for ls in outputs.latent_states:
            if ls is not None:
                vecs.append(ls[0, 0, :].detach().cpu())   # (hidden_size,)

        if len(vecs) == n_latent:
            latent_pool.append(torch.stack(vecs))          # (n_latent, hidden_size)
        else:
            latent_pool.append(None)

    return latent_pool


# ---------------------------------------------------------------------------
# 带单位置替换的前向 + 贪心解码
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_single_replacement(model, input_ids, replace_pos, replacement_embed,
                                     device, max_new_tokens=128):
    """
    只替换第 replace_pos 个 latent（0-based）的向量为 replacement_embed。
    其余 latent 照常由 hidden state 填充。

    replace_pos:      int，0-based；-1 表示不替换（baseline）
    replacement_embed: Tensor (hidden_size,) 或 None
    """
    tokens = input_ids[0].detach().tolist()

    latent_indices = (input_ids == model.latent_token_id).nonzero()

    if len(latent_indices) == 0 or replace_pos < 0 or replacement_embed is None:
        # 正常 generate（baseline）
        out = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
        )
        return out[0].tolist()

    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents = max(len(l) for l in latent_lists)

    inputs_embeds  = model.embedding(input_ids)
    attention_mask = torch.ones_like(input_ids)
    position_ids   = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)

    next_compute_range = (0, latent_indices[:, 1].min().item())
    kv_cache = None

    for pass_idx in range(max_n_latents):
        if kv_cache is None:
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                output_hidden_states=True,
            )
            hidden_states_offset = 0
        else:
            past_key_values = [
                (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                for k, v in kv_cache
            ]
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            hidden_states_offset = next_compute_range[0]

        hidden_states = outputs.hidden_states[-1]
        kv_cache = outputs.past_key_values

        next_compute_range = (
            next_compute_range[1],
            input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1,
        )

        filling_indices = [
            (batch_idx, latent_lists[batch_idx][pass_idx])
            for batch_idx in range(len(latent_lists))
            if len(latent_lists[batch_idx]) > pass_idx
        ]

        # 逐位置修改 inputs_embeds（避免 in-place 操作）
        tensor_list = [
            [inputs_embeds[b, p, :] for p in range(inputs_embeds.shape[1])]
            for b in range(inputs_embeds.shape[0])
        ]

        for batch_idx, token_idx in filling_indices:
            if pass_idx == replace_pos:
                # 替换该轮 latent 向量
                tensor_list[batch_idx][token_idx] = replacement_embed.to(device)
            else:
                # 正常：用前一个位置的 hidden state 填充
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

        inputs_embeds = torch.stack([
            torch.stack(tensor_list[b])
            for b in range(inputs_embeds.shape[0])
        ])

    # ---- 最终 pass ----
    past_kv = (
        [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
         for k, v in kv_cache]
        if kv_cache else None
    )
    final_outputs = model.base_causallm(
        inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
        attention_mask=attention_mask[:, :next_compute_range[1]],
        position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
        past_key_values=past_kv,
        output_hidden_states=False,
    )

    # 贪婪解码
    next_token = torch.argmax(final_outputs.logits[0, -1]).item()
    tokens.append(next_token)
    new_inputs_embeds = torch.cat(
        [inputs_embeds,
         model.embedding(torch.tensor(next_token, device=device)).view(1, 1, -1)],
        dim=1,
    )

    for _ in range(max_new_tokens - 1):
        out = model.base_causallm(inputs_embeds=new_inputs_embeds)
        next_token = torch.argmax(out.logits[0, -1]).item()
        if next_token == model.eos_token_id:
            break
        tokens.append(next_token)
        new_inputs_embeds = torch.cat(
            [new_inputs_embeds,
             model.embedding(torch.tensor(next_token, device=device)).view(1, 1, -1)],
            dim=1,
        )

    return tokens


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_replace_pos(model, tokenizer, dataloader, answers_val, latent_pool,
                         replace_pos, mode, device):
    """
    replace_pos: 0-based，替换第几个 latent；-1 表示 baseline（不替换）
    mode: "random" 或 "other"（baseline 时忽略）
    """
    hidden_size = model.embedding.embedding_dim
    correct, total = 0, 0

    for sample_i, batch in enumerate(dataloader):
        test_idx  = batch["idx"][0].item()
        answer    = answers_val[test_idx]
        input_ids = batch["input_ids"].to(device)

        if replace_pos < 0:
            replacement = None
        elif mode == "random":
            replacement = torch.randn(hidden_size, device=device)
        else:  # "other"
            candidates = [
                v[replace_pos] for j, v in enumerate(latent_pool)
                if j != sample_i and v is not None
            ]
            replacement = (random.choice(candidates).to(device)
                           if candidates else torch.randn(hidden_size, device=device))

        tokens = generate_with_single_replacement(
            model, input_ids, replace_pos, replacement, device, max_new_tokens=128
        )

        text_output   = tokenizer.decode(tokens, skip_special_tokens=True)
        answer_output = text_output.split("#")[-1].replace(",", "").strip()
        correct += int(answer_output == answer)
        total   += 1

    return correct / total if total > 0 else 0.0, correct, total


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_results(baseline_acc, results_random, results_other, n_latent, output_path):
    positions   = list(range(1, n_latent + 1))   # 1-based for display
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
    ax.set_title("Vanilla COCONUT: Effect of Single-Position Latent Replacement\n(ProsQA test set)",
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
    parser.add_argument("--checkpoint",  default="models/prosqa-coconut/checkpoint_38")
    parser.add_argument("--val_path",    default="data/prosqa_test.json")
    parser.add_argument("--stage",       type=int, default=6,
                        help="评估时使用的 latent stage 数（即注入多少个 latent token）")
    parser.add_argument("--output_json", default="results/latent_replacement.json")
    parser.add_argument("--output_plot", default="results/latent_replacement.png")
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
    model, tokenizer, latent_id, start_id, end_id = load_model(args.checkpoint, device)

    configs = build_config(args.val_path, args.stage)

    print("Loading dataset ...")
    base_dataset = get_dataset(args.val_path, tokenizer)
    answers_val  = [
        d["answer"].replace(",", "").strip()
        for d in json.load(open(args.val_path))
    ]

    dataset = get_question_latent_dataset(
        scheduled_stage=args.stage,
        base_dataset_valid=base_dataset,
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        no_special_marker=False,
    )

    collator   = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collator
    )

    # ---- Step 1: 收集所有样本的全部 latent hidden state ----
    print(f"\nCollecting all latent states (stage={args.stage}) ...")
    latent_pool = collect_all_latents(model, dataloader, args.stage, device)
    valid_count = sum(1 for v in latent_pool if v is not None)
    print(f"  Collected {valid_count}/{len(latent_pool)} valid latents")

    # ---- Step 2: baseline ----
    print("\n--- baseline (no replacement) ---")
    baseline_acc, bc, bt = evaluate_replace_pos(
        model, tokenizer, dataloader, answers_val, latent_pool,
        replace_pos=-1, mode="random", device=device,
    )
    print(f"  {bc}/{bt} = {baseline_acc*100:.2f}%")

    # ---- Step 3: 逐位置替换 ----
    results_random = {}
    results_other  = {}

    for pos in range(args.stage):
        for mode, store in [("random", results_random), ("other", results_other)]:
            print(f"\n--- replace pos={pos+1}/{args.stage}, mode={mode} ---")
            acc, correct, total = evaluate_replace_pos(
                model, tokenizer, dataloader, answers_val, latent_pool,
                replace_pos=pos, mode=mode, device=device,
            )
            store[pos] = {"accuracy": acc, "correct": correct, "total": total}
            drop = (baseline_acc - acc) * 100
            print(f"  {correct}/{total} = {acc*100:.2f}%  (drop: {drop:+.2f}pp)")

    # ---- 保存 JSON ----
    output = {
        "checkpoint": args.checkpoint,
        "stage":      args.stage,
        "baseline":   {"accuracy": baseline_acc, "correct": bc, "total": bt},
        "random":     {str(k + 1): v for k, v in results_random.items()},
        "other":      {str(k + 1): v for k, v in results_other.items()},
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    plot_results(baseline_acc, results_random, results_other, args.stage, args.output_plot)

    # ---- 汇总 ----
    print("\n=== Summary ===")
    print(f"  {'pos':>4}  {'random':>10}  {'drop':>8}  |  {'other':>10}  {'drop':>8}")
    print(f"  {'base':>4}  {baseline_acc*100:>9.2f}%  {'—':>8}  |  {baseline_acc*100:>9.2f}%  {'—':>8}")
    for pos in range(args.stage):
        ra = results_random[pos]["accuracy"]
        oa = results_other[pos]["accuracy"]
        print(f"  {pos+1:>4}  {ra*100:>9.2f}%  {(baseline_acc-ra)*100:>+7.2f}pp  |  "
              f"{oa*100:>9.2f}%  {(baseline_acc-oa)*100:>+7.2f}pp")


if __name__ == "__main__":
    main()
