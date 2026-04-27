"""
eval_latent_sweep_direct.py

测试 DirectLatentModel 在不同 latent token 数量（0 到 n_latent_max）下的推理准确率，
绘制折线图。

用法:
    python eval_latent_sweep_direct.py \
        --checkpoint models/direct_latent/direct_latent_prosqa_best.pt \
        --val_path data/prosqa_test.json \
        --n_latent_max 6 \
        --output_json results/latent_sweep_direct.json \
        --output_plot  results/latent_sweep_direct.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel

from dataset import get_dataset
from direct_latent_dataset import get_direct_latent_eval_dataset, DirectLatentCollator
from direct_latent_model import DirectLatentModel
from translator_v3 import CoconutTranslator


def load_model(checkpoint_path, n_latent, device):
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"

    special_tokens = ["<latent>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")
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
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.to(device).eval()
    return model, tokenizer, latent_id, start_id, end_id


@torch.no_grad()
def evaluate_with_n_latent(model, tokenizer, base_dataset, answers,
                            n_latent, latent_id, start_id, end_id, device):
    """构建只含 n_latent 个 latent token 的评估集，贪心解码计算准确率。"""
    # 临时改 model.n_latent
    original_n = model.n_latent
    model.n_latent = n_latent

    eval_ds = get_direct_latent_eval_dataset(
        base_dataset=base_dataset,
        n_latent=n_latent,
        start_id=start_id, latent_id=latent_id, end_id=end_id,
    )
    for col in ["answer", "steps"]:
        if col in eval_ds.column_names:
            eval_ds = eval_ds.remove_columns(col)

    collator = DirectLatentCollator(tokenizer=tokenizer, latent_id=latent_id, p_mask=1.0)
    loader   = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collator)

    correct, total = 0, 0
    for batch in loader:
        idx       = batch["idx"][0].item()
        answer    = answers[idx]
        input_ids = batch["input_ids"].to(device)

        gen_ids = model.generate(input_ids, tokenizer, max_new_tokens=128, show_thoughts=False)
        text    = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred    = text.split("#")[-1].replace(",", "").strip()
        correct += int(pred == answer)
        total   += 1

    model.n_latent = original_n
    return correct / total if total > 0 else 0.0, correct, total


def plot_results(results, output_path, title):
    ns   = [r["n_latent"]  for r in results]
    accs = [r["accuracy"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, accs, marker="o", linewidth=2, markersize=8, color="steelblue")

    for x, y in zip(ns, accs):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)

    ax.set_xlabel("Number of Latent Tokens", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(ns)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="models/direct_latent/direct_latent_prosqa_best.pt")
    parser.add_argument("--val_path",     default="data/prosqa_test.json")
    parser.add_argument("--n_latent_max", type=int, default=6)
    parser.add_argument("--output_json",  default="results/latent_sweep_direct.json")
    parser.add_argument("--output_plot",  default="results/latent_sweep_direct.png")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading model (n_latent=max for init) ...")
    model, tokenizer, latent_id, start_id, end_id = load_model(
        args.checkpoint, args.n_latent_max, device
    )

    print("Loading dataset ...")
    base_dataset = get_dataset(args.val_path, tokenizer)
    answers      = [d["answer"].replace(",", "").strip()
                    for d in json.load(open(args.val_path))]

    results = []
    for n in range(0, args.n_latent_max + 1):
        print(f"\n--- n_latent={n} ---")
        acc, correct, total = evaluate_with_n_latent(
            model, tokenizer, base_dataset, answers,
            n, latent_id, start_id, end_id, device,
        )
        print(f"  {correct}/{total} = {acc*100:.2f}%")
        results.append({"n_latent": n, "accuracy": acc, "correct": correct, "total": total})

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    plot_results(
        results, args.output_plot,
        title="DirectLatentModel: Accuracy vs. Number of Latent Tokens\n(ProsQA test set)",
    )

    print("\n=== Summary ===")
    for r in results:
        print(f"  n_latent={r['n_latent']}: {r['accuracy']*100:.2f}%  ({r['correct']}/{r['total']})")


if __name__ == "__main__":
    main()
