"""
Evaluate Coconut model accuracy with different numbers of latent thought tokens (0-6).
Runs on a single GPU, no torchrun needed.

Usage:
    python eval_latent_sweep.py \
        --checkpoint models/prosqa-coconut/checkpoint_42 \
        --val_path data/prosqa_test.json \
        --output_json results/latent_sweep.json
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut import Coconut
from dataset import get_dataset, get_question_latent_dataset, MyCollator
from utils import Config


LATENT_STAGES = list(range(15))  # 0 .. 14


def build_config(val_path):
    return Config({
        "coconut": True,
        "cot": False,
        "no_thoughts": False,
        "no_cot": False,
        "c_thought": 1,
        "max_latent_stage": max(LATENT_STAGES),  # must be >= max stage to avoid capping
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

    # Add special tokens (must match training)
    special_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id   = tokenizer.convert_tokens_to_ids("<|end-latent|>")
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


@torch.no_grad()
def evaluate_stage(model, tokenizer, base_dataset, configs, stage,
                   latent_id, start_id, end_id, answers_val, device):
    """Evaluate accuracy for a given number of latent tokens (= stage * c_thought)."""
    dataset = get_question_latent_dataset(
        scheduled_stage=stage,
        base_dataset_valid=base_dataset,
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        no_special_marker=False,
    )

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
    )

    correct, total = 0, 0

    for batch in dataloader:
        test_idx = batch["idx"][0].item()
        answer = answers_val[test_idx]

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            synced_gpus=False,
        )

        text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_output = text_output.split("#")[-1].replace(",", "").strip()

        correct += int(answer_output == answer)
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def plot_results(results, output_path):
    stages = [r["n_latent"] for r in results]
    accs   = [r["accuracy"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stages, accs, marker="o", linewidth=2, markersize=8, color="steelblue")

    for x, y in zip(stages, accs):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)

    ax.set_xlabel("Number of Latent Thought Tokens", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("Coconut: Accuracy vs. Number of Latent Thought Tokens\n(ProsQA test set)", fontsize=14)
    ax.set_xticks(stages)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/prosqa-coconut/checkpoint_42")
    parser.add_argument("--val_path",   default="data/prosqa_test.json")
    parser.add_argument("--output_json", default="results/latent_sweep.json")
    parser.add_argument("--output_plot", default="results/latent_sweep.png")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading model...")
    model, tokenizer, latent_id, start_id, end_id = load_model(args.checkpoint, device)

    configs = build_config(args.val_path)

    print("Loading dataset...")
    base_dataset = get_dataset(args.val_path, tokenizer)
    answers_val = [
        d["answer"].replace(",", "").strip()
        for d in json.load(open(args.val_path))
    ]

    results = []
    for stage in LATENT_STAGES:
        n_latent = stage  # c_thought=1, so tokens = stage
        print(f"\n--- Evaluating with {n_latent} latent token(s) (stage={stage}) ---")
        acc, correct, total = evaluate_stage(
            model, tokenizer, base_dataset, configs,
            stage, latent_id, start_id, end_id, answers_val, device,
        )
        print(f"  Accuracy: {correct}/{total} = {acc*100:.2f}%")
        results.append({"n_latent": n_latent, "accuracy": acc, "correct": correct, "total": total})

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    plot_results(results, args.output_plot)

    print("\n=== Summary ===")
    for r in results:
        print(f"  n_latent={r['n_latent']}: {r['accuracy']*100:.2f}% ({r['correct']}/{r['total']})")


if __name__ == "__main__":
    main()
