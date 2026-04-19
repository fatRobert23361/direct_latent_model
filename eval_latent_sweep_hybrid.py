"""
eval_latent_sweep_hybrid.py

测试 CoconutWithTranslator（混合模型）在不同 latent stage 数（0-max_stage）下的推理准确率。
对应纯 Coconut 版本：eval_latent_sweep.py

用法:
    python eval_latent_sweep_hybrid.py \
        --checkpoint models/uniform_sweep/uniform_prob_0p0/uniform_prob_0p0_best.pt \
        --val_path data/prosqa_test.json \
        --max_stage 6 \
        --c_thought 2 \
        --output_json results/latent_sweep_hybrid.json \
        --output_plot  results/latent_sweep_hybrid.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from mixed import CoconutWithTranslator
from mixed_dataset import get_question_latent_dataset, MyCollator
from dataset import get_dataset
from translator_v3 import CoconutTranslator
from utils import Config


def build_config(val_path, max_stage, c_thought):
    return Config({
        "coconut": True,
        "cot": False,
        "no_thoughts": False,
        "no_cot": False,
        "c_thought": c_thought,
        "max_latent_stage": max_stage,
        "pad_latent_to_max": True,
        "uniform_prob": 0.0,
        "epochs_per_stage": 5,
        "val_path": val_path,
        "debug": False,
    })


def load_hybrid_model(checkpoint_path, device):
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    special_tokens = ["<latent>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    pad_id    = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    print(f"  latent_id={latent_id}, start_id={start_id}, end_id={end_id}, vocab_size={vocab_size}")

    base_model = GPT2LMHeadModel.from_pretrained(model_id)
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

    model = CoconutWithTranslator(
        base_causallm=base_model,
        translator=translator,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model, tokenizer, latent_id, start_id, end_id


@torch.no_grad()
def evaluate_stage(model, tokenizer, base_dataset, configs, stage,
                   latent_id, start_id, end_id, answers_val, device):
    """评估某个 latent stage 下的生成准确率。"""
    dataset = get_question_latent_dataset(
        scheduled_stage=stage,
        base_dataset=base_dataset,
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        no_special_marker=False,
    )

    # mixed_dataset 保留了 answer/steps 字符串列，collator 无法 tensorize，需提前移除
    for col in ["answer", "steps"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)

    collator = MyCollator(tokenizer=tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collator
    )

    correct, total = 0, 0
    for batch in dataloader:
        test_idx = batch["idx"][0].item()
        answer   = answers_val[test_idx]

        input_ids = batch["input_ids"].to(device)

        gen_ids = model.generate(input_ids, tokenizer, max_new_tokens=128, show_thoughts=False)

        text_output   = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        answer_output = text_output.split("#")[-1].replace(",", "").strip()

        correct += int(answer_output == answer)
        total   += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def plot_results(results, output_path, title_suffix=""):
    stages = [r["stage"] for r in results]
    accs   = [r["accuracy"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stages, accs, marker="o", linewidth=2, markersize=8, color="steelblue")

    for x, y in zip(stages, accs):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)

    ax.set_xlabel("Latent Stage", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title(f"CoconutWithTranslator: Accuracy vs. Latent Stage\n(ProsQA test set{title_suffix})", fontsize=13)
    ax.set_xticks(stages)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--val_path",    default="data/prosqa_test.json")
    parser.add_argument("--max_stage",   type=int, default=6)
    parser.add_argument("--c_thought",   type=int, default=1)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_plot", default=None)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 默认把结果放在 checkpoint 同级目录
    ckpt_dir = os.path.dirname(args.checkpoint)
    if args.output_json is None:
        args.output_json = os.path.join(ckpt_dir, "latent_sweep_results.json")
    if args.output_plot is None:
        args.output_plot = os.path.join(ckpt_dir, "latent_sweep_results.png")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    print("Loading hybrid model ...")
    model, tokenizer, latent_id, start_id, end_id = load_hybrid_model(args.checkpoint, device)

    configs = build_config(args.val_path, args.max_stage, args.c_thought)

    print("Loading dataset ...")
    base_dataset = get_dataset(args.val_path, tokenizer)
    answers_val  = [
        d["answer"].replace(",", "").strip()
        for d in json.load(open(args.val_path))
    ]

    results = []
    for stage in range(args.max_stage + 1):
        n_latent = stage * args.c_thought
        print(f"\n--- Stage {stage} ({n_latent} latent tokens) ---")
        acc, correct, total = evaluate_stage(
            model, tokenizer, base_dataset, configs,
            stage, latent_id, start_id, end_id, answers_val, device,
        )
        print(f"  Accuracy: {correct}/{total} = {acc*100:.2f}%")
        results.append({
            "stage":    stage,
            "n_latent": n_latent,
            "accuracy": acc,
            "correct":  correct,
            "total":    total,
        })

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    title_suffix = f", c_thought={args.c_thought}"
    plot_results(results, args.output_plot, title_suffix)

    print("\n=== Summary ===")
    for r in results:
        print(f"  stage={r['stage']} ({r['n_latent']} latent tokens): "
              f"{r['accuracy']*100:.2f}% ({r['correct']}/{r['total']})")


if __name__ == "__main__":
    main()
