"""
eval_cot_accuracy.py

加载 COT SFT 模型权重，按照 train_direct_latent.py 的评估逻辑计算 accuracy：
  - 答案提取：text.split("#")[-1].replace(",", "").strip()
  - 指标：accuracy（直接字符串匹配，无 normalize）
  - 可配置评估样本数上限（num_eval_samples）

用法：
    python eval_cot_accuracy.py
    python eval_cot_accuracy.py --config args/sft_hotpot.yaml --checkpoint models/cot_hotpot/cot_best.pt
    python eval_cot_accuracy.py --num_eval_samples 500
"""

import argparse
import json

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

from dataset import get_dataset


def build_eval_prompts(raw_dataset):
    return [
        {
            "input_ids":      sample["question_tokenized"],
            "attention_mask": [1] * len(sample["question_tokenized"]),
        }
        for sample in raw_dataset
    ]


@torch.no_grad()
def evaluate(model, raw_dataset, answers_list, tokenizer, device, cfg, num_eval_samples=None):
    model.eval()

    num_eval = len(raw_dataset)
    if num_eval_samples is not None:
        num_eval = min(num_eval_samples, num_eval)

    raw_dataset  = raw_dataset.select(range(num_eval))
    answers_list = answers_list[:num_eval]
    prompts      = build_eval_prompts(raw_dataset)

    max_new_tokens = cfg.get("max_new_tokens_eval", 300)
    correct, total = 0, 0

    for i, prompt in enumerate(tqdm(prompts, desc="eval")):
        input_ids = torch.tensor([prompt["input_ids"]],      dtype=torch.long).to(device)
        attn_mask = torch.tensor([prompt["attention_mask"]], dtype=torch.long).to(device)

        max_new = min(max_new_tokens, 1024 - input_ids.shape[1])
        if max_new <= 0:
            total += 1
            continue

        gen_ids = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # 解码完整序列（prompt + 生成），与 train_direct_latent.py 一致
        text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        pred = text.split("#")[-1].replace(",", "").strip()

        answer     = answers_list[i]
        is_correct = (pred == answer)
        correct   += int(is_correct)
        total     += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default="args/sft_hotpot.yaml")
    parser.add_argument("--checkpoint",      default=None,
                        help="模型权重路径，默认用 save_path/name_best.pt")
    parser.add_argument("--split",           default="test", choices=["val", "test"])
    parser.add_argument("--num_eval_samples", type=int, default=None,
                        help="评估样本数上限，None 表示全量")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = GPT2LMHeadModel.from_pretrained(cfg["model_id"]).to(device)

    # 加载 checkpoint
    ckpt_path = args.checkpoint or f"{cfg['save_path']}/{cfg['name']}_best.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  → epoch={ckpt.get('epoch', '?')}  val_em={ckpt.get('val_em', '?'):.4f}")

    data_path = cfg["val_path"] if args.split == "val" else cfg["test_path"]
    print(f"Loading {args.split} set: {data_path}")
    raw_dataset  = get_dataset(data_path, tokenizer)
    answers_list = [d["answer"].replace(",", "").strip() for d in json.load(open(data_path))]

    num_eval = args.num_eval_samples or len(raw_dataset)
    print(f"Evaluating {min(num_eval, len(raw_dataset))} samples ...")

    accuracy, correct, total = evaluate(
        model, raw_dataset, answers_list, tokenizer, device, cfg,
        num_eval_samples=args.num_eval_samples,
    )

    print(f"\n[{args.split}] accuracy = {accuracy*100:.2f}%  ({correct}/{total})")
    print("(评估逻辑：text.split('#')[-1].replace(',','').strip() == answer，无 normalize)")


if __name__ == "__main__":
    main()
