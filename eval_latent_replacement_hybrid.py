"""
eval_latent_replacement_hybrid.py

测试混合模型 (CoconutWithTranslator) 中将第一个 latent token 的 embedding
替换为其他数据的 latent 或随机向量后，对推理准确率的影响。

对应纯 Coconut 版本：eval_latent_replacement.py

用法:
    python eval_latent_replacement_hybrid.py \
        --checkpoint models/prosqa-coconut-hybrid_v5/prosqa-coconut-hybrid-v5_stage6_epoch7.pt \
        --val_path data/prosqa_test.json \
        --stage 6
"""

import argparse
import json
import os
import random

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from mixed import CoconutWithTranslator
from translator_v3 import CoconutTranslator
from dataset import get_dataset
from mixed_dataset import get_question_latent_dataset, MyCollator
from utils import Config


# ------------------------------------------------------------------
# 配置构建
# ------------------------------------------------------------------

def build_config(val_path, stage, c_thought=1):
    return Config({
        "coconut": True,
        "cot": False,
        "no_thoughts": False,
        "no_cot": False,
        "c_thought": c_thought,
        "max_latent_stage": stage,
        "pad_latent_to_max": True,
        "uniform_prob": 0.0,
        "epochs_per_stage": 5,
        "val_path": val_path,
        "debug": False,
    })


# ------------------------------------------------------------------
# 模型加载
# 注意：混合模型的 token 顺序为 <latent>(50257), <|start-latent|>(50258), <|end-latent|>(50259)
# ------------------------------------------------------------------

def load_hybrid_model(checkpoint_path, device):
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 混合模型的特殊 token 添加顺序与纯 Coconut 不同
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

    # 混合模型 checkpoint 格式: {"model_state_dict": ..., "optimizer_state_dict": ..., ...}
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


# ------------------------------------------------------------------
# 替换版 generate（复现 CoconutWithTranslator.forward 逻辑，在 pass_idx==0 时注入）
# ------------------------------------------------------------------

@torch.no_grad()
def generate_with_replaced_latent(model, input_ids, replacement_embed, max_new_tokens=128):
    """
    与 CoconutWithTranslator.generate() 等价，但在 pass_idx==0 时将
    第一个 latent 位置的 hidden state 替换为 replacement_embed。

    Args:
        model:              CoconutWithTranslator 实例
        input_ids:          shape (1, seq_len)，包含 <latent> 占位符
        replacement_embed:  shape (hidden_size,)，用于替换第一个 latent
                            传 None 则退化为正常 generate
        max_new_tokens:     最大生成 token 数

    Returns:
        all_tokens: List[int]，包含输入和生成的全部 token id
    """
    device = input_ids.device
    all_tokens = input_ids[0].detach().tolist()

    latent_indices = (input_ids == model.latent_token_id).nonzero()

    if len(latent_indices) == 0:
        # 无 latent token：直接走正常 generate
        out = model.generate(input_ids=input_ids, tokenizer=None,
                             max_new_tokens=max_new_tokens, show_thoughts=False)
        return all_tokens + out[0].tolist()

    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents = max(len(l) for l in latent_lists)

    inputs_embeds = model.embedding(input_ids)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).reshape(1, -1)

    next_compute_range = (0, latent_indices[:, 1].min().item())
    kv_cache = None

    # ---- 多 pass 前向，逐个填充 latent 位置 ----
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
            (i, latent_lists[i][pass_idx])
            for i in range(len(latent_lists))
            if len(latent_lists[i]) > pass_idx
        ]

        if filling_indices:
            batch_idx_t = torch.tensor([b for b, _ in filling_indices], device=device)
            token_idx_t = torch.tensor([t for _, t in filling_indices], device=device)

            if pass_idx == 0 and replacement_embed is not None:
                # 替换所有 batch 内的第一个 latent 为同一个 replacement
                new_values = replacement_embed.to(device).unsqueeze(0).expand(len(filling_indices), -1)
            else:
                new_values = torch.stack([
                    hidden_states[b, t - 1 - hidden_states_offset, :]
                    for b, t in filling_indices
                ])

            inputs_embeds = inputs_embeds.index_put((batch_idx_t, token_idx_t), new_values)

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
    )

    # 贪婪解码
    next_token = torch.argmax(final_outputs.logits[0, -1]).item()
    all_tokens.append(next_token)
    curr_embeds = torch.cat(
        [inputs_embeds,
         model.embedding(torch.tensor([[next_token]], device=device))],
        dim=1,
    )

    for _ in range(max_new_tokens - 1):
        out = model.base_causallm(inputs_embeds=curr_embeds)
        next_token = torch.argmax(out.logits[0, -1]).item()
        if next_token == model.eos_token_id:
            break
        all_tokens.append(next_token)
        curr_embeds = torch.cat(
            [curr_embeds,
             model.embedding(torch.tensor([[next_token]], device=device))],
            dim=1,
        )

    return all_tokens


# ------------------------------------------------------------------
# 收集第一个 latent 的 hidden state
# ------------------------------------------------------------------

@torch.no_grad()
def collect_first_latents(model, dataloader, device):
    """
    遍历 dataloader，对每个样本运行一次 forward，
    收集第一个 latent 位置（pass_idx=0）的 hidden state。

    Returns:
        latents: List[Tensor|None]，每个元素 shape (hidden_size,)
        idxs:    List[int]，对应的样本 idx
    """
    latents = []
    idxs = []

    for batch in dataloader:
        test_idx = batch["idx"][0].item()
        input_ids     = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        position_ids  = batch["position_ids"].to(device)
        labels        = input_ids.clone()

        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

        if outputs.latent_states and outputs.latent_states[0] is not None:
            # latent_states[0]: (1, 1, hidden_size) for batch_size=1
            latent_vec = outputs.latent_states[0][0, 0, :]  # (hidden_size,)
            latents.append(latent_vec)
        else:
            latents.append(None)

        idxs.append(test_idx)

    return latents, idxs


# ------------------------------------------------------------------
# 评估（三种模式）
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate_mode(model, tokenizer, dataloader, answers_val, device,
                  latent_pool=None, mode="baseline"):
    """
    mode:
      "baseline" — 不替换，正常 generate
      "random"   — 第一个 latent 替换为 N(0,1) 随机向量
      "other"    — 第一个 latent 替换为另一条数据的 latent
    """
    hidden_size = model.embedding.embedding_dim
    correct, total = 0, 0

    for sample_i, batch in enumerate(dataloader):
        test_idx = batch["idx"][0].item()
        answer   = answers_val[test_idx]
        input_ids = batch["input_ids"].to(device)

        if mode == "baseline":
            replacement = None
        elif mode == "random":
            replacement = torch.randn(hidden_size, device=device)
        elif mode == "other":
            candidates = [
                v for j, v in enumerate(latent_pool)
                if j != sample_i and v is not None
            ]
            replacement = (random.choice(candidates).to(device)
                           if candidates
                           else torch.randn(hidden_size, device=device))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        tokens = generate_with_replaced_latent(
            model, input_ids, replacement_embed=replacement, max_new_tokens=128
        )

        text_output   = tokenizer.decode(tokens, skip_special_tokens=True)
        answer_output = text_output.split("#")[-1].replace(",", "").strip()

        correct += int(answer_output == answer)
        total   += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="models/prosqa-coconut-hybrid_v5/prosqa-coconut-hybrid-v5_stage6_epoch7.pt")
    parser.add_argument("--val_path",    default="data/prosqa_test.json")
    parser.add_argument("--stage",       type=int, default=6,
                        help="评估时使用的 latent stage 数")
    parser.add_argument("--c_thought",   type=int, default=1)
    parser.add_argument("--output_json", default="results/latent_replacement_hybrid.json")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading hybrid model ...")
    model, tokenizer, latent_id, start_id, end_id = load_hybrid_model(args.checkpoint, device)

    configs = build_config(args.val_path, args.stage, args.c_thought)

    print("Loading dataset ...")
    base_dataset = get_dataset(args.val_path, tokenizer)
    answers_val  = [
        d["answer"].replace(",", "").strip()
        for d in json.load(open(args.val_path))
    ]

    dataset = get_question_latent_dataset(
        scheduled_stage=args.stage,
        base_dataset=base_dataset,
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        no_special_marker=False,
    )

    # mixed_dataset 保留了 answer/steps 字符串列，collator 无法将其转为 tensor，需提前移除
    for col in ["answer", "steps"]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collator
    )

    # ---- Step 1: 收集所有样本的第一个 latent hidden state ----
    print(f"\nCollecting first-latent hidden states (stage={args.stage}) ...")
    latent_pool, sample_idxs = collect_first_latents(model, dataloader, device)
    valid_count = sum(1 for v in latent_pool if v is not None)
    print(f"  Collected {valid_count}/{len(latent_pool)} valid latents")

    # ---- Step 2: 三种模式评估 ----
    results = {}
    for mode in ("baseline", "random", "other"):
        print(f"\n--- Mode: {mode} ---")
        acc, correct, total = evaluate_mode(
            model, tokenizer, dataloader, answers_val, device,
            latent_pool=latent_pool, mode=mode,
        )
        results[mode] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  {correct}/{total} = {acc * 100:.2f}%")

    # ---- 保存结果 ----
    output = {
        "checkpoint": args.checkpoint,
        "stage": args.stage,
        "c_thought": args.c_thought,
        "results": results,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    print("\n=== Summary ===")
    for mode, r in results.items():
        print(f"  {mode:10s}: {r['accuracy'] * 100:.2f}%  ({r['correct']}/{r['total']})")

    baseline_acc = results["baseline"]["accuracy"]
    for mode in ("random", "other"):
        drop = (baseline_acc - results[mode]["accuracy"]) * 100
        print(f"  Accuracy drop ({mode} vs baseline): {drop:+.2f}pp")


if __name__ == "__main__":
    main()
