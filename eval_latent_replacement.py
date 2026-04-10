"""
eval_latent_replacement.py

测试将第一个 latent token 的 embedding 替换为其他数据的 latent 或随机 embedding
对模型推理准确率的影响。

用法:
    python eval_latent_replacement.py \
        --checkpoint models/prosqa-coconut/checkpoint_38 \
        --val_path data/prosqa_test.json \
        --stage 6
"""

import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut import Coconut
from dataset import get_dataset, get_question_latent_dataset, MyCollator
from utils import Config


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


@torch.no_grad()
def generate_with_replaced_latent(model, input_ids, replacement_embed, max_new_tokens=128):
    """
    与 model.generate() 逻辑相同，但在 pass_idx==0 时，将第一个 latent 的
    hidden state 替换为 replacement_embed。

    Args:
        model:              Coconut 模型实例
        input_ids:          shape (1, seq_len)，包含 <|latent|> 占位符
        replacement_embed:  shape (1, hidden_size)，用于替换第一个 latent 的向量
                            传 None 则不替换（退化为正常 generate）
        max_new_tokens:     最大生成 token 数

    Returns:
        tokens: List[int]，包含输入和生成的全部 token id
    """
    device = input_ids.device
    tokens = input_ids[0].detach().tolist()

    latent_indices = (input_ids == model.latent_token_id).nonzero()

    if len(latent_indices) == 0:
        # 没有 latent token，直接调用普通 generate
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

        # 拆成 list of list 避免 in-place 操作
        tensor_list = [
            [inputs_embeds[b, p, :] for p in range(inputs_embeds.shape[1])]
            for b in range(inputs_embeds.shape[0])
        ]

        for batch_idx, token_idx in filling_indices:
            if pass_idx == 0 and replacement_embed is not None:
                # 替换第一个 latent 的向量
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

    # 贪婪解码生成答案
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


@torch.no_grad()
def collect_first_latents(model, dataloader, device):
    """
    遍历 dataloader，对每个样本运行一次 forward，
    收集第一个 latent 位置（pass_idx=0）的 hidden state。

    Returns:
        latents: List[Tensor]，每个元素 shape (hidden_size,)
        idxs:    List[int]，对应的样本 idx
    """
    latents = []
    idxs = []

    for batch in dataloader:
        test_idx = batch["idx"][0].item()
        input_ids     = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        position_ids  = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).reshape(1, -1)
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


@torch.no_grad()
def evaluate_mode(model, tokenizer, dataloader, answers_val, device,
                  latent_pool=None, mode="baseline"):
    """
    mode:
      "baseline"   — 不替换，正常 generate
      "random"     — 第一个 latent 替换为 N(0,1) 随机向量（每条数据独立采样）
      "other"      — 第一个 latent 替换为数据集内另一条数据的 latent
    latent_pool: List[Tensor|None]，collect_first_latents 的结果（mode=="other" 时必须提供）
    """
    hidden_size = model.embedding.embedding_dim
    correct, total = 0, 0
    all_pool = [v for v in latent_pool if v is not None] if latent_pool else []

    for sample_i, batch in enumerate(dataloader):
        test_idx = batch["idx"][0].item()
        answer   = answers_val[test_idx]
        input_ids = batch["input_ids"].to(device)

        # 确定替换向量
        if mode == "baseline":
            replacement = None
        elif mode == "random":
            replacement = torch.randn(hidden_size, device=device)
        elif mode == "other":
            # 从 pool 中随机选一个不是自己的 latent
            own_latent = latent_pool[sample_i] if latent_pool else None
            candidates = [
                v for j, v in enumerate(latent_pool)
                if j != sample_i and v is not None
            ]
            if candidates:
                replacement = random.choice(candidates).to(device)
            else:
                replacement = torch.randn(hidden_size, device=device)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="models/prosqa-coconut/checkpoint_38")
    parser.add_argument("--val_path",    default="data/prosqa_test.json")
    parser.add_argument("--stage",       type=int, default=6,
                        help="评估时使用的 latent stage 数（即注入多少个 latent token）")
    parser.add_argument("--output_json", default="results/latent_replacement.json")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
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

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collator
    )

    # ---- Step 1: 收集所有样本的第一个 latent hidden state ----
    print(f"\nCollecting first-latent hidden states (stage={args.stage}) ...")
    latent_pool, sample_idxs = collect_first_latents(model, dataloader, device)
    valid_count = sum(1 for v in latent_pool if v is not None)
    print(f"  Collected {valid_count}/{len(latent_pool)} valid latents")

    # ---- Step 2: 三种模式分别评估 ----
    results = {}
    for mode in ("baseline", "random", "other"):
        print(f"\n--- Mode: {mode} ---")
        acc, correct, total = evaluate_mode(
            model, tokenizer, dataloader, answers_val, device,
            latent_pool=latent_pool, mode=mode,
        )
        results[mode] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  {correct}/{total} = {acc*100:.2f}%")

    # ---- 保存结果 ----
    output = {
        "checkpoint": args.checkpoint,
        "stage": args.stage,
        "results": results,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    print("\n=== Summary ===")
    for mode, r in results.items():
        print(f"  {mode:10s}: {r['accuracy']*100:.2f}%  ({r['correct']}/{r['total']})")

    baseline_acc = results["baseline"]["accuracy"]
    for mode in ("random", "other"):
        drop = (baseline_acc - results[mode]["accuracy"]) * 100
        print(f"  Accuracy drop ({mode} vs baseline): {drop:+.2f}pp")


if __name__ == "__main__":
    main()
