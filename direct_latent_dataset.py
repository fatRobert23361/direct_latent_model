import itertools
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


def get_direct_latent_train_dataset(
    base_dataset,
    n_latent,
    start_id,
    latent_id,
    end_id,
    eos_id,
    no_special_marker=False,
    shuffle=False,
):
    """
    训练数据集。Dataset 本身始终输出无 COT 的基础格式：
        [question][<bot>][latent×n][<eot>][answer]

    同时携带 cot_tokens（完整推理链 token 列表）和 n_prefix（前缀长度），
    供 DirectLatentCollator 在组 batch 时按 p_mask 概率插入 COT。

    这样 dataset 只构建一次，COT 插入的随机决策在每个 batch 实时进行，
    避免每个 epoch 重新 map 整个数据集。
    """
    n_additional = 0 if no_special_marker else 2

    def process(sample):
        cot_tokens = list(itertools.chain.from_iterable(sample["steps_tokenized"]))

        # 基础序列（无 COT）
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent
            + ([] if no_special_marker else [end_id])
            + sample["answer_tokenized"]
        )
        # prefix 长度 = question + <bot> + latents + <eot>
        n_prefix = len(sample["question_tokenized"]) + n_latent + n_additional
        # 无 COT 时的 labels
        labels = [-100] * n_prefix + sample["answer_tokenized"]

        return {
            "input_ids":         tokens,
            "labels":            labels,
            "attention_mask":    [1] * len(tokens),
            "idx":               sample["idx"],
            "position_ids":      list(range(len(tokens))),
            # COT 信息：由 collator 在 batch 时按 p_mask 决定是否插入
            "cot_tokens":        cot_tokens,
            "n_prefix":          n_prefix,
            # translator 目标：始终是完整 COT，不受 p_mask 影响
            "translator_tokens": cot_tokens + [eos_id],
        }

    dataset = base_dataset.map(process, remove_columns=list(base_dataset.features), num_proc=4)
    if shuffle:
        dataset = dataset.shuffle()
    return dataset


def get_direct_latent_eval_dataset(
    base_dataset,
    n_latent,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):
    """
    评估数据集（推理格式）：只返回 [question][<bot>][latent×n][<eot>]，不含答案。
    保留 answer / steps 字段供外层计算准确率。
    """
    def process(sample):
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent
            + ([] if no_special_marker else [end_id])
        )
        return {
            "input_ids":      tokens,
            "labels":         tokens,
            "attention_mask": [1] * len(tokens),
            "idx":            sample.get("idx", 0),
            "position_ids":   list(range(len(tokens))),
        }

    columns_to_remove = [
        c for c in base_dataset.features.keys() if c not in ["answer", "steps"]
    ]
    return base_dataset.map(process, remove_columns=columns_to_remove, num_proc=4)


@dataclass
class DirectLatentCollator:
    """
    Batch collator for DirectLatentModel.

    p_mask（0→1 随训练进度线性增大）：
      - p_mask = 0：所有样本都带完整 COT（backbone 监督信号最丰富，最容易学）
      - p_mask = 1：所有样本都不含 COT（与推理格式一致）
      - 中间值：随机混合两种格式

    每个 epoch 训练前在外部更新 collator.p_mask 即可，dataset 不需要重建。
    """
    tokenizer:           PreTrainedTokenizerBase
    latent_id:           Optional[int] = None
    label_pad_token_id:  Optional[int] = -100
    p_mask:              float = 0.0   # 由训练脚本逐 epoch 更新

    def __call__(self, features, return_tensors=None):
        assert self.tokenizer.padding_side == "right"

        # ------------------------------------------------------------------
        # 1. 按 p_mask 决定每条样本是否插入 COT
        #    插入点：prefix（question+bot+latents+eot）之后、answer 之前
        # ------------------------------------------------------------------
        for f in features:
            if "cot_tokens" in f and random.random() >= self.p_mask:
                # 插入 COT
                cot  = f["cot_tokens"]
                n_pre = f["n_prefix"]
                answer_toks = f["input_ids"][n_pre:]   # 原始 answer 部分

                f["input_ids"]      = f["input_ids"][:n_pre] + cot + answer_toks
                f["labels"]         = [-100] * n_pre + cot + answer_toks
                f["attention_mask"] = [1] * len(f["input_ids"])
                f["position_ids"]   = list(range(len(f["input_ids"])))

        # ------------------------------------------------------------------
        # 2. KV cache 对齐：左填充让所有样本的第一个 latent 对齐
        # ------------------------------------------------------------------
        earliest_latent = [
            f["input_ids"].index(self.latent_id)
            for f in features
            if self.latent_id in f["input_ids"]
        ]
        if earliest_latent:
            latest_earliest = max(earliest_latent)
            pad_tok = self.tokenizer.pad_token_id
            for f in features:
                n_pad = latest_earliest - (
                    f["input_ids"].index(self.latent_id)
                    if self.latent_id in f["input_ids"] else 0
                )
                f["position_ids"]   = [0] * n_pad + list(range(len(f["input_ids"])))
                f["input_ids"]      = [pad_tok] * n_pad + f["input_ids"]
                f["labels"]         = [self.label_pad_token_id] * n_pad + f["labels"]
                f["attention_mask"] = [0] * n_pad + f["attention_mask"]

        # ------------------------------------------------------------------
        # 3. translator_tokens → (batch, max_trans_len) + mask
        # ------------------------------------------------------------------
        has_trans = "translator_tokens" in features[0]
        batch_translator_labels = None
        translator_labels_mask  = None

        if has_trans:
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            max_len = max(max(len(f["translator_tokens"]) for f in features), 2)
            b = len(features)
            batch_translator_labels = torch.full((b, max_len), pad_id, dtype=torch.long)
            translator_labels_mask  = torch.zeros((b, max_len), dtype=torch.long)
            for i, f in enumerate(features):
                toks = f["translator_tokens"]
                batch_translator_labels[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
                translator_labels_mask[i, :len(toks)]  = 1

        # ------------------------------------------------------------------
        # 4. 标准 padding（排除辅助字段）
        # ------------------------------------------------------------------
        return_tensors = "pt"
        label_name = "label" if "label" in features[0] else "labels"
        skip_keys  = {label_name, "position_ids", "translator_tokens", "cot_tokens", "n_prefix"}
        non_label_pos_features = [
            {k: v for k, v in f.items() if k not in skip_keys}
            for f in features
        ]
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, non_label_pos_features, padding=True, return_tensors=return_tensors
        )

        for key in [label_name, "position_ids"]:
            if key in features[0]:
                data    = [f[key] for f in features]
                max_l   = max(len(d) for d in data)
                pad_val = self.label_pad_token_id if key == label_name else 0
                batch[key] = torch.tensor(
                    [d + [pad_val] * (max_l - len(d)) for d in data], dtype=torch.int64
                )

        if batch_translator_labels is not None:
            batch["translator_labels"]      = batch_translator_labels
            batch["translator_labels_mask"] = translator_labels_mask

        return batch
