import itertools
from dataclasses import dataclass
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
    训练数据集。每个样本序列格式：
        [question] [start_latent] [latent × n_latent] [end_latent] [answer]

    Translator 监督目标：所有 steps 拼接后的完整推理链（+ EOS）。
    """
    n_additional_tokens = 0 if no_special_marker else 2

    def process(sample):
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent
            + ([] if no_special_marker else [end_id])
            + sample["answer_tokenized"]
        )
        # Coconut loss 只在答案部分计算
        labels = (
            [-100] * (len(sample["question_tokenized"]) + n_latent + n_additional_tokens)
            + sample["answer_tokenized"]
        )
        # 完整推理链 = 所有 steps 拼接 + EOS
        all_steps = list(itertools.chain.from_iterable(sample["steps_tokenized"]))
        translator_tokens = all_steps + [eos_id]

        return {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
            "translator_tokens": translator_tokens,
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
    评估数据集：只保留 [question][start_latent][latent×n_latent][end_latent]，
    不含答案；保留 answer / steps 字段供外层计算准确率。
    """
    def process(sample):
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent
            + ([] if no_special_marker else [end_id])
        )
        return {
            "input_ids": tokens,
            "labels": tokens,
            "attention_mask": [1] * len(tokens),
            "idx": sample.get("idx", 0),
            "position_ids": list(range(len(tokens))),
        }

    # 保留 answer / steps 供外部访问；其余原始列全部移除
    columns_to_remove = [
        c for c in base_dataset.features.keys() if c not in ["answer", "steps"]
    ]
    return base_dataset.map(process, remove_columns=columns_to_remove, num_proc=4)


@dataclass
class DirectLatentCollator:
    """
    Batch collator for DirectLatentModel.

    和 Coconut 原版 collator 一样，通过左填充对齐第一个 latent 位置，
    以最大化 KV cache 复用。
    额外处理 translator_tokens → (batch, max_trans_len) 张量。
    """
    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):
        assert self.tokenizer.padding_side == "right"

        # KV cache 对齐：左填充到最晚的 first_latent 位置
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]
        if earliest_latent:
            latest_earliest = max(earliest_latent)
            pad_tok = self.tokenizer.pad_token_id
            for f in features:
                n_pad = latest_earliest - (
                    f["input_ids"].index(self.latent_id)
                    if self.latent_id in f["input_ids"]
                    else 0
                )
                f["position_ids"] = [0] * n_pad + list(range(len(f["input_ids"])))
                f["input_ids"] = [pad_tok] * n_pad + f["input_ids"]
                if "labels" in f:
                    f["labels"] = [self.label_pad_token_id] * n_pad + f["labels"]
                f["attention_mask"] = [0] * n_pad + f["attention_mask"]

        # translator_tokens → (batch, max_trans_len) + mask
        has_trans = "translator_tokens" in features[0]
        batch_translator_labels = None
        translator_labels_mask = None

        if has_trans:
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            max_len = max(max(len(f["translator_tokens"]) for f in features), 2)
            b = len(features)
            batch_translator_labels = torch.full((b, max_len), pad_id, dtype=torch.long)
            translator_labels_mask = torch.zeros((b, max_len), dtype=torch.long)
            for i, f in enumerate(features):
                toks = f["translator_tokens"]
                batch_translator_labels[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)
                translator_labels_mask[i, :len(toks)] = 1

        # 常规 padding（input_ids、attention_mask 等）
        return_tensors = "pt"
        label_name = "label" if "label" in features[0] else "labels"
        non_label_pos_features = [
            {k: v for k, v in f.items() if k not in [label_name, "position_ids", "translator_tokens"]}
            for f in features
        ]
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, non_label_pos_features, padding=True, return_tensors=return_tensors
        )

        for key in [label_name, "position_ids"]:
            if key in features[0]:
                data = [f[key] for f in features]
                max_l = max(len(d) for d in data)
                pad_val = self.label_pad_token_id if key == label_name else 0
                batch[key] = torch.tensor(
                    [d + [pad_val] * (max_l - len(d)) for d in data], dtype=torch.int64
                )

        if batch_translator_labels is not None:
            batch["translator_labels"] = batch_translator_labels
            batch["translator_labels_mask"] = translator_labels_mask

        return batch
