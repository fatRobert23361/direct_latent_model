import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple

Outputs = namedtuple("Outputs", ["loss", "coconut_loss", "translator_loss", "logits", "latent_states", "inputs_embeds"])


class DirectLatentModel(nn.Module):
    """
    与 CoconutWithTranslator (mixed.py) 的核心区别：

    1. 不使用多阶段课程学习：n_latent 固定，不随 stage 递增
    2. Translator 将全部 n_latent 个向量一次性翻译为完整推理链
       （目标文本 = 所有 steps 拼接，而非 mixed.py 里逐步预测）
    3. Translator 的梯度直接传回 backbone（不 detach），实现端到端联合优化
    """

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        n_latent=6,
        translator=None,
        lambda_translator=0.5,
    ):
        super().__init__()
        self.base_causallm = base_causallm
        self.translator = translator
        self.n_latent = n_latent
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.lambda_translator = lambda_translator

        if hasattr(self.base_causallm, "transformer"):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        translator_labels=None,
        translator_labels_mask=None,
        **kwargs,
    ):
        logits = []

        # 1. 找到所有 latent token 位置
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max(len(l) for l in latent_lists) if latent_lists else 0

        inputs_embeds = self.embedding(input_ids)
        next_compute_range = (
            0,
            latent_indices[:, 1].min().item() if max_n_latents > 0 else input_ids.shape[1],
        )

        first_latent_pos = next_compute_range[1]
        context_ids_batch = input_ids[:, :first_latent_pos]

        kv_cache = None
        # 每个样本累积的 latent 向量列表，每项形状 (1, 1, hidden_size)
        all_latent_vecs = [[] for _ in range(input_ids.shape[0])]

        # 2. 多轮迭代：依次将每个 latent 位置替换为对应 hidden state
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.base_causallm(
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
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            filling_indices = []
            new_latent_vecs = []

            for b_idx, mask_list in enumerate(latent_lists):
                if len(mask_list) > pass_idx:
                    token_idx = mask_list[pass_idx]
                    latent_vec = hidden_states[
                        b_idx:b_idx + 1,
                        token_idx - 1 - hidden_states_offset:token_idx - hidden_states_offset,
                        :,
                    ]
                    # 关键：不 detach，梯度可从 translator_loss 流回 backbone
                    all_latent_vecs[b_idx].append(latent_vec)
                    filling_indices.append((b_idx, token_idx))
                    new_latent_vecs.append(latent_vec)

            if filling_indices:
                batch_indices = torch.tensor(
                    [b for b, _ in filling_indices], device=inputs_embeds.device
                )
                token_indices = torch.tensor(
                    [t for _, t in filling_indices], device=inputs_embeds.device
                )
                new_values = torch.stack([v.squeeze(0).squeeze(0) for v in new_latent_vecs])
                inputs_embeds = inputs_embeds.index_put((batch_indices, token_indices), new_values)

            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1,
            )

        # 3. 最后一轮：处理 latent 区域之后的全部 tokens
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=(
                [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                if kv_cache else None
            ),
            output_hidden_states=True,
        )
        logits.append(outputs.logits)

        # 4. Coconut Loss：只在答案部分计算
        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        coconut_loss = CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # 5. Translator Loss：全部 latent 向量 → 完整推理链文本
        translator_loss = torch.tensor(0.0, device=input_ids.device)

        if self.translator is not None and translator_labels is not None:
            active_indices = [
                b for b in range(input_ids.shape[0]) if len(all_latent_vecs[b]) > 0
            ]

            if active_indices:
                # (num_active, n_latent, hidden_size)
                latent_seqs = torch.cat(
                    [torch.cat(all_latent_vecs[b], dim=1) for b in active_indices],
                    dim=0,
                )
                context_ids_active = context_ids_batch[active_indices]
                t_input_ids = translator_labels[active_indices]

                if translator_labels_mask is not None:
                    t_mask = translator_labels_mask[active_indices]
                    t_labels = t_input_ids.clone()
                    t_labels[t_mask == 0] = -100
                    t_attention_mask = t_mask.long()
                else:
                    pad_id = self.translator.pad_id
                    t_labels = t_input_ids.clone()
                    t_labels[t_labels == pad_id] = -100
                    t_attention_mask = (t_input_ids != pad_id).long()

                if (t_labels != -100).any():
                    # 不 detach latent_seqs —— translator_loss 的梯度流回 backbone
                    t_loss, _ = self.translator(
                        latent_states=latent_seqs,
                        context_ids=context_ids_active,
                        input_ids=t_input_ids,
                        labels=t_labels,
                        attention_mask=t_attention_mask,
                    )
                    translator_loss = t_loss

        total_loss = coconut_loss + self.lambda_translator * translator_loss

        # 构造 per-pass latent states 列表（接口兼容）
        latent_states_by_pass = []
        for pass_idx in range(max_n_latents):
            vecs = [
                all_latent_vecs[b][pass_idx]
                for b in range(input_ids.shape[0])
                if len(all_latent_vecs[b]) > pass_idx
            ]
            latent_states_by_pass.append(torch.cat(vecs, dim=0) if vecs else None)

        return Outputs(
            loss=total_loss,
            coconut_loss=coconut_loss,
            translator_loss=translator_loss,
            logits=logits,
            latent_states=latent_states_by_pass,
            inputs_embeds=inputs_embeds,
        )

    @torch.no_grad()
    def translate_latents(self, latent_states_list, context_ids, tokenizer):
        """将全部 latent 状态一次性翻译为完整推理链文本，返回单条字符串列表。"""
        if self.translator is None:
            return [""]
        valid_vecs = [v for v in latent_states_list if v is not None]
        if not valid_vecs:
            return []

        # (1, n_latent, hidden_size)
        all_latents = torch.cat(valid_vecs, dim=1)
        thought_tokens = self.translator.translate(
            latent_states=all_latents,
            context_ids=context_ids,
            max_new_tokens=100,
        )
        text = tokenizer.decode(thought_tokens[0], skip_special_tokens=True)
        return [text.strip()]

    @torch.no_grad()
    def generate(self, input_ids, tokenizer, max_new_tokens=64, show_thoughts=True, **kwargs):
        self.gen_forward_cnt = 0
        device = input_ids.device

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        first_latent_pos = (
            latent_indices[0, 1].item() if len(latent_indices) > 0 else input_ids.shape[1]
        )
        context_ids = input_ids[:, :first_latent_pos]

        dummy_labels = input_ids.clone()
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=dummy_labels,
            position_ids=torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0),
        )

        if show_thoughts and outputs.latent_states:
            print("--- 解译推理链 ---")
            thoughts = self.translate_latents(outputs.latent_states, context_ids, tokenizer)
            for t in thoughts:
                print(t)
            print("--- 推理结束，开始生成答案 ---")

        inputs_embeds = outputs.inputs_embeds
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        generated_tokens = [next_token]
        curr_embeds = inputs_embeds

        # 防止 position_id 超过 GPT-2 的 n_positions(1024) 上限触发 CUDA assert
        n_positions = getattr(self.base_causallm.config, "n_positions", 1024)
        max_steps = min(max_new_tokens - 1, n_positions - input_ids.shape[1] - 1)

        for _ in range(max(0, max_steps)):
            new_token_embed = self.embedding(torch.tensor([[next_token]], device=device))
            curr_embeds = torch.cat([curr_embeds, new_token_embed], dim=1)

            out = self.base_causallm(inputs_embeds=curr_embeds)
            self.gen_forward_cnt += 1

            next_token = torch.argmax(out.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            generated_tokens.append(next_token)

        return torch.tensor([generated_tokens], device=device)
