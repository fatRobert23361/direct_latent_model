"""
对比实验：用纯 Coconut 模型自身的 lm_head 解码 latent hidden state。

不引入任何额外参数，测试 latent 表示中是否蕴含可被模型自身
词表投影（lm_head）直接恢复的推理步骤信息。

与 mixed.py 的对比维度：
  mixed.py                    → 额外训练 TranslatorV3（GPT-2 6层）做 latent → text 映射
  mixed_current_latent_only.py → 同上但 translator 只看当前 step 的 c_thought 个 latent
  coconut_lmhead_decode.py    → 直接用 Coconut lm_head 解码，零额外参数

解码逻辑：
  在每组 c_thought 个 latent 的最后一个 pass 结束后：
  1. 取该 latent 位置的 hidden state（已过 ln_f，可直接接 lm_head）
  2. lm_head(hidden_state) → argmax → token_0（第一个预测 token）
  3. 将 token_0 嵌入后送入模型，配合当前 KV cache 自回归续写
  4. 重复直到 EOS 或达到最大长度

注意：模型在 latent 位置被训练来预测下一个 latent 或 <|end-latent|>，
而非推理步骤文本。因此解码结果反映的是 hidden state 中的信息在
原始词表空间的投影强度，与 translator 方案形成对照。

使用方式（eval loop 中替换 model.translate_latents 调用）：
  decoder = CoconutSelfDecoder(base_causallm, ...)
  decoded = decoder.translate_latents_lmhead(input_ids, tokenizer)
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple

SelfDecodeOutputs = namedtuple(
    "SelfDecodeOutputs",
    ["loss", "logits", "latent_hidden_states", "kv_caches_per_group", "group_end_positions", "inputs_embeds"]
)


class CoconutSelfDecoder(nn.Module):
    """
    纯 Coconut 模型，无 translator。
    forward 在每组 latent 结束时额外保存 KV cache 快照和 hidden state，
    供后续 translate_latents_lmhead 用 lm_head 自回归解码。
    """

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        c_thought=1,
    ):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.c_thought = c_thought

        # 兼容 GPT-2（transformer.xxx）和 Llama（model.xxx）两种结构
        if hasattr(base_causallm, "transformer"):
            self.embedding = base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = base_causallm.get_input_embeddings()

        self.lm_head = base_causallm.lm_head

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        save_kv_per_group=False,   # eval 时设为 True 以收集 KV cache 快照
        **kwargs,
    ):
        """
        标准 Coconut forward + 可选的 per-group KV cache 快照收集。

        save_kv_per_group=True 时，在每组最后一个 latent pass 结束时保存：
          - kv_cache 快照（用于自回归续写）
          - 该 latent 位置的 hidden state（用于 lm_head 首 token 预测）
          - 该 latent 的 position index（用于续写时的 position_ids）
        """
        logits_list = []
        latent_hidden_states = []      # 每个 pass 的 latent hidden state（detach）
        kv_caches_per_group = []       # 每组最后一个 latent 的 KV cache 快照
        group_end_positions = []       # 每组最后一个 latent 的 position index

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

        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = [
                    (k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits_list.append(outputs.logits)
            # hidden_states[-1]：GPT-2 实现中最后一层输出已过 ln_f，可直接接 lm_head
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            current_pass_latents = []
            filling_indices = []

            for b_idx, mask_list in enumerate(latent_lists):
                if len(mask_list) > pass_idx:
                    token_idx = mask_list[pass_idx]
                    latent_vec = hidden_states[
                        b_idx : b_idx + 1,
                        token_idx - 1 - hidden_states_offset : token_idx - hidden_states_offset,
                        :,
                    ]
                    current_pass_latents.append(latent_vec)
                    filling_indices.append((b_idx, token_idx))

            if filling_indices:
                batch_indices = torch.tensor(
                    [b for b, _ in filling_indices], device=inputs_embeds.device
                )
                token_indices = torch.tensor(
                    [t for _, t in filling_indices], device=inputs_embeds.device
                )
                new_values = torch.stack(
                    [v.squeeze(0).squeeze(0) for v in current_pass_latents]
                )
                inputs_embeds = inputs_embeds.index_put(
                    (batch_indices, token_indices), new_values
                )

            if current_pass_latents:
                latent_hidden_states.append(
                    torch.cat([v.detach() for v in current_pass_latents], dim=0)
                )

                # 在每组最后一个 latent pass 时保存 KV cache 快照（batch=1 场景）
                if save_kv_per_group and (pass_idx + 1) % self.c_thought == 0:
                    # 当前 pass 处理的 latent token 的 position index
                    latent_pos = next_compute_range[0]  # 当前 pass 只处理这一个位置
                    kv_caches_per_group.append(kv_cache)
                    group_end_positions.append(latent_pos)
            else:
                latent_hidden_states.append(None)

            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1]
                if pass_idx + 1 >= max_n_latents
                else next_compute_range[1] + 1,
            )

        # 最后一轮：处理最后一个 latent 之后的剩余 tokens
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )
        logits_list.append(outputs.logits)

        logits = torch.cat(logits_list, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return SelfDecodeOutputs(
            loss=loss,
            logits=logits,
            latent_hidden_states=latent_hidden_states,
            kv_caches_per_group=kv_caches_per_group,
            group_end_positions=group_end_positions,
            inputs_embeds=inputs_embeds,
        )

    @torch.no_grad()
    def decode_step_with_lmhead(self, hidden_state, kv_cache, latent_end_pos, max_new_tokens=40):
        """
        从单个 latent 位置的 hidden state 出发，用 lm_head 贪婪解码。

        Args:
            hidden_state:    该 latent 的 hidden state，shape (1, 1, hidden_size)
            kv_cache:        该 latent pass 结束时的 KV cache 快照
            latent_end_pos:  该 latent token 的 position index
            max_new_tokens:  最多生成的 token 数

        Returns:
            List[int]  生成的 token id 序列（不含 EOS）
        """
        device = hidden_state.device

        # 用 lm_head 对 latent hidden state 做第一个 token 的预测
        # hidden_states[-1] 在 GPT-2 中已过 ln_f，可直接接 lm_head
        first_logits = self.lm_head(hidden_state)          # (1, 1, vocab_size)
        next_token = torch.argmax(first_logits[0, -1]).item()

        generated = []
        if next_token == self.eos_token_id:
            return generated
        generated.append(next_token)

        current_pos = latent_end_pos + 1
        current_kv = kv_cache

        for _ in range(max_new_tokens - 1):
            token_embed = self.embedding(
                torch.tensor([[next_token]], device=device)
            )
            out = self.base_causallm(
                inputs_embeds=token_embed,
                past_key_values=current_kv,
                position_ids=torch.tensor([[current_pos]], device=device),
                use_cache=True,
            )
            current_kv = out.past_key_values
            current_pos += 1
            next_token = torch.argmax(out.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            generated.append(next_token)

        return generated

    @torch.no_grad()
    def translate_latents_lmhead(self, input_ids, tokenizer, max_new_tokens=40):
        """
        完整评估入口：运行 forward（含 KV cache 快照收集），
        对每组 latent 用 lm_head 解码，返回解码文本列表。

        Args:
            input_ids:      shape (1, seq_len)，仅支持 batch_size=1
            tokenizer:      用于将 token id 转换为文本
            max_new_tokens: 每步最多生成的 token 数

        Returns:
            List[str]  每个 latent group 对应的解码文本
        """
        device = input_ids.device
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
            save_kv_per_group=True,
        )

        decoded_thoughts = []
        latent_hidden_states = outputs.latent_hidden_states  # per-pass list
        kv_caches = outputs.kv_caches_per_group
        positions = outputs.group_end_positions

        # latent_hidden_states 按 pass 排列，每组最后一个 pass 的索引是
        # (group_idx + 1) * c_thought - 1
        group_idx = 0
        for pass_idx, hidden in enumerate(latent_hidden_states):
            if hidden is None:
                continue
            # 只在每组最后一个 pass 处解码
            if (pass_idx + 1) % self.c_thought != 0:
                continue
            if group_idx >= len(kv_caches):
                break

            kv = kv_caches[group_idx]
            pos = positions[group_idx]
            group_idx += 1

            # hidden shape: (1, 1, hidden_size)
            token_ids = self.decode_step_with_lmhead(
                hidden_state=hidden.unsqueeze(1).to(device),  # (1, 1, H)
                kv_cache=kv,
                latent_end_pos=pos,
                max_new_tokens=max_new_tokens,
            )
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            decoded_thoughts.append(text.strip())

        return decoded_thoughts

    @torch.no_grad()
    def generate(self, input_ids, tokenizer, max_new_tokens=64, **kwargs):
        """标准 Coconut 生成，不做 thought 解码。"""
        device = input_ids.device
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=input_ids,
            position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
            save_kv_per_group=False,
        )

        inputs_embeds = outputs.inputs_embeds
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        generated_tokens = [next_token]
        curr_embeds = inputs_embeds

        for _ in range(max_new_tokens - 1):
            new_embed = self.embedding(torch.tensor([[next_token]], device=device))
            curr_embeds = torch.cat([curr_embeds, new_embed], dim=1)
            out = self.base_causallm(inputs_embeds=curr_embeds)
            next_token = torch.argmax(out.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            generated_tokens.append(next_token)

        return torch.tensor([generated_tokens], device=device)
