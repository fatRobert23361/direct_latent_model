# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple(
    "Outputs",
    ["loss", "inputs_embeds", "logits", "latent_states", "latent_contexts",
     "kv_caches_per_group", "group_end_positions"],
    defaults=[None, None],   # kv_caches_per_group 和 group_end_positions 默认为 None
)
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        c_thought=1,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.c_thought = c_thought

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        # lm_head 直接通过 self.base_causallm.lm_head 访问，
        # 不单独注册为子模块，避免与 checkpoint 的 base_causallm.lm_head.weight 冲突。

    def forward(self, input_ids, attention_mask, labels, position_ids,
                save_kv_per_group=False, **kwargs):

        logits = []
        last_hidden_states = []
        kv_caches_per_group = []   # save_kv_per_group=True 时填充
        group_end_positions = []   # 每组最后一个 latent 的 position index
        # latent_contexts = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states
            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values
            # save the last hidden states for the current pass
            # --- 修改开始 ---
            # 不要直接存下整个 hidden_states，它太大了。
            # 我们只需要在该 pass 中，对应被反馈位置的前一个向量。
            # 对于 batch_size=1 且 stage 提取的情况：
            # 在 pass_idx 轮，我们反馈的是 token_idx - 1 位置的向量。
            
            current_pass_latents = []
            # current_pass_contexts = []
            for instance_idx, mask_list in enumerate(latent_lists):
                if len(mask_list) > pass_idx:
                    token_idx = mask_list[pass_idx]
                    # 提取具体的向量：(batch_idx, seq_idx_in_current_pass, hidden_size)
                    # 注意减去 offset
                    latent_vec = hidden_states[
                        instance_idx:instance_idx+1, 
                        token_idx - 1 - hidden_states_offset : token_idx - hidden_states_offset, 
                        :
                    ]
                    # current_pass_contexts.append(input_ids[instance_idx, :token_idx].detach().cpu())
                    current_pass_latents.append(latent_vec.detach().cpu())
            
            # 如果该 batch 中有人产生了 latent，保存这些具体的向量
            if current_pass_latents:
                # 拼接成 (num_samples_with_latent, 1, 768)
                last_hidden_states.append(torch.cat(current_pass_latents, dim=0))

                # 每组最后一个 latent pass 时，保存 KV cache 快照供 lm_head 解码使用。
                # latent_pos 使用 next_compute_range[1]-1，对 c_thought=1 的 pass_idx=0
                # 也能正确定位到前缀末尾（prefix 最后一个位置），避免原始实现中
                # next_compute_range[0]=0 的 off-by-one 问题。
                if save_kv_per_group and (pass_idx + 1) % self.c_thought == 0:
                    kv_caches_per_group.append(kv_cache)
                    group_end_positions.append(next_compute_range[1] - 1)
                # latent_contexts.append(current_pass_contexts)
            else:
                last_hidden_states.append(None) # 占位
                # latent_contexts.append(None)
            # --- 修改结束 ---
            # last_hidden_states.append(hidden_states.detach().cpu())
            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            latent_states=last_hidden_states,
            latent_contexts=None,
            kv_caches_per_group=kv_caches_per_group if save_kv_per_group else None,
            group_end_positions=group_end_positions if save_kv_per_group else None,
        )

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    @torch.no_grad()
    def decode_step_with_lmhead(self, hidden_state, kv_cache, latent_end_pos, max_new_tokens=40):
        """
        从单个 latent 位置的 hidden state 出发，用 lm_head 贪婪解码。

        Args:
            hidden_state:    该 latent 的 hidden state，shape (1, 1, hidden_size)
                             GPT-2 的 hidden_states[-1] 已过 ln_f，可直接接 lm_head。
            kv_cache:        该 latent pass 结束时的 KV cache 快照。
            latent_end_pos:  KV cache 中最后一个有效位置的 position index。
                             续写的第一个 token 从 latent_end_pos+1 开始编号。
            max_new_tokens:  最多生成的 token 数。

        Returns:
            List[int]  生成的 token id 序列（不含 EOS）。
        """
        device = hidden_state.device

        # 用 lm_head 对 latent hidden state 投影得到第一个 token 的 logits
        first_logits = self.base_causallm.lm_head(hidden_state)          # (1, 1, vocab_size)
        next_token = torch.argmax(first_logits[0, -1]).item()

        generated = []
        if next_token == self.eos_token_id:
            return generated
        generated.append(next_token)

        current_pos = latent_end_pos + 1
        current_kv  = kv_cache

        for _ in range(max_new_tokens - 1):
            token_embed = self.embedding(torch.tensor([[next_token]], device=device))
            out = self.base_causallm(
                inputs_embeds=token_embed,
                past_key_values=current_kv,
                position_ids=torch.tensor([[current_pos]], device=device),
                use_cache=True,
            )
            current_kv  = out.past_key_values
            current_pos += 1
            next_token   = torch.argmax(out.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            generated.append(next_token)

        return generated

    @torch.no_grad()
    def translate_latents_lmhead(self, input_ids, tokenizer, max_new_tokens=40):
        """
        完整评估入口：运行一次 forward（含 KV cache 快照收集），
        对每组 c_thought 个 latent 用 lm_head 自回归解码，返回解码文本列表。

        仅支持 batch_size=1。

        Args:
            input_ids:      shape (1, seq_len)，包含 <latent> 占位符的问题序列。
            tokenizer:      用于将 token id 解码为文本。
            max_new_tokens: 每步最多生成 token 数。

        Returns:
            List[str]  长度等于 latent group 数（= n_latent_stages），
                       每个元素是该 group 对应的解码文本。
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
        kv_caches  = outputs.kv_caches_per_group
        positions  = outputs.group_end_positions
        group_idx  = 0

        for pass_idx, hidden in enumerate(outputs.latent_states):
            if hidden is None:
                continue
            # 只在每组的最后一个 pass 处解码
            if (pass_idx + 1) % self.c_thought != 0:
                continue
            if group_idx >= len(kv_caches):
                break

            kv  = kv_caches[group_idx]
            pos = positions[group_idx]
            group_idx += 1

            # hidden: (1, 1, hidden_size)，直接传入，不额外 unsqueeze
            token_ids = self.decode_step_with_lmhead(
                hidden_state=hidden.to(device),
                kv_cache=kv,
                latent_end_pos=pos,
                max_new_tokens=max_new_tokens,
            )
            decoded_thoughts.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip())

        return decoded_thoughts

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)
