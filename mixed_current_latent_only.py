"""
方案3对比实验版本：translator 只使用当前 step 对应的 c_thought 个 latent，
不累积历史所有 latent。

与 mixed.py 的唯一区别在 forward() 的 translator loss 计算部分：
  mixed.py                    → history_seq = cat(accumulated_latents[b_idx])
  mixed_current_latent_only.py → history_seq = cat(accumulated_latents[b_idx][-self.c_thought:])

实验结论（2026-03-29）：
  该方案使 thought_accuracy 从 ~30% 骤降至 ~5%，说明历史 latent 上下文
  对 translator 解码当前步骤文本是必要的，不应丢弃。
  保留此文件仅供后续对比实验复现。
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple

Outputs = namedtuple("Outputs", ["loss", "coconut_loss", "translator_loss", "logits", "latent_states", "inputs_embeds"])

class CoconutWithTranslator(nn.Module):
    def __init__(
        self,
        base_causallm,
        translator,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        lambda_translator=0.5,
        c_thought=1,
    ):
        super().__init__()
        self.base_causallm = base_causallm
        self.translator = translator
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.lambda_translator = lambda_translator
        self.c_thought = c_thought

        if hasattr(self.base_causallm, "transformer"):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, translator_labels=None, translator_labels_mask=None, **kwargs):
        logits = []
        last_hidden_states = []

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        inputs_embeds = self.embedding(input_ids)
        next_compute_range = (0, latent_indices[:, 1].min().item() if max_n_latents > 0 else input_ids.shape[1])

        first_latent_pos = next_compute_range[1]
        context_ids_batch = input_ids[:, :first_latent_pos]

        kv_cache = None
        translator_losses = []
        accumulated_latents = [[] for _ in range(input_ids.shape[0])]

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0] : next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            current_pass_latents = []
            active_indices = []
            filling_indices = []

            for b_idx, mask_list in enumerate(latent_lists):
                if len(mask_list) > pass_idx:
                    token_idx = mask_list[pass_idx]
                    latent_vec = hidden_states[b_idx : b_idx+1, token_idx - 1 - hidden_states_offset : token_idx - hidden_states_offset, :]
                    accumulated_latents[b_idx].append(latent_vec)
                    current_pass_latents.append(latent_vec)
                    active_indices.append(b_idx)
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
                last_hidden_states.append(torch.cat([v.detach() for v in current_pass_latents], dim=0))

                if translator_labels is not None:
                    # 【方案3】只取当前 step 对应的 c_thought 个 latent，不累积历史。
                    # 注意：实验表明此方案会使 thought_accuracy 从 ~30% 骤降至 ~5%。
                    history_latents_for_active = []
                    for b_idx in active_indices:
                        history_seq = torch.cat(accumulated_latents[b_idx][-self.c_thought:], dim=1)
                        history_latents_for_active.append(history_seq)

                    batch_history_latents = torch.cat(history_latents_for_active, dim=0)

                    target_ids_active = translator_labels[active_indices, pass_idx, :]
                    context_ids_active = context_ids_batch[active_indices]
                    t_input_ids = target_ids_active.clone()

                    if translator_labels_mask is not None:
                        t_mask_active = translator_labels_mask[active_indices, pass_idx, :]
                        t_labels = t_input_ids.clone()
                        t_labels[t_mask_active == 0] = -100
                        t_attention_mask = t_mask_active.long()
                    else:
                        pad_id = self.translator.pad_id
                        t_labels = t_input_ids.clone()
                        t_labels[t_labels == pad_id] = -100
                        t_attention_mask = (t_input_ids != pad_id).long()

                    if (t_labels != -100).any():
                        t_loss, _ = self.translator(
                            latent_states=batch_history_latents.detach(),
                            context_ids=context_ids_active,
                            input_ids=t_input_ids,
                            labels=t_labels,
                            attention_mask=t_attention_mask
                        )
                        translator_losses.append(t_loss)
            else:
                last_hidden_states.append(None)

            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1
            )

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [(k[:, :, : next_compute_range[0], :], v[:, :, : next_compute_range[0], :]) for k, v in kv_cache] if kv_cache else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        coconut_loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_translator_loss = torch.stack(translator_losses).mean() if translator_losses else torch.tensor(0.0).to(input_ids.device)
        total_loss = coconut_loss + self.lambda_translator * total_translator_loss

        return Outputs(
            loss=total_loss,
            coconut_loss=coconut_loss,
            translator_loss=total_translator_loss,
            logits=logits,
            latent_states=last_hidden_states,
            inputs_embeds=inputs_embeds
        )

    @torch.no_grad()
    def translate_latents(self, latent_states_list, context_ids, tokenizer, c_thought=1):
        decoded_thoughts = []
        cumulative_latents = []

        for i, latent_vec in enumerate(latent_states_list):
            if latent_vec is None:
                continue

            cumulative_latents.append(latent_vec)

            if (i + 1) % c_thought != 0:
                continue

            # 【方案3】推理时同样只用最近 c_thought 个 latent，与训练对齐
            current_history = torch.cat(cumulative_latents[-c_thought:], dim=1)
            thought_tokens = self.translator.translate(
                latent_states=current_history,
                context_ids=context_ids,
                max_new_tokens=40
            )
            text = tokenizer.decode(thought_tokens[0], skip_special_tokens=True)
            decoded_thoughts.append(text.strip())

        return decoded_thoughts

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        tokenizer,
        max_new_tokens=64,
        show_thoughts=True,
        **kwargs
    ):
        self.gen_forward_cnt = 0
        device = input_ids.device

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        first_latent_pos = latent_indices[0, 1].item() if len(latent_indices) > 0 else input_ids.shape[1]
        context_ids = input_ids[:, :first_latent_pos]

        dummy_labels = input_ids.clone()
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=dummy_labels,
            position_ids=torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0),
        )

        if show_thoughts and outputs.latent_states:
            print("--- 正在解译隐藏思维 ---")
            thought_text = self.translate_latents(outputs.latent_states, context_ids, tokenizer, c_thought=getattr(self, 'c_thought', 1))
            print(thought_text)
            print("--- 思维结束，开始生成回答 ---")

        inputs_embeds = outputs.inputs_embeds
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        generated_tokens = [next_token]
        curr_embeds = inputs_embeds

        for _ in range(max_new_tokens - 1):
            new_token_embed = self.embedding(torch.tensor([[next_token]], device=device))
            curr_embeds = torch.cat([curr_embeds, new_token_embed], dim=1)

            out = self.base_causallm(inputs_embeds=curr_embeds)
            self.gen_forward_cnt += 1

            next_token = torch.argmax(out.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            generated_tokens.append(next_token)

        return torch.tensor([generated_tokens], device=device)
