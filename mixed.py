import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple

# 保持原来的 Outputs 结构
Outputs = namedtuple("Outputs", ["loss", "coconut_loss", "translator_loss", "logits", "latent_states", "inputs_embeds"])

class CoconutWithTranslator(nn.Module):
    def __init__(
        self,
        base_causallm,
        translator,  # 传入你的新版 CoconutTranslator 实例
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        lambda_translator=0.5, # 翻译器 loss 的权重
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
        
        # 1. 识别 Latent Token 位置
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        inputs_embeds = self.embedding(input_ids)
        next_compute_range = (0, latent_indices[:, 1].min().item() if max_n_latents > 0 else input_ids.shape[1])
        
        # context_ids
        first_latent_pos = next_compute_range[1]
        context_ids_batch = input_ids[:, :first_latent_pos]
        
        kv_cache = None
        translator_losses = []
        
        # 用于记录每个样本历史累积的 latent 向量 (保留梯度) ---
        accumulated_latents = [[] for _ in range(input_ids.shape[0])]

        # 2. 多轮迭代与 Latent 提取
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
                    # 获取产生 latent 的上一个位置的 hidden state
                    latent_vec = hidden_states[b_idx : b_idx+1, token_idx - 1 - hidden_states_offset : token_idx - hidden_states_offset, :]

                    # 记录到历史累积列表中 (不 detach，保留梯度传回主模型)
                    accumulated_latents[b_idx].append(latent_vec)

                    # 收集当前轮的独立 latent 向量
                    current_pass_latents.append(latent_vec)
                    active_indices.append(b_idx)
                    filling_indices.append((b_idx, token_idx))

            # Bug1修复：用 tensor_list 方式重建 inputs_embeds，避免原地操作破坏计算图
            # 原先直接 inputs_embeds[b_idx, token_idx, :] = ... 是对计算图中张量的原地写入，
            # 会导致 autograd 梯度错误。参照 coconut.py 的 tensor_list + torch.stack 方案。
            if filling_indices:
                tensor_list = [
                    [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
                for (b_idx, token_idx), latent_vec in zip(filling_indices, current_pass_latents):
                    tensor_list[b_idx][token_idx] = latent_vec.squeeze(0).squeeze(0)
                inputs_embeds = torch.stack([
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ])

            # --- 计算该轮的翻译 Loss ---
            if current_pass_latents:
                
                last_hidden_states.append(torch.cat([v.detach().cpu() for v in current_pass_latents], dim=0))
                
                if translator_labels is not None:
                    # 只取当前 step 对应的 c_thought 个 latent，不累积历史。
                    # 累积历史会导致高 stage 时输入序列线性增长，早期 latent 与
                    # 当前步骤无关反而引入噪声，且让任务难度随 stage 递增。
                    history_latents_for_active = []
                    for b_idx in active_indices:
                        # 沿 seq 维度拼接最近的 c_thought 个 latent: (1, c_thought, hidden_size)
                        history_seq = torch.cat(accumulated_latents[b_idx][-self.c_thought:], dim=1)
                        history_latents_for_active.append(history_seq)
                    
                    # (num_active_in_batch, current_steps, hidden_size)
                    batch_history_latents = torch.cat(history_latents_for_active, dim=0)

                    # 提取翻译目标与上下文
                    target_ids_active = translator_labels[active_indices, pass_idx, :]
                    context_ids_active = context_ids_batch[active_indices]

                    # 构造 Translator 所需的输入
                    t_input_ids = target_ids_active.clone()

                    # 用基于序列长度的 mask 来构建 t_labels 和 t_attention_mask，
                    # 避免 pad_id == eos_id 时 EOS 被错误 mask 掉导致翻译器学不到停止信号。
                    if translator_labels_mask is not None:
                        t_mask_active = translator_labels_mask[active_indices, pass_idx, :]
                        t_labels = t_input_ids.clone()
                        t_labels[t_mask_active == 0] = -100
                        t_attention_mask = t_mask_active.long()
                    else:
                        # 兼容旧数据：fallback 到原来靠 token 值判断的方式
                        pad_id = self.translator.pad_id
                        t_labels = t_input_ids.clone()
                        t_labels[t_labels == pad_id] = -100
                        t_attention_mask = (t_input_ids != pad_id).long()

                    if (t_labels != -100).any():
                        # 对 latent 向量做 detach，阻止翻译器的梯度反传进 Coconut 主干。
                        # Coconut 的参数只由 coconut_loss 更新，翻译器独立收敛，
                        # 避免早期翻译器随机梯度噪声破坏主干的推理能力。
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

        # last round compute for remaining tokens after last latent
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
        
        # loss
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
        """
        工具函数：在 generate 时将每一步解译出的历史向量解译回思维文本。
        c_thought > 1 时，只在每组的最后一个 latent（即 (i+1) % c_thought == 0）
        处调用翻译器，与训练时的监督信号对齐。
        此时解码所用的 latent 序列长度依次为 c_thought, 2*c_thought, 3*c_thought, ...
        """
        decoded_thoughts = []
        cumulative_latents = []

        for i, latent_vec in enumerate(latent_states_list):
            if latent_vec is None:
                continue

            cumulative_latents.append(latent_vec.to(self.base_causallm.device))

            # 只在每组最后一个 latent 处解码
            if (i + 1) % c_thought != 0:
                continue

            current_history = torch.cat(cumulative_latents, dim=1)
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
        
        # --- 提取 context_ids 给翻译器用 ---
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