import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel


class CoconutTranslatorCA(nn.Module):
    """
    Cross-Attention 版翻译器（v4）。

    与 v3（soft-prompt 拼接）的核心区别：
    - v3: decoder 输入 = [context, <start>, latent_states, <end>, target]，latent 占据序列位置，
          和 context/target 在同一个 self-attention 空间竞争，位置编码混乱。
    - v4: decoder 输入 = [context, <start>, target]，
          latent_states 作为 encoder_hidden_states 送入 GPT-2 的独立 cross-attention 层，
          K/V 来自 latent，Q 来自 decoder 隐层，信息通路清晰，loss 收敛更快。

    接口与 v3 完全兼容（forward / translate 签名相同），可直接替换 mixed.py 中的 translator。
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 50260,
        start_id: int = None,
        end_id: int = None,
        pad_id: int = None,
        eos_id: int = None,
    ):
        super().__init__()

        self.start_id = start_id
        self.end_id = end_id
        self.pad_id = pad_id
        self.eos_id = eos_id

        cfg = GPT2Config.from_pretrained("gpt2")
        cfg.n_layer = 6
        cfg.n_head = 12
        cfg.vocab_size = vocab_size
        cfg.add_cross_attention = True   # 启用 cross-attention

        self.decoder = GPT2LMHeadModel.from_pretrained(
            "gpt2",
            config=cfg,
            ignore_mismatched_sizes=True,
        )
        self.decoder.resize_token_embeddings(vocab_size)

        # 若 Coconut 主干的 hidden_size 与 GPT-2 的 n_embd 不一致，做线性投影
        decoder_dim = cfg.n_embd  # 768 for GPT-2
        if hidden_size != decoder_dim:
            self.latent_proj = nn.Linear(hidden_size, decoder_dim, bias=False)
        else:
            self.latent_proj = None

    # ------------------------------------------------------------------
    # forward：teacher-forcing 训练
    # ------------------------------------------------------------------
    def forward(
        self,
        latent_states: torch.Tensor,        # (B, L, hidden_size)
        context_ids: torch.Tensor,           # (B, ctx_len)
        input_ids: torch.Tensor,             # (B, tgt_len)  目标 token（不含 shift）
        labels: torch.Tensor = None,         # (B, tgt_len)  已处理好 -100 mask 的标签
        attention_mask: torch.Tensor = None, # (B, tgt_len)
    ):
        batch_size = latent_states.shape[0]
        device = latent_states.device

        if latent_states.dim() == 2:
            latent_states = latent_states.unsqueeze(1)  # (B, 1, H)

        if self.latent_proj is not None:
            latent_states = self.latent_proj(latent_states)

        # --- 构造 decoder 输入序列：[context | <start> | target] ---
        context_embeds = self.decoder.transformer.wte(context_ids)        # (B, ctx, D)
        start_embed = self.decoder.transformer.wte(
            torch.tensor([[self.start_id]], device=device).expand(batch_size, 1)
        )                                                                   # (B, 1, D)
        target_embeds = self.decoder.transformer.wte(input_ids)           # (B, tgt, D)

        full_embeds = torch.cat([context_embeds, start_embed, target_embeds], dim=1)
        # shape: (B, ctx+1+tgt, D)

        # --- 注意力掩码 ---
        context_mask = (context_ids != self.pad_id).long()                # (B, ctx)
        start_mask   = torch.ones(batch_size, 1, device=device, dtype=torch.long)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        full_attention_mask = torch.cat([context_mask, start_mask, attention_mask], dim=1)

        # --- Labels：context + <start> 位置全部 -100，只对 target 计算 loss ---
        if labels is not None:
            ignore_len   = context_ids.shape[1] + 1   # context + <start>
            ignore_part  = torch.full((batch_size, ignore_len), -100, device=device)
            full_labels  = torch.cat([ignore_part, labels], dim=1)
        else:
            full_labels = None

        # --- 前向传播：latent 作为 encoder_hidden_states 进 cross-attention ---
        outputs = self.decoder(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            encoder_hidden_states=latent_states,   # K/V 来自 latent
            labels=full_labels,
        )
        return outputs.loss, outputs.logits

    # ------------------------------------------------------------------
    # translate：自回归推理（贪心解码）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def translate(
        self,
        latent_states: torch.Tensor,   # (B, L, hidden_size)
        context_ids: torch.Tensor,     # (B, ctx_len)
        max_new_tokens: int = 40,
    ) -> torch.Tensor:
        device = latent_states.device
        batch_size = latent_states.shape[0]

        if latent_states.dim() == 2:
            latent_states = latent_states.unsqueeze(1)
        if self.latent_proj is not None:
            latent_states = self.latent_proj(latent_states)

        # 初始 decoder 前缀：[context | <start>]
        context_embeds = self.decoder.transformer.wte(context_ids)
        start_embed = self.decoder.transformer.wte(
            torch.tensor([[self.start_id]], device=device).expand(batch_size, 1)
        )
        current_embeds = torch.cat([context_embeds, start_embed], dim=1)

        generated_ids = []
        for _ in range(max_new_tokens):
            outputs = self.decoder(
                inputs_embeds=current_embeds,
                encoder_hidden_states=latent_states,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(next_token)

            # 追加新 token embedding，继续自回归
            next_embed = self.decoder.transformer.wte(next_token)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

            if next_token.item() in [self.eos_id, self.end_id]:
                break

        return torch.cat(generated_ids, dim=1)   # (B, generated_len)
