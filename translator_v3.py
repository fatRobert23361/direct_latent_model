import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class CoconutTranslator(nn.Module):
    def __init__(
        self, 
        hidden_size=768, 
        vocab_size=50260, 
        start_id=None, 
        end_id=None, 
        pad_id=None, 
        eos_id=None,
        mode="context_latent"
    ):
        super().__init__()
        
        # 1. 直接保存外部传进来的 Token ID
        self.start_id = start_id
        self.end_id = end_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        
        # 2. 动态配置模型
        self.config = GPT2Config.from_pretrained("gpt2")
        self.config.n_layer = 6
        self.config.n_head = 12
        self.config.vocab_size = vocab_size 
        self.config.add_cross_attention = False # 软提示不需要 cross-attention
        
        # 3. 加载预训练权重，并调整 Embedding 层大小
        self.decoder = GPT2LMHeadModel.from_pretrained(
            "gpt2", 
            config=self.config, 
            ignore_mismatched_sizes=True
        )
        self.decoder.resize_token_embeddings(vocab_size)

    def forward(self, latent_states, context_ids, input_ids, labels=None, attention_mask=None):
        batch_size = latent_states.shape[0]
        device = latent_states.device
        
        if latent_states.dim() == 2:
            latent_states = latent_states.unsqueeze(1)
            
        # 动态获取输入的 latent 长度（支持历史累积拼接）
        latent_len = latent_states.shape[1]
        
        # 1. 转换特殊 Token 为 Embedding
        context_embeds = self.decoder.transformer.wte(context_ids)
        
        # 使用 expand 快速生成 batch 形状的 start 和 end
        start_embed = self.decoder.transformer.wte(torch.tensor([[self.start_id]], device=device).expand(batch_size, 1))
        end_embed = self.decoder.transformer.wte(torch.tensor([[self.end_id]], device=device).expand(batch_size, 1))
        
        target_embeds = self.decoder.transformer.wte(input_ids)

        # 2. 拼接序列
        full_embeds = torch.cat([context_embeds, start_embed, latent_states, end_embed, target_embeds], dim=1)

        # 3. 构造 Mask 和 Labels
        # Context 掩码
        context_mask = (context_ids != self.pad_id).long()
        # Start(1) + Latents(latent_len) + End(1) = latent_len + 2
        latent_mask = torch.ones((batch_size, latent_len + 2), device=device)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        full_attention_mask = torch.cat([context_mask, latent_mask, attention_mask], dim=1)

        if labels is not None:
            # 前面的 context、start、latents、end 都设为 -100 不计算 Loss
            ignore_len = context_ids.shape[1] + latent_len + 2
            latent_labels = torch.full((batch_size, ignore_len), -100, device=device)
            full_labels = torch.cat([latent_labels, labels], dim=1)
        else:
            full_labels = None

        # 截断到 GPT-2 最大位置编码长度
        max_seq_len = 1024
        if full_embeds.shape[1] > max_seq_len:
            full_embeds         = full_embeds[:, :max_seq_len, :]
            full_attention_mask = full_attention_mask[:, :max_seq_len]
            if full_labels is not None:
                full_labels = full_labels[:, :max_seq_len]

        outputs = self.decoder(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels
        )
        return outputs.loss, outputs.logits
    
    @torch.no_grad()
    def translate(self, latent_states, context_ids, max_new_tokens=40):
        """纯 Latent 解码推理"""
        device = latent_states.device
        batch_size = latent_states.shape[0]

        if latent_states.dim() == 2:
            latent_states = latent_states.unsqueeze(1)

        # 转换前缀
        context_embeds = self.decoder.transformer.wte(context_ids)
        start_embed = self.decoder.transformer.wte(torch.tensor([[self.start_id]], device=device).expand(batch_size, 1))
        end_embed = self.decoder.transformer.wte(torch.tensor([[self.end_id]], device=device).expand(batch_size, 1))
            
        current_embeds = torch.cat([context_embeds, start_embed, latent_states, end_embed], dim=1)
        
        generated_ids = []
        
        for _ in range(max_new_tokens):
            outputs = self.decoder(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            generated_ids.append(next_token)
            
            # 转换新生成的 Token 为 Embedding 并拼接
            next_embed = self.decoder.transformer.wte(next_token)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            
            # 如果 batch=1 且遇到 EOS 就可以提前结束
            # if next_token.item() == self.eos_id:
            if next_token.item() in [self.eos_id, self.end_id]:  # 同时考虑两种可能的结束符
                break
                
        return torch.cat(generated_ids, dim=1)