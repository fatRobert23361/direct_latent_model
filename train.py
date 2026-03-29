import os
import yaml
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, GPT2LMHeadModel

from mixed import CoconutWithTranslator
from mixed_dataset import get_cot_latent_dataset, MyCollator, get_question_latent_dataset
from dataset import get_dataset
from translator_v3 import CoconutTranslator

def save_checkpoint(model, optimizer, stage, epoch, path, name):
    os.makedirs(path, exist_ok=True)
    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage": stage,
        "epoch": epoch
    }
    torch.save(save_dict, os.path.join(path, f"{name}_stage{stage}_epoch{epoch}.pt"))

def train():
    # 1. 加载配置与设备
    with open("/home/haoyang/haoyang/coconut/args/mixed_coconut.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Tokenizer 处理 (添加所有的特殊 Token)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # 注册 Coconut 和 Translator 需要的所有 Token
    special_tokens = ["<latent>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 3. 初始化 Base LLM (Coconut 的主体)
    print(f"Initializing Base Model: {cfg['model_id']}")
    base_model = GPT2LMHeadModel.from_pretrained(cfg['model_id'])
    vocab_size = len(tokenizer)

    # 4. 加载纯 Coconut 的权重
    # 4. 加载纯 Coconut 的权重
    if cfg.get("load_model_path"):
        print(f"Loading Coconut weights from {cfg['load_model_path']}")
        coconut_ckpt = torch.load(cfg["load_model_path"], map_location="cpu")
        state_dict = coconut_ckpt.get("model_state_dict", coconut_ckpt)
        
        # 清洗 base_causallm 前缀
        new_state_dict = {k.replace("base_causallm.", ""): v for k, v in state_dict.items()}

        # 动态获取词表大小并扩容
        ckpt_vocab_size = new_state_dict["transformer.wte.weight"].size(0)
        print(f"Detected vocab size from checkpoint: {ckpt_vocab_size}")
        
        # 先临时扩容到 ckpt 的大小，为了能顺利加载权重
        base_model.resize_token_embeddings(ckpt_vocab_size)

        # 剔除冗余的 embedding.weight
        final_state_dict = {}
        for k, v in new_state_dict.items():
            if k == "embedding.weight" or k.startswith("embedding."):
                continue
            final_state_dict[k] = v

        # 加载干净的权重
        missing_keys, unexpected_keys = base_model.load_state_dict(final_state_dict, strict=False)
        print(f"Successfully loaded Coconut base weights.")
        
        # --- 关键修复：加载完权重后，必须再次扩容到当前 Tokenizer 的真实大小！ ---
        actual_vocab_size = len(tokenizer)
        if ckpt_vocab_size != actual_vocab_size:
            print(f"Expanding model vocab from {ckpt_vocab_size} to {actual_vocab_size} for new special tokens.")
            base_model.resize_token_embeddings(actual_vocab_size)
            
        vocab_size = actual_vocab_size # 更新全局词表大小给 Translator 用
    else:
        # 如果从头训练，只根据 tokenizer 扩充词表
        base_model.resize_token_embeddings(vocab_size)

    # 5. 初始化 Translator (使用传入的 Token IDs)
    print(f"Initializing Translator with vocab size: {vocab_size}")
    translator = CoconutTranslator(
        hidden_size=base_model.config.n_embd,
        vocab_size=vocab_size, 
        start_id=start_id,
        end_id=end_id,
        pad_id=pad_id,
        eos_id=tokenizer.eos_token_id,
        mode="context_latent"
    )
    
    # 6. 加载预训练 Translator (如果有)
    if cfg.get("load_translator_path"):
        print(f"Loading Translator weights from {cfg['load_translator_path']}")
        t_ckpt = torch.load(cfg["load_translator_path"], map_location="cpu")
        t_state_dict = t_ckpt.get("model_state_dict", t_ckpt)
        
        # 同样进行清洗
        t_final_dict = {k.replace("decoder.", ""): v for k, v in t_state_dict.items() 
                        if k != "embedding.weight" and not k.startswith("embedding.")}
                        
        translator.decoder.resize_token_embeddings(vocab_size)
        translator.decoder.load_state_dict(t_final_dict, strict=False)
        
    # 7. 组装最终的混合模型
    model = CoconutWithTranslator(
        base_causallm=base_model,
        translator=translator,
        latent_token_id=latent_id,
        start_latent_id=start_id, 
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        lambda_translator=cfg.get("lambda_translator", 0.5),
        c_thought=cfg.get("c_thought", 1),
    ).to(device).to(torch.bfloat16)
    
    # Bug3修复：checkpoint 只加载一次，同时用于恢复模型权重和优化器状态。
    # 原先加载了两次同一个文件（第一次 map_location="cpu"，第二次 map_location=device），
    # 造成冗余 I/O，且两次加载语义不一致。
    resume_ckpt = None
    if cfg.get("resume_from_checkpoint"):
        print(f"Resuming training from checkpoint: {cfg['resume_from_checkpoint']}")
        resume_ckpt = torch.load(cfg["resume_from_checkpoint"], map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        print(f"Resumed model weights from checkpoint.")

    # 8. 加载原始数据
    print("Loading raw datasets...")
    raw_train = get_dataset(cfg["train_path"], tokenizer)
    raw_val = get_dataset(cfg["val_path"], tokenizer)

    # 9. 优化器：coconut 主干与 translator 使用独立的 param group，
    # 允许为 translator 设置更高的 lr，且 stage 切换时可以只重置 translator 的 Adam state。
    translator_lr = cfg.get("translator_lr", cfg["lr"] * 5)
    optimizer = AdamW([
        {"params": list(model.base_causallm.parameters()), "lr": cfg["lr"]},
        {"params": list(model.translator.parameters()), "lr": translator_lr, "name": "translator"},
    ], weight_decay=cfg["weight_decay"])

    if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        print(f"Resumed optimizer state from checkpoint.")

    # 10. 初始化 WandB
    wandb.init(project=cfg["project"], name=cfg["name"], config=cfg)

    global_step = 0

    start_stage = cfg.get("resume_stage", 1)
    start_epoch = cfg.get("resume_epoch", 0)
    
    # 11. 主训练循环：分阶段 (Curriculum Learning)
    warmup_steps = cfg.get("warmup_steps_per_stage", 100)

    for stage in range(start_stage, cfg["max_latent_stage"] + 1):
        print(f"\n>>> Starting Stage {stage} (Adding {stage} latent thought steps)")

        # 问题2b修复：动态 lambda_translator，随 stage 线性递增。
        # 早期 stage latent 质量差，翻译器信号噪声大，给低权重；
        # 后期 stage latent 表示成熟，再逐步加大翻译器监督比例。
        current_lambda = cfg.get("lambda_translator", 0.5) * (stage / cfg["max_latent_stage"])
        model.lambda_translator = current_lambda
        print(f"lambda_translator = {current_lambda:.3f}")

        # 每个 stage 开始时重置 LR 并启动 warmup 调度器。
        # 同时重置 translator param group 的 Adam 动量状态：
        # latent 分布在 stage 切换时发生突变，旧的动量方向会误导 translator 更新。
        translator_param_set = set(model.translator.parameters())
        for p in list(translator_param_set):
            if p in optimizer.state:
                del optimizer.state[p]
        for pg in optimizer.param_groups:
            if pg.get("name") == "translator":
                pg["lr"] = translator_lr
            else:
                pg["lr"] = cfg["lr"]
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps))
        )

        if stage == cfg["max_latent_stage"] and cfg.get("epochs_for_final_stage"):
            target_epochs = cfg["epochs_for_final_stage"]
        else:
            target_epochs = cfg["epochs_per_stage"]

        train_ds = get_cot_latent_dataset(
            scheduled_stage=stage, 
            base_dataset=raw_train, 
            configs=type('obj', (object,), cfg), 
            start_id=start_id, 
            latent_id=latent_id, 
            end_id=end_id,
            shuffle=True,
            eos_id=tokenizer.eos_token_id
        )
        
        collator = MyCollator(tokenizer=tokenizer, latent_id=latent_id)
        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg["batch_size_training"], 
            shuffle=True, 
            collate_fn=collator
        )
        first_epoch = start_epoch if stage == start_stage else 0
        for epoch in range(first_epoch, target_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Stage {stage} | Epoch {epoch}/{target_epochs-1}")
            
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
                loss = outputs.loss / cfg["gradient_accumulation_steps"]
                loss.backward()

                if (global_step + 1) % cfg["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()  # 问题3修复：warmup 调度
                    optimizer.zero_grad()

                if global_step % 10 == 0:
                    wandb.log({
                        "train/total_loss": outputs.loss.item(),
                        "train/coconut_loss": outputs.coconut_loss.item(),
                        "train/translator_loss": outputs.translator_loss.item(),
                        "meta/stage": stage,
                        "meta/epoch": epoch,
                        "meta/step": global_step,
                        "meta/lr": optimizer.param_groups[0]['lr'],
                        "meta/lambda_translator": model.lambda_translator,
                    })
                
                pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})
                global_step += 1

            # --- 每个 Epoch 跑一次验证 ---
            print(f"\nRunning Evaluation for Stage {stage}, Epoch {epoch}...")
            evaluate_and_log_wandb(model, raw_val, tokenizer, stage, epoch, device, cfg, latent_id, start_id, end_id)
            
            # 保存混合模型的 Checkpoint
            save_checkpoint(model, optimizer, stage, epoch, cfg["save_path"], cfg["name"])

    wandb.finish()

@torch.no_grad()
def evaluate_and_log_wandb(model, raw_val, tokenizer, stage, epoch, device, cfg, latent_id, start_id, end_id):
    model.eval()
    
    # 选定评估集大小 (比如评估 300 条来算准确率)
    num_eval = min(cfg.get("num_eval_samples", 300), len(raw_val))
    eval_raw = raw_val.select(range(num_eval))
    
    # ==========================================
    # 1. 计算 Validation Loss (使用完整序列)
    # ==========================================
    # eval 的 val loss 必须禁用 uniform_prob，否则每次采样不同 stage 导致 loss 不可重复
    eval_loss_ds = get_cot_latent_dataset(
        scheduled_stage=stage,
        base_dataset=eval_raw,
        configs=type('obj', (object,), {**cfg, 'uniform_prob': 0.0}),
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        shuffle=False,
        eos_id=tokenizer.eos_token_id
    )
    
    collator = MyCollator(tokenizer=tokenizer, latent_id=latent_id)
    val_loss_loader = DataLoader(eval_loss_ds, batch_size=cfg["batch_size_training"], collate_fn=collator)
    
    total_val_loss, total_val_coconut, total_val_trans = 0.0, 0.0, 0.0
    
    print(f"\n--- Calculating Validation Loss ---")
    for batch in tqdm(val_loss_loader, desc=f"Stage {stage} Val Loss"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
        total_val_loss += outputs.loss.item()
        total_val_coconut += outputs.coconut_loss.item()
        total_val_trans += outputs.translator_loss.item()
        
    avg_val_loss = total_val_loss / len(val_loss_loader)
    avg_val_coconut = total_val_coconut / len(val_loss_loader)
    avg_val_trans = total_val_trans / len(val_loss_loader)
    print(f"Validation Loss: {avg_val_loss:.4f} (Coconut: {avg_val_coconut:.4f}, Translator: {avg_val_trans:.4f})")

    # ==========================================
    # 2. 计算 Generation Accuracy (只使用问题 Prompt)
    # ==========================================
    # Bug2修复：原先使用了 tokenizer.bos_token_id 和 tokenizer.eos_token_id，
    # 导致 eval 序列的边界 token 与训练时不一致，准确率数字完全不可信。
    # 现在通过函数参数传入正确的 <|start-latent|> 和 <|end-latent|> ID。
    eval_gen_ds = get_question_latent_dataset(
        scheduled_stage=stage,
        base_dataset=eval_raw,
        configs=type('obj', (object,), cfg),
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id
    )
    
    table = wandb.Table(columns=[
        "Stage", "Epoch", "Question", 
        "GT_Thoughts", "Decoded_Thoughts",
        "GT_Answer", "Generated_Answer", "Answer_Match"
    ])
    
    correct_answers, correct_thoughts, total_thoughts = 0, 0, 0
    
    print(f"\n--- Running Generation Evaluation (Total: {num_eval} samples) ---")
    for i in tqdm(range(num_eval), desc=f"Evaluating Generation Stage {stage}"):
        sample_gen = eval_gen_ds[i]
        raw_sample = eval_raw[i]
        
        # 1. 获取 Ground Truth
        gt_answer = str(raw_sample["answer"]).strip()
        gt_steps = raw_sample["steps"]
        gt_thoughts_text = {idx: step.strip() for idx, step in enumerate(gt_steps[:stage])}
        
        # 2. 准备模型输入 (现在里面干干净净只有问题和 <latent>，没有任何答案泄露)
        input_ids = torch.tensor(sample_gen["input_ids"]).unsqueeze(0).to(device)
        
        # 提取 Context 给翻译器
        first_latent_pos = (input_ids[0] == latent_id).nonzero()
        first_pos = first_latent_pos[0, 0].item() if len(first_latent_pos) > 0 else input_ids.shape[1]
        context_ids = input_ids[:, :first_pos]
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 3. 跑一遍 Dummy Forward 拿取隐层向量 (不需要 labels)
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=input_ids,
                position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0).to(device)
            )

            # 解密隐藏思维
            decoded_thoughts_list = model.translate_latents(outputs.latent_states, context_ids, tokenizer, c_thought=cfg.get("c_thought", 1))

            # 正式生成回答 (调大 max_new_tokens 给模型留足空间)
            gen_ids = model.generate(input_ids, tokenizer, max_new_tokens=150, show_thoughts=False)
        decoded_thoughts_text = {idx: text.strip() for idx, text in enumerate(decoded_thoughts_list)}
        answer = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        question = tokenizer.decode(context_ids[0], skip_special_tokens=True)
        pure_answer = answer.split("#")[-1].replace(",", "").strip()  # 假设答案格式是 "Thoughts... # Final Answer"
        
        # 4. 计算匹配正确率
        answer_is_correct = gt_answer == pure_answer
        if answer_is_correct:
            correct_answers += 1
            
        for idx, gt_step in gt_thoughts_text.items():
            clean_gt_thought = gt_step.replace(" ", "").replace("\n", "").lower()
            decoded_step = decoded_thoughts_text.get(idx, "")
            clean_decoded_thought = decoded_step.replace(" ", "").replace("\n", "").lower()
            
            if clean_gt_thought and clean_gt_thought == clean_decoded_thought:
                correct_thoughts += 1
            elif not clean_gt_thought and not clean_decoded_thought:
                correct_thoughts += 1
            total_thoughts += 1

        if i < 5:
            gt_str = "\n".join([f"Thought {k+1}: {v}" for k, v in gt_thoughts_text.items()])
            decoded_str = "\n".join([f"Thought {k+1}: {v}" for k, v in decoded_thoughts_text.items()])
            table.add_data(
                stage, epoch, question, 
                gt_str, decoded_str, 
                gt_answer, answer, answer_is_correct
            )

    ans_acc = correct_answers / num_eval
    thought_acc = correct_thoughts / total_thoughts if total_thoughts > 0 else 0.0
    
    print(f"\n[Stage {stage} - Epoch {epoch} Eval] Answer Acc: {ans_acc*100:.2f}% | Thought Acc: {thought_acc*100:.2f}%")
    
    # 5. 上传所有的指标到 WandB
    wandb.log({
        "eval/loss": avg_val_loss,
        "eval/coconut_loss": avg_val_coconut,
        "eval/translator_loss": avg_val_trans,
        "eval/samples_table": table,
        "eval/answer_accuracy": ans_acc,
        "eval/thought_accuracy": thought_acc,
        "meta/stage": stage,
        "meta/epoch": epoch
    })
    
    model.train()

if __name__ == "__main__":
    train()