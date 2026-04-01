"""
test_lmhead_decode.py

测试 Coconut.translate_latents_lmhead()：
  用模型自身的 lm_head 对 latent hidden state 解码，看 latent 是否编码了可读的推理步骤。

权重：models/prosqa-coconut/checkpoint_42  (stage6, c_thought=1, GPT-2)
数据：data/prosqa_valid.json               (ProntoQA 验证集)
"""

import json
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from coconut import Coconut
from dataset import get_dataset


# ---------------------------------------------------------------------------
# 配置（与 args/prosqa_coconut.yaml 一致）
# ---------------------------------------------------------------------------
CHECKPOINT  = "models/prosqa-coconut/checkpoint_42"
MODEL_ID    = "openai-community/gpt2"
VAL_PATH    = "data/prosqa_valid.json"
C_THOUGHT   = 1
MAX_STAGE   = 6
NUM_SAMPLES = 20     # 评估样本数
MAX_DECODE  = 40     # 每步最多解码 token 数
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 初始化 Tokenizer 与模型
# ---------------------------------------------------------------------------
def build_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = ["<latent>", "<|start-latent|>", "<|end-latent|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # 加载 base GPT-2 并扩容词表（50257 → 50260）
    base_model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    base_model.resize_token_embeddings(len(tokenizer))

    # 初始化 Coconut（添加 c_thought 参数）
    model = Coconut(
        base_causallm=base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        c_thought=C_THOUGHT,
    )

    # 加载 checkpoint（直接是 state_dict，键带 base_causallm. 前缀）
    print(f"Loading checkpoint: {CHECKPOINT}")
    state_dict = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)

    print(f"Model loaded. vocab_size={len(tokenizer)}, c_thought={C_THOUGHT}, stage={MAX_STAGE}")
    return model, tokenizer, latent_id, start_id, end_id


# ---------------------------------------------------------------------------
# 构造评估输入序列：[question] <|start-latent|> <latent>×n_stage <|end-latent|>
# ---------------------------------------------------------------------------
def build_latent_input(sample, tokenizer, latent_id, start_id, end_id, n_stage):
    """
    用实际 step 数与 n_stage 的较小值构造 latent 序列，与训练时一致。
    返回 (input_ids tensor, gt_steps list)
    """
    q_ids   = sample["question_tokenized"]
    n_steps = min(n_stage, len(sample["steps_tokenized"]))
    tokens  = q_ids + [start_id] + [latent_id] * (n_steps * C_THOUGHT) + [end_id]
    gt_steps = [sample["steps"][i] for i in range(n_steps)]
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
    return input_ids, gt_steps


# ---------------------------------------------------------------------------
# 主测试逻辑
# ---------------------------------------------------------------------------
def main():
    model, tokenizer, latent_id, start_id, end_id = build_model()

    # 加载验证集（使用 get_dataset 与训练代码保持一致）
    dataset = get_dataset(VAL_PATH, tokenizer)
    num_eval = min(NUM_SAMPLES, len(dataset))

    correct_thoughts = 0
    total_thoughts   = 0
    correct_answers  = 0

    print(f"\n{'='*60}")
    print(f"Testing lm_head decode on {num_eval} samples (stage={MAX_STAGE})")
    print(f"{'='*60}\n")

    for i in range(num_eval):
        sample = dataset[i]
        input_ids, gt_steps = build_latent_input(
            sample, tokenizer, latent_id, start_id, end_id, n_stage=MAX_STAGE
        )

        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            # 用 lm_head 解码 latent 向量
            decoded_thoughts = model.translate_latents_lmhead(
                input_ids=input_ids,
                tokenizer=tokenizer,
                max_new_tokens=MAX_DECODE,
            )

            # 同时用 generate 获取最终答案
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=64,
            )

        gt_answer   = str(sample["answer"]).strip()
        gen_text    = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        pure_answer = gen_text.split("#")[-1].replace(",", "").strip()

        ans_correct = (gt_answer == pure_answer)
        if ans_correct:
            correct_answers += 1

        # 打印前 5 条的详细结果
        if i < 5:
            question = tokenizer.decode(
                torch.tensor(sample["question_tokenized"]), skip_special_tokens=True
            )
            print(f"[Sample {i+1}]")
            print(f"  Question   : {question}")
            print(f"  GT Answer  : {gt_answer}")
            print(f"  Gen Answer : {pure_answer}  ({'✓' if ans_correct else '✗'})")
            for step_idx, (gt, decoded) in enumerate(zip(gt_steps, decoded_thoughts)):
                gt_clean      = gt.replace(" ", "").replace("\n", "").lower()
                decoded_clean = decoded.replace(" ", "").replace("\n", "").lower()
                match = "✓" if gt_clean == decoded_clean else "✗"
                print(f"  Step {step_idx+1} GT      : {gt}")
                print(f"  Step {step_idx+1} Decoded : {decoded}  [{match}]")
            print()

        # 统计 thought 准确率
        for step_idx in range(len(gt_steps)):
            gt_clean      = gt_steps[step_idx].replace(" ", "").replace("\n", "").lower()
            decoded_clean = (decoded_thoughts[step_idx] if step_idx < len(decoded_thoughts) else "").replace(" ", "").replace("\n", "").lower()
            if gt_clean == decoded_clean:
                correct_thoughts += 1
            total_thoughts += 1

    ans_acc    = correct_answers  / num_eval
    thought_acc = correct_thoughts / total_thoughts if total_thoughts > 0 else 0.0

    print(f"{'='*60}")
    print(f"Results over {num_eval} samples:")
    print(f"  Answer Accuracy : {ans_acc*100:.1f}%  ({correct_answers}/{num_eval})")
    print(f"  Thought Accuracy: {thought_acc*100:.1f}%  ({correct_thoughts}/{total_thoughts})")
    print(f"  (Thought Accuracy = exact-match of decoded step text vs ground truth)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
