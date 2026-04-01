"""
test_k_thoughts.py

复现论文 "k continuous thoughts" 实验：
  在推理时手动控制 latent token 数量 k ∈ {0,1,...,max_stage}，
  强制模型在完成 k 步 latent 推理后切换到语言推理，生成剩余步骤文本和最终答案。

所有 k 变体共享同一份模型权重（checkpoint_42），只有推理时的输入结构不同。

输入序列：[question] <|start-latent|> <latent>×k <|end-latent|>
生成内容：step_{k+1}\nstep_{k+2}\n...\n### answer

评估指标：
  - answer_accuracy:  最终答案与 GT 完全匹配的比例
  - step_accuracy:    生成的 step 文本与 GT 逐条完全匹配的比例（仅统计应由语言生成的部分）
  - full_step_match:  一条样本所有应生成的 steps 全部匹配的比例
"""

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from coconut import Coconut
from dataset import get_dataset, get_question_latent_dataset

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
CHECKPOINT  = "models/prosqa-coconut/checkpoint_10"  # 评估使用的模型 checkpoint
MODEL_ID    = "openai-community/gpt2"
VAL_PATH    = "data/prosqa_valid.json"
C_THOUGHT   = 1
MAX_STAGE   = 6
NUM_SAMPLES = 200     # 评估样本数
MAX_GEN_TOKENS = 200  # 每条样本最多生成的 token 数
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 模型加载（与 test_lmhead_decode.py 相同）
# ---------------------------------------------------------------------------
def build_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<latent>", "<|start-latent|>", "<|end-latent|>"]}
    )
    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base_model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    base_model.resize_token_embeddings(len(tokenizer))

    model = Coconut(
        base_causallm=base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        c_thought=C_THOUGHT,
    )
    state_dict = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    return model, tokenizer, latent_id, start_id, end_id


# ---------------------------------------------------------------------------
# 解析生成结果：从生成 token 中提取 steps 和 answer
# ---------------------------------------------------------------------------
def parse_generated(gen_text: str):
    """
    gen_text 格式：
        "step_k+1\nstep_k+2\n...\n### answer"
    返回 (steps: List[str], answer: str)
    """
    if "###" in gen_text:
        steps_part, answer_part = gen_text.split("###", 1)
        answer = answer_part.replace(",", "").strip()
    else:
        steps_part = gen_text
        answer = ""

    # 按 \n 分行，过滤空行
    steps = [s.strip() for s in steps_part.split("\n") if s.strip()]
    return steps, answer


def normalize(text: str) -> str:
    """与 train.py 的评估保持一致：去空格、去换行、小写。"""
    return text.replace(" ", "").replace("\n", "").lower()


# ---------------------------------------------------------------------------
# 单个 k 值的完整评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_k(model, tokenizer, raw_val, latent_id, start_id, end_id, k: int):
    """
    k: 使用多少个 latent token（= 多少步在隐空间推理）
    返回 dict: {answer_acc, step_acc, full_match_acc, n_lang_steps_total}
    """
    configs = type("cfg", (), {
        "max_latent_stage": MAX_STAGE,
        "c_thought": C_THOUGHT,
        "pad_latent_to_max": True,   # 与训练时保持一致
    })()

    num_eval = min(NUM_SAMPLES, len(raw_val))
    eval_raw = raw_val.select(range(num_eval))

    # 构造前缀数据集：[question] <|start-latent|> <latent>×k <|end-latent|>
    prefix_ds = get_question_latent_dataset(
        scheduled_stage=k,
        base_dataset_valid=eval_raw,
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
    )

    correct_answers     = 0
    correct_steps       = 0
    total_lang_steps    = 0   # 应由语言生成的步骤总数
    full_match_samples  = 0   # 所有语言步骤全部匹配的样本数

    for i in range(num_eval):
        sample     = prefix_ds[i]
        raw_sample = eval_raw[i]

        input_ids  = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0).to(DEVICE)
        input_len  = input_ids.shape[1]

        # 生成剩余内容
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_GEN_TOKENS,
        )

        # 切掉输入前缀，只保留新生成的 tokens
        # gen_ids 包含完整序列（input + generated），与 coconut.py generate() 行为一致
        gen_tokens = gen_ids[0, input_len:].tolist()
        gen_text   = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # 解析
        gen_steps, gen_answer = parse_generated(gen_text)

        # GT 信息
        gt_answer      = str(raw_sample["answer"]).strip()
        n_actual_steps = len(raw_sample["steps"])
        # 语言部分应生成的步骤：steps[k:] （k 步已被 latent 处理）
        n_latent_steps = min(k, n_actual_steps)     # 实际被 latent 覆盖的步骤数
        gt_lang_steps  = raw_sample["steps"][n_latent_steps:]  # 剩余语言步骤

        # 评估 answer
        if normalize(gt_answer) == normalize(gen_answer):
            correct_answers += 1

        # 评估语言步骤（逐条比较）
        n_lang = len(gt_lang_steps)
        total_lang_steps += n_lang
        sample_step_correct = 0
        for step_idx, gt_step in enumerate(gt_lang_steps):
            gen_step = gen_steps[step_idx] if step_idx < len(gen_steps) else ""
            if normalize(gt_step) == normalize(gen_step):
                correct_steps += 1
                sample_step_correct += 1

        if sample_step_correct == n_lang:
            full_match_samples += 1

    return {
        "k":               k,
        "answer_acc":      correct_answers / num_eval,
        "step_acc":        correct_steps / total_lang_steps if total_lang_steps > 0 else 1.0,
        "full_match_acc":  full_match_samples / num_eval,
        "lang_steps_avg":  total_lang_steps / num_eval,   # 平均每条需语言生成的步骤数
    }


# ---------------------------------------------------------------------------
# 打印示例（仅 k=3 的前几条）
# ---------------------------------------------------------------------------
@torch.no_grad()
def print_examples(model, tokenizer, raw_val, latent_id, start_id, end_id, k=3, n=3):
    configs = type("cfg", (), {
        "max_latent_stage": MAX_STAGE,
        "c_thought": C_THOUGHT,
        "pad_latent_to_max": True,
    })()

    prefix_ds = get_question_latent_dataset(
        scheduled_stage=k,
        base_dataset_valid=raw_val.select(range(n)),
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
    )

    print(f"\n{'='*60}")
    print(f"Examples for k={k} (latent steps: 1..{k}, language steps: {k+1}..N)")
    print(f"{'='*60}")

    for i in range(n):
        sample     = prefix_ds[i]
        raw_sample = raw_val[i]

        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0).to(DEVICE)
        input_len = input_ids.shape[1]

        gen_ids  = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_GEN_TOKENS,
        )
        gen_text = tokenizer.decode(gen_ids[0, input_len:].tolist(), skip_special_tokens=True).strip()
        gen_steps, gen_answer = parse_generated(gen_text)

        n_latent = min(k, len(raw_sample["steps"]))
        gt_lang_steps = raw_sample["steps"][n_latent:]
        gt_answer     = str(raw_sample["answer"]).strip()

        print(f"\n[Sample {i+1}]")
        print(f"  gen_text: {repr(gen_text)}")
        print(f"  [Latent   1..{n_latent}] (hidden, not generated)")
        for j, step in enumerate(gt_lang_steps):
            gen  = gen_steps[j] if j < len(gen_steps) else "<missing>"
            mark = "✓" if normalize(step) == normalize(gen) else "✗"
            print(f"  [Language step {n_latent+j+1}] GT : {step}")
            print(f"  [Language step {n_latent+j+1}] Gen: {gen}  [{mark}]")
        print(f"  GT  Answer: {gt_answer}")
        print(f"  Gen Answer: {gen_answer}  [{'✓' if normalize(gt_answer)==normalize(gen_answer) else '✗'}]")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    model, tokenizer, latent_id, start_id, end_id = build_model()
    raw_val = get_dataset(VAL_PATH, tokenizer)

    # 打印所有 k 的示例（每个 k 展示 2 条）
    for k in range(MAX_STAGE + 1):
        print_examples(model, tokenizer, raw_val, latent_id, start_id, end_id, k=k, n=2)

    # 遍历所有 k，汇总结果
    print(f"\n{'='*60}")
    print(f"{'k':>3}  {'answer_acc':>12}  {'step_acc':>10}  {'full_match':>10}  {'lang_steps/sample':>18}")
    print(f"{'='*60}")

    results = []
    for k in range(MAX_STAGE + 1):
        res = evaluate_k(model, tokenizer, raw_val, latent_id, start_id, end_id, k)
        results.append(res)
        print(
            f"{res['k']:>3}  "
            f"{res['answer_acc']*100:>11.1f}%  "
            f"{res['step_acc']*100:>9.1f}%  "
            f"{res['full_match_acc']*100:>9.1f}%  "
            f"{res['lang_steps_avg']:>18.2f}"
        )

    print(f"{'='*60}")
    print("k=0: pure language reasoning (no latent)")
    print(f"k={MAX_STAGE}: full latent reasoning (only answer generated)")


if __name__ == "__main__":
    main()
