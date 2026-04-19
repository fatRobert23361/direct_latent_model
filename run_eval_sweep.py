"""
run_eval_sweep.py

对 models/uniform_sweep/ 下的所有 uniform_prob 模型依次运行：
  1. eval_latent_sweep_hybrid.py  — 测试 stage 0-6 下的准确率（最多 12 个 latent）
  2. eval_latent_replacement_hybrid.py — 依次替换第 1-6 个 latent，测试准确率变化

结果保存在各自模型目录下。

用法:
    python run_eval_sweep.py
    python run_eval_sweep.py --sweep_dir models/uniform_sweep --probs 0.0 0.3
"""

import argparse
import os
import subprocess
import sys
import json

SWEEP_DIR  = "models/uniform_sweep"
VAL_PATH   = "data/prosqa_test.json"
MAX_STAGE  = 6
C_THOUGHT  = 1


def find_checkpoint(model_dir):
    """返回 model_dir 下的 _best.pt 文件路径，找不到返回 None。"""
    for fname in os.listdir(model_dir):
        if fname.endswith("_best.pt"):
            return os.path.join(model_dir, fname)
    return None


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [WARNING] Command exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", default=SWEEP_DIR)
    parser.add_argument("--val_path",  default=VAL_PATH)
    parser.add_argument("--max_stage", type=int, default=MAX_STAGE)
    parser.add_argument("--c_thought", type=int, default=C_THOUGHT)
    parser.add_argument("--probs",     nargs="*", type=float, default=None,
                        help="只跑指定 uniform_prob 值，如 0.0 0.3；不指定则跑全部")
    parser.add_argument("--skip_sweep",       action="store_true",
                        help="跳过 latent sweep 评估")
    parser.add_argument("--skip_replacement", action="store_true",
                        help="跳过 latent replacement 评估")
    args = parser.parse_args()

    # 找所有模型目录
    all_dirs = sorted([
        d for d in os.listdir(args.sweep_dir)
        if os.path.isdir(os.path.join(args.sweep_dir, d))
        and d.startswith("uniform_prob_")
    ])

    if args.probs is not None:
        # 按用户指定的 prob 值过滤
        def prob_to_dirname(p):
            return f"uniform_prob_{p:.1f}".replace(".", "p")
        target_dirs = {prob_to_dirname(p) for p in args.probs}
        all_dirs = [d for d in all_dirs if d in target_dirs]

    print(f"Found {len(all_dirs)} model directories: {all_dirs}")

    summary = {}

    for dirname in all_dirs:
        model_dir = os.path.join(args.sweep_dir, dirname)
        ckpt_path = find_checkpoint(model_dir)

        if ckpt_path is None:
            print(f"\n[SKIP] No _best.pt found in {model_dir}")
            continue

        print(f"\n\n{'#'*60}")
        print(f"# Processing: {dirname}")
        print(f"# Checkpoint: {ckpt_path}")
        print(f"{'#'*60}")

        sweep_json       = os.path.join(model_dir, "latent_sweep_results.json")
        sweep_plot       = os.path.join(model_dir, "latent_sweep_results.png")
        replacement_json = os.path.join(model_dir, "latent_replacement_results.json")

        # ---- 1. Latent Sweep ----
        if not args.skip_sweep:
            sweep_rc = run_cmd([
                sys.executable, "eval_latent_sweep_hybrid.py",
                "--checkpoint",  ckpt_path,
                "--val_path",    args.val_path,
                "--max_stage",   str(12),
                "--c_thought",   str(args.c_thought),
                "--output_json", sweep_json,
                "--output_plot", sweep_plot,
            ], desc=f"[{dirname}] Latent Sweep")
        else:
            sweep_rc = 0

        # ---- 2. Latent Replacement ----
        if not args.skip_replacement:
            repl_rc = run_cmd([
                sys.executable, "eval_latent_replacement_hybrid.py",
                "--checkpoint",  ckpt_path,
                "--val_path",    args.val_path,
                "--stage",       str(6),
                "--c_thought",   str(args.c_thought),
                "--output_json", replacement_json,
            ], desc=f"[{dirname}] Latent Replacement")
        else:
            repl_rc = 0

        summary[dirname] = {
            "checkpoint":       ckpt_path,
            "sweep_ok":         (sweep_rc == 0),
            "replacement_ok":   (repl_rc == 0),
            "sweep_json":       sweep_json,
            "replacement_json": replacement_json,
        }

    # 汇总
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for dirname, info in summary.items():
        sweep_status = "OK" if info["sweep_ok"] else "FAILED"
        repl_status  = "OK" if info["replacement_ok"] else "FAILED"
        print(f"  {dirname}:  sweep={sweep_status}  replacement={repl_status}")

    # 读取并打印最终准确率对比
    print(f"\n{'='*60}")
    print("ACCURACY COMPARISON (stage=6, baseline)")
    print(f"{'='*60}")
    for dirname in sorted(summary.keys()):
        repl_json = summary[dirname]["replacement_json"]
        sweep_json = summary[dirname]["sweep_json"]

        repl_acc = None
        if os.path.exists(repl_json):
            try:
                data = json.load(open(repl_json))
                repl_acc = data["results"]["baseline"]["accuracy"]
            except Exception:
                pass

        sweep_acc = None
        if os.path.exists(sweep_json):
            try:
                data = json.load(open(sweep_json))
                last = data[-1]  # stage=max_stage
                sweep_acc = last["accuracy"]
            except Exception:
                pass

        print(f"  {dirname}:")
        if sweep_acc is not None:
            print(f"    sweep stage-{MAX_STAGE} acc = {sweep_acc*100:.2f}%")
        if repl_acc is not None:
            print(f"    replacement baseline acc  = {repl_acc*100:.2f}%")

    # 保存汇总到 sweep_dir
    summary_path = os.path.join(args.sweep_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
