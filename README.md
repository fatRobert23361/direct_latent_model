# Coconut — Reasoning in a Continuous Latent Space

This repository extends the official implementation of
[Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
with two additional contributions:

- **DirectLatentModel**: a fixed-latent-count model trained with a continuous `p_mask` schedule
  and an auxiliary Translator loss that forces genuine representational load on the latent tokens.
- **Latent replacement experiments**: position-wise perturbation evaluations that test whether
  latent tokens are causally load-bearing.

---

## Repository File Guide

### Keep — core source files

| File | Role |
|------|------|
| `coconut.py` | Vanilla COCONUT model (multi-stage curriculum) |
| `direct_latent_model.py` | DirectLatentModel + Translator |
| `translator_v3.py` | GPT-2-based Translator decoder |
| `dataset.py` | Dataset for `run.py` (ProsQA / GSM8K vanilla COCONUT) |
| `direct_latent_dataset.py` | Dataset for `train_direct_latent.py` |
| `utils.py` | Config wrapper and seed utilities |
| `run.py` | Training loop for vanilla COCONUT and ProsQA/GSM8K baselines |
| `train_direct_latent.py` | Training loop for DirectLatentModel (all datasets) |
| `train_sft_hotpot.py` | CoT SFT baseline for HotpotQA (alternative, reports EM + F1) |
| `train_nocot_hotpot.py` | No-CoT baseline for HotpotQA (alternative, reports EM + F1) |
| `eval_latent_replacement.py` | Latent replacement eval — vanilla COCONUT |
| `eval_latent_replacement_direct.py` | Latent replacement eval — DirectLatentModel |
| `eval_latent_sweep.py` | Accuracy vs. latent count sweep — vanilla COCONUT |
| `eval_latent_sweep_direct.py` | Accuracy vs. latent count sweep — DirectLatentModel |
| `eval_cot_accuracy.py` | One-shot CoT model evaluation (GSM8K / ProsQA) |
| `eval_baseline_hotpot.py` | Raw GPT-2 (no fine-tuning) baseline on HotpotQA |
| `prepare_hotpot_cot.py` | Build HotpotQA-CoT dataset from fsiddiqui2 corpus |
| `prepare_hotpotqa.py` | HotpotQA preprocessing utilities |
| `preprocessing/` | GSM8K and ProntoQA preprocessing scripts |

### Delete — redundant files

```bash
# Dead code after mixed.py chain removal
git rm mixed.py mixed_dataset.py train.py \
       eval_latent_replacement_hybrid.py train_uniform_sweep.py \
       args/mixed_coconut.yaml

# Stale configs (placeholder paths or superseded)
git rm args/prontoqa_coconut.yaml args/prontoqa_coconut_eval.yaml \
       args/hotpot_no_thoughts.yaml
```

---

## Environment Setup

```bash
conda create --name coconut python=3.12
conda activate coconut
pip install -r requirements.txt
wandb login          # required before any training run
```

All multi-GPU training uses `torchrun`. Commands below assume **4 × A100 (80 GB)**.
Adjust `--nproc_per_node`, `batch_size_training`, and `gradient_accumulation_steps` for
other hardware.

---

## Data Preparation

### ProsQA (already bundled)

Pre-processed files are in `data/prosqa_{train,valid,test}.json`. No extra step needed.

### GSM8K

```bash
bash preprocessing/gsm_icot.bash
```

Produces `data/gsm_{train,valid,test}.json`.
The training set uses the augmented iCoT corpus (~385k samples).

### HotpotQA

```bash
python prepare_hotpot_cot.py   # downloads fsiddiqui2/hotpot-qa-cot-reasoning and builds CoT chains
python prepare_hotpotqa.py     # filters to ≤1024 tokens and writes train/valid/test splits
```

Produces `data/hotpot_cot_{train,valid,test}.json`.

---

## Experiment 1 — Baselines (CoT and No-CoT)

### ProsQA

```bash
# CoT baseline
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_cot.yaml

# No-CoT baseline
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_no_thoughts.yaml
```

### GSM8K

```bash
# CoT baseline  (also serves as warm-start checkpoint for vanilla COCONUT)
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_cot.yaml

# No-CoT baseline — create args/gsm_no_thoughts.yaml by copying prosqa_no_thoughts.yaml
# and setting:
#   train_path: data/gsm_train.json
#   val_path:   data/gsm_valid.json
#   test_path:  data/gsm_test.json
#   name:       gsm-no-thoughts
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_no_thoughts.yaml
```

> **Note:** `args/gsm_cot.yaml` sets `name: nihaoyang2002-kth-royal-institute-of-technology`
> (a legacy experiment name). Rename the field if desired before running.

### HotpotQA

```bash
# CoT baseline
torchrun --nnodes 1 --nproc_per_node 4 run.py args/hotpot_cot.yaml

# No-CoT baseline
torchrun --nnodes 1 --nproc_per_node 4 run.py args/hotpot_no_thoughts.yaml
```

> **Note:** `run.py` reports accuracy as direct string match (same metric as ProsQA/GSM8K),
> not the normalized EM + token F1 used in the original HotpotQA benchmark.

---

## Experiment 2 — Vanilla COCONUT

Supported on **ProsQA** and **GSM8K** only.

### ProsQA

```bash
# Train
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut.yaml

# Evaluate — fill in the best checkpoint path in args/prosqa_coconut_eval.yaml first:
#   load_model_path: models/prosqa-coconut/checkpoint_XX
torchrun --nnodes 1 --nproc_per_node 4 run.py args/prosqa_coconut_eval.yaml
```

### GSM8K

GSM8K vanilla COCONUT requires a CoT warm-start checkpoint.

1. Train the CoT baseline (Experiment 1) and identify a checkpoint where validation
   accuracy is ~40%.
2. Set `load_model_path` in `args/gsm_coconut.yaml` to that checkpoint path.
3. Run:

```bash
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml

# Evaluate — fill load_model_path in args/gsm_coconut_eval.yaml first
torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut_eval.yaml
```

### Key hyperparameters (`args/prosqa_coconut.yaml`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `max_latent_stage` | 6 | Replace up to 6 CoT steps with latent tokens |
| `c_thought` | 1 | One latent token per replaced step |
| `epochs_per_stage` | 5 | Epochs trained at each curriculum stage |
| `uniform_prob` | 0.3 | Probability of sampling lower-stage data (regularisation) |
| `reset_optimizer` | true | Optimizer reset at each stage transition |

---

## Experiment 3 — DirectLatentModel

Supported on **ProsQA**, **GSM8K**, and **HotpotQA**.
All variants use `train_direct_latent.py` (single-process, no `torchrun` needed).

### With Translator auxiliary loss

```bash
# ProsQA — create args/direct_latent_prosqa.yaml (copy direct_latent_no_trans_prosqa.yaml,
#           remove use_translator: false or set use_translator: true, set save_path/name)
python train_direct_latent.py --config args/direct_latent_prosqa.yaml

# GSM8K
python train_direct_latent.py --config args/direct_latent_gsm8k.yaml

# HotpotQA
python train_direct_latent.py --config args/direct_latent_hotpot.yaml
```

### Without Translator (ablation)

```bash
python train_direct_latent.py --config args/direct_latent_no_trans_prosqa.yaml
python train_direct_latent.py --config args/direct_latent_no_trans_gsm8k.yaml
python train_direct_latent.py --config args/direct_latent_no_trans_hotpotQA.yaml
```

### Key hyperparameters (`args/direct_latent_no_trans_prosqa.yaml` / `direct_latent_gsm8k.yaml`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_latent` | 6 | Fixed number of latent tokens (no curriculum) |
| `p_mask_start` | 0.0 | CoT always visible at epoch 0 |
| `p_mask_end` | 1.0 | CoT fully masked at ramp end |
| `p_mask_warmup_epochs` | 5 (ProsQA) / 15 (GSM8K) | Epochs before ramp begins |
| `p_mask_ramp_end_epoch` | 49 (ProsQA) / 85 (GSM8K) | Epoch at which `p_mask` reaches 1.0 |
| `lambda_translator` | 0.5 | Translator loss weight |
| `use_translator` | true / false | Enable or disable Translator auxiliary loss |

---

## Evaluation — Latent Replacement Experiment

The replacement experiment substitutes each latent position with (a) Gaussian noise or
(b) a vector from an unrelated sample, measuring accuracy drop at each position.

```bash
# Vanilla COCONUT
python eval_latent_replacement.py \
    --checkpoint models/prosqa-coconut/checkpoint_XX \
    --val_path   data/prosqa_test.json \
    --stage      6 \
    --output_json results/latent_replacement.json \
    --output_plot results/latent_replacement.png

# DirectLatentModel
python eval_latent_replacement_direct.py \
    --checkpoint models/direct_latent_prosqa/best.pt \
    --val_path   data/prosqa_test.json \
    --output_json results/latent_replacement_direct.json
```

## Evaluation — Latent Sweep

Measures accuracy as a function of the number of active latent tokens (0 → n_latent).

```bash
# Vanilla COCONUT
python eval_latent_sweep.py \
    --checkpoint models/prosqa-coconut/checkpoint_XX \
    --val_path   data/prosqa_test.json \
    --output_json results/latent_sweep.json

# DirectLatentModel
python eval_latent_sweep_direct.py \
    --checkpoint models/direct_latent_prosqa/best.pt \
    --val_path   data/prosqa_test.json \
    --output_json results/latent_sweep_direct.json
```

---

## Logging

All training scripts log to [Weights & Biases](https://wandb.ai). Set `debug: true` in
any config file to disable wandb and checkpoint saving (useful for quick sanity checks).

---

## Citation

```bibtex
@article{hao2024training,
  title   = {Training Large Language Models to Reason in a Continuous Latent Space},
  author  = {Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian
             and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal = {arXiv preprint arXiv:2412.06769},
  year    = {2024}
}
```

## License

This code is released under the MIT license (see [LICENSE](LICENSE)).
