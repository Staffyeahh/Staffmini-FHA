# Staffmini-FHA

Clean-room repo for the V14 Fractal Hybrid Architecture experiments.

This repo intentionally contains only the FHA design line:

- token-level causal micro mixer
- block-level Gating Anchor bridge
- causal macro transformer over compressed anchors
- shifted global feedback into the next token block
- optional internal anchor-prediction auxiliary loss

It does not include the older V10/V11 MoE line, V12/V13 diffusion line, or external repair/retrieval logic.

## Layout

```text
src/staffmini_fha/
  config.py        minimal FHA config
  model.py         self-contained FHA model
  data.py          packed token dataset

configs/
  model_config_v14_baseline_mean_50m.yaml
  model_config_v14_2_gated_delta_anchor_50m.yaml
  model_config_v14_3_anchor_predictive_50m.yaml
  train_config_300.yaml

scripts/
  count_params.py
  smoke_test.py
  train.py
  eval_causal_suite.py
  generate.py
  quick_benchmark.py

docs/
  FHA_TIMELINE.md
  GEMINI_FHA_CONTEXT.md
  previous V14 test reports
```

## Quick Start

Count params:

```powershell
python scripts/count_params.py configs/model_config_v14_2_gated_delta_anchor_50m.yaml
```

Smoke test:

```powershell
python scripts/smoke_test.py --config configs/model_config_v14_2_gated_delta_anchor_50m.yaml --seq-len 512 --batch-size 4
```

Train 300 steps:

```powershell
python scripts/train.py --model-config configs/model_config_v14_2_gated_delta_anchor_50m.yaml --train-config configs/train_config_300.yaml --token-bin data/tokenized/cosmopedia_mix_100m.bin --output-dir outputs/v14_2_gated_delta_300
```

Evaluate:

```powershell
python scripts/eval_causal_suite.py --model-config configs/model_config_v14_2_gated_delta_anchor_50m.yaml --checkpoint outputs/v14_2_gated_delta_300/step_0000300/model.pt --token-bin data/tokenized/cosmopedia_mix_5m.bin --sequence-lengths 512 1024 2048 --batches 50 --batch-size 8 --output-md outputs/v14_2_gated_delta_300/eval_causal_300.md
```

Generate:

```powershell
python scripts/generate.py --model-config configs/model_config_v14_2_gated_delta_anchor_50m.yaml --checkpoint outputs/v14_2_gated_delta_300/step_0000300/model.pt --prompt "The capital of France is"
```
