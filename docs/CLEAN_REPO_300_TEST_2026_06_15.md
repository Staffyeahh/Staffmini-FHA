# Clean Repo 300-Step Test - 2026-06-15

## Scope

Created a clean FHA-only repo at:

```text
C:\Users\Steff\source\repos\Staffmini-FHA
```

This repo contains only the V14/FHA design line:

- self-contained FHA config
- self-contained FHA model
- packed token dataset
- 300-step train config
- train/eval/smoke/generate/param-count scripts
- tokenizer
- Cosmopedia 100M train token bin
- Cosmopedia 5M eval token bin
- FHA timeline and Gemini context docs

No older V10/V11/V12/V13 model files were copied.

## Tested Model

Config:

```text
configs/model_config_v14_2_gated_delta_anchor_50m.yaml
```

This is the current recommended FHA baseline:

```text
anchor = mean_proj(mean(block)) + sigmoid(delta_gate) * delta_proj(selected_token_state + endpoints)
```

Parameter count:

| metric | value |
|---|---:|
| total params | 47.51M |
| trainable params | 47.51M |

## Smoke Test

Command:

```powershell
python scripts/smoke_test.py --config configs/model_config_v14_2_gated_delta_anchor_50m.yaml --seq-len 512 --batch-size 4
```

Result:

| metric | value |
|---|---:|
| logits shape | `(4, 512, 30294)` |
| loss | 10.3907 |
| lm_loss | 10.3907 |
| anchor_prediction_loss | 0.0000 |

## 300-Step Training

Command:

```powershell
python scripts/train.py --model-config configs/model_config_v14_2_gated_delta_anchor_50m.yaml --train-config configs/train_config_300.yaml --token-bin data/tokenized/cosmopedia_mix_100m.bin --output-dir outputs/v14_2_gated_delta_300
```

Checkpoint:

```text
outputs/v14_2_gated_delta_300/step_0000300/model.pt
```

Training summary:

| step | loss | lm_loss | lr | tok/s | anchor entropy |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.4120 | 10.4120 | 4.44e-05 | 28,805 | 1.869 |
| 50 | 7.2526 | 7.2526 | 4.00e-04 | 53,451 | 1.945 |
| 100 | 6.5444 | 6.5444 | 4.00e-04 | 54,328 | 1.602 |
| 150 | 6.1073 | 6.1073 | 4.00e-04 | 54,499 | 1.543 |
| 200 | 5.7858 | 5.7858 | 4.00e-04 | 55,020 | 1.504 |
| 250 | 5.5686 | 5.5686 | 2.47e-04 | 54,280 | 1.520 |
| 300 | 5.4381 | 5.4381 | 0.00e+00 | 54,439 | 1.523 |

Peak VRAM:

```text
8.63 GB
```

## Held-Out Eval

Command:

```powershell
python scripts/eval_causal_suite.py --model-config configs/model_config_v14_2_gated_delta_anchor_50m.yaml --checkpoint outputs/v14_2_gated_delta_300/step_0000300/model.pt --token-bin data/tokenized/cosmopedia_mix_5m.bin --sequence-lengths 512 1024 2048 --batches 50 --batch-size 8 --output-json outputs/v14_2_gated_delta_300/eval_causal_300.json --output-md outputs/v14_2_gated_delta_300/eval_causal_300.md
```

Result:

| seq | loss | ppl | top-1 | tok/s |
|---:|---:|---:|---:|---:|
| 512 | 5.4279 | 227.7 | 22.63% | 60,372 |
| 1024 | 5.4932 | 243.0 | 21.99% | 111,934 |
| 2048 | 5.5320 | 252.7 | 21.64% | 165,438 |

## Generation Smoke

Prompt:

```text
The capital of France is
```

Output starts:

```text
The capital of France is its unique to make it about the right experience.
```

Prompt:

```text
If a glass falls from a table, it will usually
```

Output starts:

```text
If a glass falls from a table, it will usually require a new time for you.
```

Interpretation:

- The model is not random token noise.
- It has learned common text rhythm and frequent phrases.
- It is still too undertrained for factual or causal answers.

## Notes

This clean repo uses a simpler pure-PyTorch train loop instead of the original large multi-architecture training script. Results are close enough for a fresh baseline but not bit-identical to earlier V14.2 runs.

The important success criterion for this task is met: the copied/cleaned FHA code trains, saves, evaluates, and generates inside the new repo.
