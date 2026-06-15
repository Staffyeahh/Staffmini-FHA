# V14 FHA Lite Test - 2026-06-15

## Goal

Test a small Fractal Hybrid Architecture prototype:

- Micro layer: causal local token mixer for syntax/flow.
- Gating anchor: compress every 16 tokens into a block anchor.
- Macro layer: causal transformer over anchors.
- Feedback loop: previous completed anchor state is projected back into the next token block.

This is a causal LM experiment, not a diffusion model.

## Implementation

Files:

- `src/staffyeahh_mini/modeling_mini_fha_v14.py`
- `configs/model_config_v14_fha_lite_50m_sample25.yaml`
- `configs/train_config_v14_fha_lite_300.yaml`
- `scripts/eval_causal_suite.py`

Important leakage guard:

The macro anchor for a block is not fed back into the same block. It is shifted forward and used by the next block, so token prediction cannot see future tokens from its own 16-token block through pooling.

## Model Size

| Component | Params |
|---|---:|
| total | 41.55M |
| embedding / tied LM head | 11.63M |
| macro attention projections | 5.90M |
| other FHA/micro/FFN params | 24.02M |

## Smoke Test

Command:

```powershell
python scripts/smoke_test.py --config configs/model_config_v14_fha_lite_50m_sample25.yaml --seq-len 512 --batch-size 4
```

Result:

| Metric | Value |
|---|---:|
| logits shape | `(4, 512, 30294)` |
| initial loss | 10.3944 |

## 300-Step Training

Command:

```powershell
python scripts/train.py --model-config configs/model_config_v14_fha_lite_50m_sample25.yaml --train-config configs/train_config_v14_fha_lite_300.yaml --token-bin data/tokenized/cosmopedia_mix_100m.bin --output-dir outputs/v14_fha_lite_scratch_300
```

Final checkpoint:

```text
outputs/v14_fha_lite_scratch_300/step_0000300/model.pt
```

Training metrics:

| Step | Loss | LR | Tok/s |
|---:|---:|---:|---:|
| 1 | 10.3951 | 4.44e-05 | 30,196 |
| 50 | 7.1927 | 4.00e-04 | 56,424 |
| 100 | 6.6664 | 4.00e-04 | 56,768 |
| 150 | 5.8495 | 4.00e-04 | 56,561 |
| 200 | 5.6690 | 4.00e-04 | 56,887 |
| 250 | 5.5452 | 2.47e-04 | 56,740 |
| 300 | 5.4690 | 0.00e+00 | 56,915 |

Runtime:

| Metric | Value |
|---|---:|
| total time | ~3m 15s |
| avg step | 0.65s |
| max VRAM | ~9.70GB |

TensorBoard/protobuf warnings appeared after training finished, but the checkpoint was saved successfully.

## Causal Eval

Command:

```powershell
python scripts/eval_causal_suite.py --model-config configs/model_config_v14_fha_lite_50m_sample25.yaml --checkpoint outputs/v14_fha_lite_scratch_300/step_0000300/model.pt --token-bin data/tokenized/cosmopedia_mix_5m.bin --sequence-lengths 512 1024 2048 --batches 50 --batch-size 8 --output-json outputs/v14_fha_lite_scratch_300/eval_causal_300.json --output-md outputs/v14_fha_lite_scratch_300/eval_causal_300.md
```

| Seq | Loss | PPL | Next-token top1 | Tok/s |
|---:|---:|---:|---:|---:|
| 512 | 5.3469 | 210.0 | 23.25% | 25,228 |
| 1024 | 5.4088 | 223.4 | 22.52% | 48,114 |
| 2048 | 5.4644 | 236.1 | 22.06% | 83,749 |

Learned feedback gate after 300 steps:

```text
0.1201, 0.1210, 0.1214, 0.1222, 0.1218, 0.1213, 0.1211, 0.1214
```

Mean feedback gate: `0.1213`.

## Generation Samples

Prompt:

```text
The capital of France is
```

Output:

```text
The capital of France is a new way to understand the way.

In conclusion, the first glance, the first example of the first time to find the same number of the same number of the original number of the equation
```

Prompt:

```text
Once upon a time
```

Output:

```text
Once upon a time, we can be a closer look at the same time.

Now, let's consider the first example of the first example of the first number of the equation.
```

Prompt:

```text
In Python, a function is
```

Output:

```text
In Python, a function is to find the same of the world of the world of the same.

Now, let's consider the first example of the equation:
```

## Interpretation

V14 FHA-lite trains cleanly and quickly. Unlike the recent diffusion copy-circuit test, it immediately learns a usable causal LM signal and produces grammatical continuations after only 300 steps.

The weakness is also clear: generation falls into generic repetition and high-prior phrase loops. That is expected at 300 steps, but it means the current test only proves trainability and speed, not intelligence.

The architectural idea is worth continuing if the next test measures whether the macro anchor actually helps long-context behavior. Right now the feedback gate is active but small (`~0.12`), so the model may still be mostly driven by the micro/token path.

## Next Test

Recommended V14.1:

1. Add a no-macro ablation config with the same parameter budget.
2. Train both FHA-lite and no-macro for 1500 steps.
3. Compare eval loss at 512/1024/2048 plus a synthetic long-range dependency benchmark.
4. Only scale if FHA beats the no-macro ablation at longer sequence lengths.

