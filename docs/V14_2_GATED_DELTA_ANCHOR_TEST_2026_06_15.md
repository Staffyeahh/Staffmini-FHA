# V14.2 Gated Delta Anchor Test - 2026-06-15

## Goal

Test a cleaner Gating Anchor idea than V14.1.

V14.1 used 4 anchor slots per 16-token block. That made the compressor expressive, but also increased macro sequence length and made optimization harder.

V14.2 keeps the original compression ratio:

```text
16 tokens -> 1 anchor
```

But replaces plain mean pooling with:

```text
anchor = mean_pool(tokens) + sigmoid(gate) * selective_delta(tokens)
```

This keeps the stable mean anchor as a backbone and lets the model add learned, token-selective information only when useful.

## Implementation

New module:

- `GatedDeltaAnchor`

Files:

- `src/staffyeahh_mini/modeling_mini_fha_v14.py`
- `configs/model_config_v14_2_gated_delta_anchor_50m_sample25.yaml`

Compression details:

- `anchor_stride: 16`
- `anchor_slots: 1`
- `anchor_temperature: 0.75`
- learned token importance over the 16-token block
- learned value projection for selected token content
- first/last token endpoint carry
- learned delta gate initialized conservatively
- feedback remains shifted to the next block to avoid future leakage

## Model Size

| Model | Params |
|---|---:|
| V14 FHA-lite mean anchor | 41.55M |
| V14.1 selective 4-slot anchor | 45.17M |
| V14.2 gated delta anchor | 46.33M |

## 300-Step Training

Command:

```powershell
python scripts/train.py --model-config configs/model_config_v14_2_gated_delta_anchor_50m_sample25.yaml --train-config configs/train_config_v14_fha_lite_300.yaml --token-bin data/tokenized/cosmopedia_mix_100m.bin --output-dir outputs/v14_2_gated_delta_anchor_scratch_300
```

Final checkpoint:

```text
outputs/v14_2_gated_delta_anchor_scratch_300/step_0000300/model.pt
```

Training metrics:

| Step | Loss | Feedback gate | Anchor entropy | Tok/s |
|---:|---:|---:|---:|---:|
| 1 | 10.4231 | 0.119 | 1.818 | 25,600 |
| 50 | 7.4074 | 0.120 | 2.026 | 45,330 |
| 100 | 6.4127 | 0.120 | 1.646 | 45,991 |
| 150 | 5.8540 | 0.120 | 1.514 | 45,801 |
| 200 | 5.5006 | 0.121 | 1.469 | 47,132 |
| 250 | 5.2850 | 0.121 | 1.414 | 47,987 |
| 300 | 5.4220 | 0.121 | 1.421 | 48,724 |

Runtime:

| Metric | Value |
|---|---:|
| total time | ~3m 48s |
| avg step | 0.76s |
| max VRAM | ~10.49GB |

## Causal Eval

Command:

```powershell
python scripts/eval_causal_suite.py --model-config configs/model_config_v14_2_gated_delta_anchor_50m_sample25.yaml --checkpoint outputs/v14_2_gated_delta_anchor_scratch_300/step_0000300/model.pt --token-bin data/tokenized/cosmopedia_mix_5m.bin --sequence-lengths 512 1024 2048 --batches 50 --batch-size 8 --output-json outputs/v14_2_gated_delta_anchor_scratch_300/eval_causal_300.json --output-md outputs/v14_2_gated_delta_anchor_scratch_300/eval_causal_300.md
```

| Model | Seq | Loss | PPL | Top1 | Tok/s |
|---|---:|---:|---:|---:|---:|
| V14 mean anchor | 512 | 5.3469 | 210.0 | 23.25% | 25,228 |
| V14.1 4-slot anchor | 512 | 5.3925 | 219.8 | 22.85% | 24,497 |
| V14.2 gated delta | 512 | 5.3406 | 208.6 | 23.27% | 24,054 |
| V14 mean anchor | 1024 | 5.4088 | 223.4 | 22.52% | 48,114 |
| V14.1 4-slot anchor | 1024 | 5.4519 | 233.2 | 22.18% | 46,142 |
| V14.2 gated delta | 1024 | 5.4031 | 222.1 | 22.53% | 45,307 |
| V14 mean anchor | 2048 | 5.4644 | 236.1 | 22.06% | 83,749 |
| V14.1 4-slot anchor | 2048 | 5.5075 | 246.5 | 21.72% | 83,616 |
| V14.2 gated delta | 2048 | 5.4594 | 235.0 | 22.08% | 84,506 |

## Generation Samples

Prompt:

```text
The capital of France is
```

Output:

```text
The capital of France is a significant role in shaping the importance of the world of the world of the world of the world of the world.
```

Prompt:

```text
In Python, a function is
```

Output:

```text
In Python, a function is the same time to be a new number of life.

Now, let's consider the same number of the number of the number of the number of the number of the number of the number of
```

## Interpretation

This is the first positive Gating Anchor result.

V14.2 beats V14 mean pooling on all three held-out eval lengths after the same 300-step training budget:

- 512: `5.3469 -> 5.3406`
- 1024: `5.4088 -> 5.4031`
- 2048: `5.4644 -> 5.4594`

The gain is small, but it is consistent. V14.1 failed because it increased macro sequence length and made the compression task too broad. V14.2 is cleaner: it preserves the stable mean anchor and adds selective information as a controlled delta.

The generation samples are still repetitive, so this does not prove a smart model yet. It does show that the bridge idea is not dead. The Gating Anchor can improve the training/eval signal when it is constrained properly.

## Decision

V14.2 is the best FHA bridge variant so far.

Recommended next step:

Train V14.2 and V14 mean-anchor baseline to 1500 steps with the same seed/config and compare:

- eval loss at 512/1024/2048
- next-token top1
- generation repetition
- a synthetic long-range dependency benchmark

Only scale V14.2 further if the small 300-step edge grows at 1500 steps.

