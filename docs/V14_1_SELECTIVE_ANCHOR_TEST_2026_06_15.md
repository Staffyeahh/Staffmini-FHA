# V14.1 Selective Gating Anchor Test - 2026-06-15

## Goal

Replace V14's simple 16-token mean pooling bridge with a stronger learned compression module.

V14 baseline bridge:

```text
16 token states -> mean pooling -> linear projection -> 1 anchor
```

V14.1 bridge:

```text
16 token states -> learned slot queries + token salience -> softmax selection -> 4 anchors
```

The hypothesis was that a selective multi-slot anchor would preserve more important information than averaging.

## Design

New module: `SelectiveMultiSlotAnchor`

Files:

- `src/staffyeahh_mini/modeling_mini_fha_v14.py`
- `configs/model_config_v14_1_selective_anchor_50m_sample25.yaml`
- `scripts/train.py`

Compression details:

- `anchor_stride: 16`
- `anchor_slots: 4`
- `anchor_temperature: 0.75`
- each slot has a learned query vector
- each token gets learned salience scores
- intra-block position embeddings are added before compression
- first/last token endpoint carry is projected into each anchor
- macro transformer sees flattened block-slot anchors
- feedback is still shifted to the next block to avoid future leakage

This is still internal to the model. No retrieval, no external repair, no inference add-on.

## Model Size

| Model | Params |
|---|---:|
| V14 FHA-lite | 41.55M |
| V14.1 selective anchor | 45.17M |

V14.1 adds about 3.62M parameters.

## Smoke Test

Command:

```powershell
python scripts/smoke_test.py --config configs/model_config_v14_1_selective_anchor_50m_sample25.yaml --seq-len 512 --batch-size 4
```

Result:

| Metric | Value |
|---|---:|
| logits shape | `(4, 512, 30294)` |
| initial loss | 10.3789 |

## 300-Step Training

Command:

```powershell
python scripts/train.py --model-config configs/model_config_v14_1_selective_anchor_50m_sample25.yaml --train-config configs/train_config_v14_fha_lite_300.yaml --token-bin data/tokenized/cosmopedia_mix_100m.bin --output-dir outputs/v14_1_selective_anchor_scratch_300
```

Final checkpoint:

```text
outputs/v14_1_selective_anchor_scratch_300/step_0000300/model.pt
```

Training metrics:

| Step | Loss | Feedback gate | Anchor entropy | Tok/s |
|---:|---:|---:|---:|---:|
| 1 | 10.4030 | 0.119 | 1.834 | 26,307 |
| 50 | 7.2736 | 0.120 | 2.214 | 48,236 |
| 100 | 6.6340 | 0.120 | 1.944 | 48,598 |
| 150 | 6.1096 | 0.120 | 1.785 | 49,549 |
| 200 | 5.8401 | 0.120 | 1.767 | 50,242 |
| 250 | 5.4635 | 0.121 | 1.702 | 50,569 |
| 300 | 5.4150 | 0.121 | 1.723 | 50,956 |

Runtime:

| Metric | Value |
|---|---:|
| total time | ~3m 38s |
| avg step | 0.73s |
| max VRAM | ~10.91GB |

## Causal Eval

Command:

```powershell
python scripts/eval_causal_suite.py --model-config configs/model_config_v14_1_selective_anchor_50m_sample25.yaml --checkpoint outputs/v14_1_selective_anchor_scratch_300/step_0000300/model.pt --token-bin data/tokenized/cosmopedia_mix_5m.bin --sequence-lengths 512 1024 2048 --batches 50 --batch-size 8 --output-json outputs/v14_1_selective_anchor_scratch_300/eval_causal_300.json --output-md outputs/v14_1_selective_anchor_scratch_300/eval_causal_300.md
```

| Model | Seq | Loss | PPL | Top1 | Tok/s |
|---|---:|---:|---:|---:|---:|
| V14 FHA-lite | 512 | 5.3469 | 210.0 | 23.25% | 25,228 |
| V14.1 selective | 512 | 5.3925 | 219.8 | 22.85% | 24,497 |
| V14 FHA-lite | 1024 | 5.4088 | 223.4 | 22.52% | 48,114 |
| V14.1 selective | 1024 | 5.4519 | 233.2 | 22.18% | 46,142 |
| V14 FHA-lite | 2048 | 5.4644 | 236.1 | 22.06% | 83,749 |
| V14.1 selective | 2048 | 5.5075 | 246.5 | 21.72% | 83,616 |

## Generation Samples

Prompt:

```text
The capital of France is
```

Output:

```text
The capital of France is a crucial role in shaping the importance of the world.

In conclusion, the world of the CBD's case study.

In conclusion, the world of the world of the world
```

Prompt:

```text
In Python, a function is
```

Output:

```text
In Python, a function is a crucial role in shaping the importance of the world of the world.

In conclusion, the world of the world of the world, the world of the world of the world of the world of
```

## Interpretation

The new bridge is technically working:

- training is stable
- anchor entropy changes during training
- feedback gate stays active around `0.12`
- train loss ends slightly lower than V14 (`5.4150` vs `5.4690`)

But the held-out causal eval is worse than V14 at 512, 1024, and 2048 tokens. This means the stronger bridge is not yet a win.

Most likely reason:

The selective compressor has more capacity and a harder optimization problem. At 300 steps it learns to fit the train stream, but its anchor slots are not yet aligned with generalizable semantic compression. The macro path is also still weak because the feedback gate remains low.

## Decision

Do not scale V14.1 as-is.

The idea is still promising, but the result says the bridge needs either:

1. a better objective, or
2. a simpler single-anchor version, or
3. an ablation to prove whether multiple slots help after longer training.

Recommended next version:

V14.2 should use a single learned gated anchor, not 4 slots:

```text
anchor = mean_pool(tokens) + selective_delta(tokens)
```

This keeps the original 16 -> 1 compression ratio but adds learned importance without quadrupling macro sequence length. That is the cleaner test of whether Gating Anchor compression improves the architecture.

