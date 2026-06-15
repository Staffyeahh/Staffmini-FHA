# V14.2 Gated Delta Anchor 1500-Step Test - 2026-06-15

## Run

Model:

```text
configs/model_config_v14_2_gated_delta_anchor_50m_sample25.yaml
```

Command:

```powershell
python scripts/train.py --model-config configs/model_config_v14_2_gated_delta_anchor_50m_sample25.yaml --train-config configs/train_config_v14_fha_lite_300.yaml --token-bin data/tokenized/cosmopedia_mix_100m.bin --output-dir outputs/v14_2_gated_delta_anchor_scratch_1500 --override max_steps=1500 save_every=500 log_every=100 schedule_steps=1500
```

Final checkpoint:

```text
outputs/v14_2_gated_delta_anchor_scratch_1500/step_0001500/model.pt
```

## Training Metrics

| Step | Loss | Feedback gate | Anchor entropy | Tok/s |
|---:|---:|---:|---:|---:|
| 1 | 10.4231 | 0.119 | 1.818 | 25,274 |
| 100 | 6.5928 | 0.120 | 1.674 | 46,467 |
| 300 | 5.3536 | 0.121 | 1.335 | 50,020 |
| 500 | 4.8719 | 0.122 | 1.348 | 51,133 |
| 800 | 4.1351 | 0.123 | 1.331 | 51,591 |
| 1000 | 4.1158 | 0.124 | 1.298 | 51,902 |
| 1200 | 4.0249 | 0.125 | 1.375 | 52,103 |
| 1500 | 3.8066 | 0.126 | 1.324 | 52,411 |

Runtime:

| Metric | Value |
|---|---:|
| total time | ~17m 36s |
| avg step | 0.70s |
| max VRAM | ~10.49GB |

## Eval

Command:

```powershell
python scripts/eval_causal_suite.py --model-config configs/model_config_v14_2_gated_delta_anchor_50m_sample25.yaml --checkpoint outputs/v14_2_gated_delta_anchor_scratch_1500/step_0001500/model.pt --token-bin data/tokenized/cosmopedia_mix_5m.bin --sequence-lengths 512 1024 2048 --batches 50 --batch-size 8 --output-json outputs/v14_2_gated_delta_anchor_scratch_1500/eval_causal_1500.json --output-md outputs/v14_2_gated_delta_anchor_scratch_1500/eval_causal_1500.md
```

| Checkpoint | Seq | Loss | PPL | Top1 |
|---|---:|---:|---:|---:|
| V14.2 300 | 512 | 5.3406 | 208.6 | 23.27% |
| V14.2 1500 | 512 | 3.8078 | 45.1 | 35.19% |
| V14.2 300 | 1024 | 5.4031 | 222.1 | 22.53% |
| V14.2 1500 | 1024 | 3.8713 | 48.0 | 34.55% |
| V14.2 300 | 2048 | 5.4594 | 235.0 | 22.08% |
| V14.2 1500 | 2048 | 3.9691 | 52.9 | 33.48% |

## Generation Samples

Prompt:

```text
The capital of France is
```

Output:

```text
The capital of France is a significant portion of the population, including the use of the state of the population. This is where the population is the population of the population, and the population is the population.
```

Prompt:

```text
Once upon a time
```

Output:

```text
Once upon a time, in the bustling city of New York, there was a small town named Harmonyville. They were known for their innovative approach to transportation and transportation, but they also had a unique perspective on the world around them.
```

Prompt:

```text
In Python, a function is
```

Output:

```text
In Python, a function is a type of object that can be used to create a system of equations. This is where the function is a function that is defined as the function of the function.
```

## Interpretation

V14.2 scales cleanly from 300 to 1500 steps. Eval loss and top1 improve strongly across all context lengths.

The Gated Delta Anchor remains active:

- feedback gate grows from `0.119` to `0.126`
- anchor entropy drops from `1.818` to around `1.324`

That means the bridge is learning more selective compression over time.

Qualitative generation improved, especially for story-like prompts, but factual and technical prompts still collapse into repeated high-prior phrases. The architecture is trainable, but the model is not yet semantically reliable.

## Next Decision

Before scaling V14.2 further, run a matching 1500-step V14 mean-anchor baseline. The 300-step result favored V14.2 slightly; the real question is whether that edge survives or grows after 1500 steps.

