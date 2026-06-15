# Gemini Context: Staffmini FHA

## What This Repo Is

This is a clean repo for a small experimental language model architecture called Fractal Hybrid Architecture (FHA). It is a causal language model, not a diffusion model.

The goal is to test whether a small model can become more capable by changing the architecture, especially the way local token states are compressed into higher-level semantic anchors.

The user wants intelligence to be internal to the model. External retrieval, external repair inference, and post-generation fixes are not considered valid solutions for the core design.

## Core Architecture

FHA has three parts inside each block:

1. Micro token mixer

- Causal depthwise convolution over token states.
- Handles local syntax and token flow.
- O(N) over sequence length.

2. Gating Anchor bridge

- Compresses each 16-token block into one or more anchor vectors.
- This is the most important experimental part.
- The bridge decides what information survives into global reasoning.

3. Macro anchor transformer

- Causal transformer over anchors, not over every token.
- For 4096 tokens and stride 16, macro attention sees about 256 anchors.
- This reduces attention cost while retaining long-range global state.

Feedback:

- Macro states are shifted into the next block before being injected back into token states.
- This prevents future leakage.

## Current Model Variants

### V14 Baseline: Mean Anchor

Config:

```text
configs/model_config_v14_baseline_mean_50m.yaml
```

Compression:

```text
anchor = linear(mean(block_tokens))
```

Strength:

- Stable.

Weakness:

- Too lossy and bland.

### V14.2: Gated Delta Anchor

Config:

```text
configs/model_config_v14_2_gated_delta_anchor_50m.yaml
```

Compression:

```text
anchor = mean_proj(mean(block)) + sigmoid(delta_gate) * delta_proj(selected_token_state + endpoints)
```

Strength:

- Best FHA direction so far.
- Preserves stable mean information while adding selective salience.

Weakness:

- Still may not force the anchor to carry enough semantic/factual/predictive information.

### V14.3: Anchor-Predictive

Config:

```text
configs/model_config_v14_3_anchor_predictive_50m.yaml
```

Adds internal auxiliary loss:

```text
anchor_t predicts anchor_{t+1}
```

Result:

- Technically works.
- Worse than V14.2 at 300 steps.

Likely problem:

- Predicting the next anchor vector may be too indirect and may fight LM learning.

## Known Test Results

### V14.2 Gated Delta, 300 steps

Held-out eval from the original repo:

| seq | loss |
|---:|---:|
| 512 | 5.3406 |
| 1024 | 5.4031 |
| 2048 | 5.4594 |

### V14.2 Gated Delta, 1500 steps

Held-out eval from the original repo:

| seq | loss | top-1 |
|---:|---:|---:|
| 512 | 3.8078 | 35.19% |
| 1024 | 3.8713 | 34.55% |
| 2048 | 3.9691 | 33.48% |

### V14.3 Anchor-Predictive, 300 steps

| seq | loss | top-1 |
|---:|---:|---:|
| 512 | 5.3781 | 22.92% |
| 1024 | 5.4394 | 22.21% |
| 2048 | 5.4912 | 21.79% |

Interpretation:

- V14.3 is not better than V14.2.
- The current best baseline is V14.2 Gated Delta Anchor.

## Important Caveat

A previous baseline-vs-V14.2 comparison had a config bug: `anchor_slots=1` implicitly selected gated-delta behavior. This clean repo fixes that with explicit `fha.anchor_type`.

Use these explicit anchor types:

```yaml
fha:
  anchor_type: mean
```

```yaml
fha:
  anchor_type: gated_delta
```

```yaml
fha:
  anchor_type: multi_slot
```

## What We Need Help With

We need a better internal bridge/objective idea.

Constraints:

- Must be internal to the model/training.
- No external retrieval.
- No external repair generation.
- Must remain trainable on consumer/local GPU for fast experiments.
- Should be scalable if signal is good.

Main question:

How should the Gating Anchor compress 16 tokens into a compact vector without losing crucial semantics?

Candidate next directions:

1. Anchor predicts next-block token distribution instead of next anchor vector.
2. Anchor predicts a small set of key tokens from the next block.
3. Anchor uses a bottleneck objective inspired by information bottleneck or predictive coding.
4. Anchor is trained with a contrastive objective: correct next block vs nearby wrong blocks.
5. Macro feedback is made more selective, e.g. token-level gate decides how much global context to accept.

Please analyze the design, identify the weakest link, and propose one minimal model-internal change that is most likely to improve held-out loss and real-life generation.
