# FHA Timeline

## Strategic Goal

Build a small, fast architecture that can be tested locally, then scaled if the design signal is real. The goal is not external repair, retrieval, or post-processing. Intelligence should come from the model architecture and training objective.

## V14 Baseline: FHA Lite

First Fractal Hybrid Architecture prototype.

Design:

- Micro layer: causal depthwise token mixer for local syntax.
- Bridge: one anchor per 16-token block.
- Macro layer: causal transformer over anchors.
- Feedback: macro state is shifted into the next block to avoid future leakage.

Result:

- The design trained and generated non-random text.
- The mean-pooling anchor was likely too lossy.

## V14.1: Selective Multi-Slot Anchor

Changed the bridge from one pooled vector to multiple learned slots per block.

Hypothesis:

- A block may contain several concepts.
- Multiple slots should preserve more information than one mean vector.

Result:

- Mechanically valid, but more expensive.
- The extra slots did not clearly justify the cost at this scale.

## V14.2: Gated Delta Anchor

Changed the bridge to:

```text
anchor = mean(block) + sigmoid(delta_gate) * selective_delta(block)
```

Hypothesis:

- Mean pooling is stable but bland.
- A learned selective delta can add high-salience information without throwing away stability.

Result:

- Best practical FHA direction so far.
- 300-step eval was stronger than V14.3.
- 1500-step run reached usable held-out losses for this small training budget.

Known caveat:

- A previous baseline comparison was compromised by an implicit default that routed `anchor_slots=1` into gated-delta behavior.
- This repo fixes that with explicit `fha.anchor_type`.

## V14.3: Anchor-Predictive FHA

Added internal auxiliary loss:

```text
anchor_t -> predict anchor_{t+1}
```

Hypothesis:

- The anchor should carry predictive semantics, not only compress the current block.

Result:

- Technically works.
- 300-step held-out eval was worse than V14.2.
- Do not scale this exact objective without changing the supervision.

## Current Recommendation

Use V14.2 Gated Delta Anchor as the working FHA baseline.

Next worthwhile idea:

- Keep the Gated Delta Anchor.
- Add a better internal semantic objective that predicts next-block token distribution or key-token summary, not next anchor vector.
- Use lower weight or warmup so the auxiliary loss does not fight LM learning early.
