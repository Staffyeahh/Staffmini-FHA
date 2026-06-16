# FHA V14.2 — Teknisk Specifikation

**Model**: Staffmini-FHA V14.2 Mamba-2 Hybrid  
**Dato**: 16. juni 2026  
**Version**: 14.2-mamba2  
**Status**: Under udvikling

---

## 1. Modeloversigt

| Parameter | Værdi |
|-----------|-------|
| Total parametre | 46,003,672 (46.00M) |
| Lag | 8 |
| d_model | 384 |
| head_dim | 64 |
| Vocab størrelse | 30,294 |
| Precision | bf16 (træning) |
| Max sekvenslængde | Ubegrænset (RoPE) |

---

## 2. Arkitektur

FHA er en hierarkisk hybrid-model med tre niveauer:

1. **Micro**: Mamba-2 SSM — token-level lokal behandling
2. **Bridge**: GatedDeltaAnchor — komprimerer 16 tokens → 1 anker
3. **Macro**: Transformer med RoPE — causal attention over ankere

Informationsflowet er hierarkisk med en feedback-loop:

```
Tokens → [Micro Mamba-2] → [Anchor Compressor] → [Macro Transformer]
              ↑                                           |
              └───────── [FiLM Feedback Loop] ←───────────┘
```

---

## 3. Param-fordeling

| Komponent | Parametre | Andel | Beskrivelse |
|-----------|-----------|-------|-------------|
| Macro Transformer | 11,802,624 | 25.7% | Causal attention over ankere |
| Embedding | 11,632,896 | 25.3% | Token embedding (tied weights) |
| Micro FFN (SwiGLU) | 10,616,832 | 23.1% | Feed-forward efter micro |
| Anchor Compressor | 5,953,552 | 12.9% | GatedDeltaAnchor (16→1) |
| Micro (FlaMamba2) | 3,635,008 | 7.9% | Mamba-2 SSM (fla Triton) |
| Feedback (FiLM) | 1,182,728 | 2.6% | Macro → micro feedback |
| Anchor Predictor | 1,179,648 | 2.6% | Self-supervised anchor loss |
| RMSNorm | 384 | 0.0% | Normalisering |

---

## 4. Per-lag Struktur

Hvert af de 8 lag indeholder følgende sub-moduler:

### 4.1 Micro Mixer: FlaMamba2Mixer

Mamba-2 implementering baseret på `flash-linear-attention` (fla) biblioteket med Triton-kernels.

| Parameter | Værdi |
|-----------|-------|
| Type | Mamba-2 (SSD — Structured State Space Duality) |
| Backend | fla Triton kernels |
| d_inner | 384 (expand=1) |
| d_state | 8 |
| d_conv | 4 |
| num_heads | 8 |
| head_dim | 48 |

**Sub-moduler per lag:**

```
in_proj:   Linear(384 → 768, no bias)    ← [x_proj, z] gating
conv1d:    Conv1d(384, kernel=4, groups=384)  ← depthwise, causal
SSM:       A_log[384, 8], D[384]         ← state space parametre
out_proj:  Linear(384 → 384, no bias)
```

**Forward pass:**

```python
xz = in_proj(norm(x))           # [batch, seq, 768]
x_proj, z = xz.chunk(2)         # each [batch, seq, 384]
x_conv = silu(conv1d(pad(x_proj)))  # causal conv
# SSM selective scan (Triton)
y = selective_scan(x_conv, A, B, C, dt)
y = y + D * x_conv              # skip connection
y = y * silu(z)                 # gating
return out_proj(y)
```

**Inference (O(1) per token):**

```python
# State update: h_t = A_bar * h_{t-1} + B_bar * x_t
# Output: y_t = C_t @ h_t + D * x_t
```

### 4.2 Micro FFN: DenseSwiGLU

| Parameter | Værdi |
|-----------|-------|
| Type | SwiGLU |
| Hidden size | 1,152 (3x d_model) |
| gate_proj | Linear(384 → 1152, no bias) |
| up_proj | Linear(384 → 1152, no bias) |
| down_proj | Linear(1152 → 384, no bias) |

```python
ffn(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
```

### 4.3 Anchor Compressor: GatedDeltaAnchor

Komprimerer 16 tokens til 1 anker-vektor.

| Parameter | Værdi |
|-----------|-------|
| stride | 16 |
| temperature | 0.75 |
| position_embed | Parameter(16, 384) |
| importance | Linear(384 → 1) |
| value_proj | Linear(384 → 384) |
| mean_proj | Linear(384 → 384) |
| delta_proj | Linear(384 → 384) |
| endpoint_proj | Linear(768 → 384) |
| delta_gate | Parameter(-1.0) |

**Algoritme:**

```python
# 1. Gennemsnit af alle 16 tokens
mean = tokens.mean(dim=stride)

# 2. Importance-weightet selektion
scores = importance(norm(tokens + position_embed)) / temperature
weights = softmax(scores, mask=valid)
selected = weighted_sum(value_proj(tokens), weights)

# 3. Endpoint features (første + sidste token)
endpoint = proj([tokens[0], tokens[-1]])

# 4. Gated delta
delta = delta_proj(selected + endpoint)
anchor = mean_proj(mean) + sigmoid(delta_gate) * delta

# 5. Entropy-tracking (information collapse detection)
entropy = -(weights * log(weights)).sum()
```

### 4.4 Macro Transformer: MacroAnchorTransformer

Causal self-attention over ankere med RoPE.

| Parameter | Værdi |
|-----------|-------|
| num_heads | 6 |
| head_dim | 64 |
| d_model | 384 (6 × 64) |
| q_proj | Linear(384 → 384, no bias) |
| k_proj | Linear(384 → 384, no bias) |
| v_proj | Linear(384 → 384, no bias) |
| o_proj | Linear(384 → 384, no bias) |
| RoPE | RotaryPositionEmbedding(dim=64, theta=10000) |
| FFN | SwiGLU(384 → 768 → 384) |

**Forward pass:**

```python
normed = input_norm(anchors)
q, k, v = proj(normed).split(heads)
q, k = apply_rope(q, k, positions)  # Rotary Position Embeddings
mask = tril(q_len, k_len, diagonal=k_len - q_len)  # Causal mask
attn = scaled_dot_product_attention(q, k, v, mask=mask)
anchors = anchors + o_proj(attn)
anchors = anchors + ffn(post_attn_norm(anchors))
```

**Antal ankere ved forskellige sekvenslængder:**

| Sekvens | Ankere | Attention-størrelse |
|---------|--------|-------------------|
| 512 | 32 | 32×32 |
| 1024 | 64 | 64×64 |
| 2048 | 128 | 128×128 |
| 4096 | 256 | 256×256 |

### 4.5 Feedback: FiLM (Feature-wise Linear Modulation)

| Parameter | Værdi |
|-----------|-------|
| feedback_norm | RMSNorm(384) |
| feedback_proj | Linear(384 → 768) ← 2× d_model for (gamma, beta) |
| feedback_gate | Parameter(-2.0) |

**Mekanisme:**

```python
# _broadcast_previous_guidance:
#   Blok 0: guidance = zeros
#   Blok N: guidance = macro_states[N-1] (repeat_interleave ×16)

guidance = broadcast_previous_guidance(macro_states, seq_len)
gamma, beta = feedback_proj(feedback_norm(guidance)).chunk(2)
hidden_states = hidden_states * (1 + sigmoid(gate) * gamma) + sigmoid(gate) * beta
```

### 4.6 Anchor Predictor (Self-Supervised)

| Parameter | Værdi |
|-----------|-------|
| anchor_predictor | Linear(384 → 384, no bias) |
| weight | 0.1 |

**Loss:**

```python
pred = normalize(anchor_predictor(anchors[:, :-1]))
target = normalize(anchors[:, 1:].detach())
loss = (1 - cosine_similarity(pred, target)).mean()
```

Tvinger MacroTransformer til at lære et forudsigeligt latent rum (JEPA-inspireret).

---

## 5. RoPE (Rotary Position Embeddings)

| Parameter | Værdi |
|-----------|-------|
| dim | 64 (= head_dim) |
| theta | 10,000 |

Anvendes kun i MacroTransformer (ikke i micro-laget).

```python
inv_freq = 1.0 / (theta ** (arange(0, dim, 2) / dim))
freqs = positions * inv_freq
cos, sin = freqs.cos(), freqs.sin()
q, k = apply_rotary(q, k, cos, sin)
```

**Fordele:**
- Ubegrænset sekvenslængde (ingen max_position_embeddings)
- Relativ positionsinformation (ikke absolut)
- 1.56M færre parametre (fjernet position_embeddings)

---

## 6. Træning

### 6.1 Konfiguration

```yaml
optimizer:     AdamW (lr=4e-4, wd=0.1, betas=[0.9, 0.95])
scheduler:     WSD (warmup 3%, stable 70%, decay 27%)
precision:     bf16
seq_len:       512
micro_bs:      24
grad_accum:    3
tokens/step:   36,864
```

### 6.2 Træningsresultater (Conv1d baseline, 15K steps)

| Step | Loss | Top-1 | NIAH | Anchor-Pred | Entropy | Gate |
|------|------|-------|------|-------------|---------|------|
| 300 | 24.71 | 16.3% | 20% | - | 2.10 | 0.119 |
| 5000 | 15.45 | 26.1% | 53% | 0.24 | 1.55 | 0.119 |
| 10000 | 13.63 | 30.0% | 53% | 0.33 | 1.52 | 0.119 |
| 15000 | 12.75 | 31.9% | 47% | 0.44 | 1.53 | 0.119 |

### 6.3 Hastighed

| Konfiguration | Tid/step | tok/s |
|---------------|----------|-------|
| Conv1d (24×512) | ~4s | ~9,200 |
| Mamba-2 fla (24×512) | ~7s | ~5,300 |

---

## 7. Inference

### 7.1 Caching (3 niveauer)

**Micro-cache (per lag):**
```
ssm_state: [batch, d_inner, d_state]  ← O(1) state update
conv_buf:  [batch, d_inner, d_conv-1] ← sidste 3 tokens
```

**Anchor-buffer (per lag):**
```
buffer: [batch, buf_len, d_model]  ← akkumulerer tokens
count:  int                         ← triggers ved stride
```

**Macro KV-cache (per lag):**
```
k_cache: [batch, heads, num_anchors, head_dim]
v_cache: [batch, heads, num_anchors, head_dim]
```

### 7.2 KV-cache størrelse

| Sekvens | FHA ankere | Standard Transformer | Reduktion |
|---------|-----------|---------------------|-----------|
| 512 | 32 | 512 | 16x |
| 1024 | 64 | 1024 | 16x |
| 2048 | 128 | 2048 | 16x |

### 7.3 Inference-hastighed

```
Prefill:   O(seq_len) — hele prompten
Generate:  O(1) per token — SSM state + anchor buffer
```

---

## 8. NIAH (Needle-in-a-Haystack) Resultater

| Needle | Type | 300 steps | 15K steps | Status |
|--------|------|-----------|-----------|--------|
| ZORBLAX | Opfundet ord | +5.3 | +19.3 | ✓ Reel retrieval |
| 7749 | Tal | FAIL | +2.8-6.4 | ✓ Forbedret |
| ultraviolet | Rigtigt ord | +30 | +17-32 | ✓ Baseline |
| Aethermoor | Opfundet navn | FAIL | FAIL | ✗ Ikke lært |
| Flamethrax | Opfundet navn | FAIL | FAIL | ✗ Ikke lært |

---

## 9. Sammenligning: Conv1d vs Mamba-2

| Egenskab | Conv1d | Mamba-2 (fla) |
|----------|--------|---------------|
| Hukommelse | 5 tokens | Teoretisk uendelig |
| Inference | O(K) per token | O(1) per token |
| Params (micro) | 3.57M | 3.64M |
| Total params | 45.94M | 46.00M |
| Træningshastighed | ~4s/step | ~7s/step |
| Forward only | ~1s | ~0.4s |
| Backend | Standard PyTorch | Triton (fla) |

---

## 10. Konfigurations-reference

```yaml
# model_config_v14_2_mamba2_lite_50m.yaml
model_name: mini_fha_50m_v14_2_mamba2_lite
vocab_size: 30294
d_model: 384
n_layers: 8
head_dim: 64

fha:
  micro_type: mamba2
  mamba_expand: 1
  mamba_d_state: 8
  mamba_d_conv: 4
  anchor_type: gated_delta
  anchor_stride: 16
  anchor_slots: 1
  anchor_temperature: 0.75
  macro_heads: 6
  micro_ffn_hidden_size: 1152
  macro_ffn_hidden_size: 768
  feedback_init: -2.0
  anchor_prediction_weight: 0.1
  use_rope: true
  rope_theta: 10000
```

---

## 11. Kendte Begrænsninger

1. **Mamba-2 backward pass**: Langsom (~7s/step vs Conv1d ~4s) pga. sequential scan i autograd
2. **Triton JIT**: Første step tager 2-3 min pga. kernel-kompilering
3. **Anchor stride=16**: Hård kompression — kan miste information i korte sætninger
4. **Gate=0.119**: Statisk — overvej FiLM for dynamisk feedback

---

## 12. Fremtidige Forbedringer

| Prioritet | Ændring | Forventning |
|-----------|---------|-------------|
| 1 | FiLM feedback-loop | Bedre NIAH på komplekse sætninger |
| 2 | Vanilla Transformer baseline | Videnskabelig sammenligning |
| 3 | Mamba-2 chunked scan | Hurtigere backward pass |
| 4 | kernel_size=7 / dilation | Lidt bredere micro-kontekst |
| 5 | 100M+ params | Skalering |
