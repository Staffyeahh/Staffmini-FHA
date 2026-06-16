# FHA V14.2 — Mamba-2 Integration Problemstilling

**Formål**: Sparringsdokument til ekstern teknisk review (Gemini/Claude/GPT)  
**Dato**: 16. juni 2026  
**Status**: Conv1d baseline kørt 15K steps, Mamba-2 forsøgt men for langsomt

---

## 1. Model-oversigt: Fractal Hybrid Architecture (FHA)

### 1.1 Kernekoncept

FHA er en hierarkisk hybrid-model med tre niveauer, designet til at kombinere lokal sekventiel behandling (micro) med global hierarkisk reasoning (macro):

```
Tokens → [Micro Mixer] → [Anchor Compressor] → [Macro Transformer]
              ↑                                          |
              └──────── [Feedback Loop] ←────────────────┘
```

**Idé**: Micro-laget håndterer lokal mønstergenkendelse (n-grams, syntaks). Macro-laget håndterer lang-distance semantik via komprimerede "ankere" (en vektor per 16 tokens). Feedback-loopen sender makro-kontekst tilbage til micro-niveau.

### 1.2 Arkitektur-detaljer

```
Model: mini_fha_50m
Params: 45.94M (Conv1d) / 46.16M (Mamba-2)
d_model: 384
n_layers: 8
head_dim: 64
vocab_size: 30,294
anchor_stride: 16 (hver 16. token komprimeres til 1 anker)
```

### 1.3 Niveau-beskrivelse

| Niveau | Komponent | Funktion | Params |
|--------|-----------|----------|--------|
| Micro | `CausalDepthwiseMixer` ELLER `Mamba2Mixer` | Token-level lokal behandling | 3.57M / 3.79M |
| Micro FFN | `DenseSwiGLU` | Feed-forward efter micro | 10.62M |
| Bridge | `GatedDeltaAnchor` | Komprimerer 16 tokens → 1 anker | 5.95M |
| Macro | `MacroAnchorTransformer` | Causal self-attention over ankere | 11.80M |
| Feedback | Gating + projektion | Macro → micro feedback | 1.18M |
| Embedding | `nn.Embedding` | Token embedding | 11.63M |

### 1.4 Informationsflow (per lag)

```python
# Forward pass per FractalHybridBlock:
hidden_states = x + micro(x)                          # Micro mixer
anchors, entropy = anchor_compressor(hidden_states)    # 16 tokens → 1 anchor
macro_states = macro(anchors)                          # Attention over anchors
guidance = broadcast_previous_guidance(macro_states)    # Shift right by 1
feedback = feedback_proj(feedance)
hidden_states = hidden_states + sigmoid(gate) * feedback  # Add feedback
hidden_states = hidden_states + ffn(norm(hidden_states))  # FFN
```

### 1.5 Feedback-mekanisme

`_broadcast_previous_guidance` skifter makro-tilstande én position til højre:
- Blok 0 får NUL feedback (intet foregående)
- Blok N får feedback fra blok N-1's makro-output
- Dette sikrer strengt autoregressivt flow uden data leakage

### 1.6 Nøglefeatures

1. **RoPE** (Rotary Position Embeddings) i MacroTransformer — ubegrænset sekvenslængde
2. **Anchor Prediction Loss** — self-supervised loss der tvinger modellen til at forudsige næste anker (JEPA-inspireret)
3. **GatedDeltaAnchor** — importance-weightet kompression med entropy-tracking

---

## 2. Conv1d Baseline (15K steps resultater)

### 2.1 Micro-laget: CausalDepthwiseMixer

```python
class CausalDepthwiseMixer(nn.Module):
    # Gated causal Conv1d (kernel_size=5)
    # Teknisk set en forenklet Hyena-variant, IKKE et SSM
    def forward(self, x):
        normed = self.input_norm(x)
        local = self.depthwise(F.pad(normed, (4, 0)))  # causal pad
        mixed = silu(local + value_proj(normed)) * sigmoid(gate_proj(normed))
        return out_proj(mixed)
```

**Egenskaber:**
- Fast vindue: ser kun 5 tokens tilbage
- O(K) inference per token
- ~15 linjer kode, standard PyTorch
- Meget hurtig: ~4s/step (24x512)

### 2.2 Træningsresultater

```
Step     Loss    Top-1    NIAH    Anchor-Pred   Entropy   Gate
────────────────────────────────────────────────────────────────
300      24.71   16.3%    20%     -             2.10      0.119
5000     15.45   26.1%    53%     0.24          1.55      0.119
10000    13.63   30.0%    53%     0.33          1.52      0.119
15000    12.75   31.9%    47%     0.44          1.53      0.119
```

### 2.3 NIAH (Needle-in-a-Haystack) resultater

| Needle | 300 steps | 15K steps | Analyse |
|--------|-----------|-----------|---------|
| ZORBLAX (opfundet ord) | +5.3 delta | +19.3 delta | Reel in-context retrieval ✓ |
| 7749 (tal) | FAIL | OK (seq=512) | Forbedret |
| ultraviolet (rigtigt ord) | OK | OK | Baseline probability |
| Aethermoor (opfundet navn) | FAIL | FAIL | Ikke lært endnu |

**ZORBLAX er bevis for reel retrieval**: Et opfundet ord der ikke findes i træningsdata. Modellen kan hente det fra kontekst over 512 tokens.

### 2.4 RoPE-validering

Loss falder KONSEKVENT ved længere sekvenser:
```
seq=512:  12.75
seq=1024: 12.68
seq=2048: 12.43
```
Dette bekræfter at RoPE + feedback-loopet virker korrekt.

---

## 3. Mamba-2 Problemstilling

### 3.1 Hvorfor Mamba-2?

Conv1d (kernel_size=5) har en hård begrænsning: den ser kun 5 tokens tilbage. En ægte SSM (State Space Model) komprimerer hele historikken til en uendelig skjult tilstand. For FHA's micro-lag ville dette give:

1. **Uendelig hukommelse** — micro-laget kan huske lang-distance kontekst
2. **O(1) inference** — state update i stedet for O(K) convolution
3. **Bedre scaling** — micro-laget behøver ikke stole 100% på macro for lang-distance

### 3.2 Implementering

Vi implementerede en ren PyTorch Mamba-2 baseret på SSD (Structured State Space Duality):

```python
class Mamba2Mixer(nn.Module):
    # Input projection: x → [z, x_proj] (gated)
    # Conv1d for local context (d_conv=4)
    # SSM parameters: B, C, dt (input-dependent)
    # Selective scan: h_t = A_bar * h_{t-1} + B_bar * x_t
    # Output: y = C @ h + D * x, gated by z
```

**Specs:**
- d_inner: 384 (expand=1)
- d_state: 8
- d_conv: 4
- Selective scan: Sequential Python for-loop (IKKE parallel CUDA kernel)

### 3.3 Problem: Hastigheden

```
Konfiguration          Tid/step (24x512)    300 steps
──────────────────────────────────────────────────────
Conv1d (baseline)      ~4s                  ~20 min
Mamba-2 (PyTorch)      ~15-45s              ~72-225 min
Mamba-2 (mamba-ssm)    ~4-6s (estimeret)    ~20-30 min
```

**Årsag**: Den selektive scan bruger en Python for-loop over seq_len=512:

```python
for t in range(seq_len):  # 512 iterationer
    h = dA[:, t] * h + dBx[:, t]        # CUDA kernel launch
    y = (h * C[:, t]).sum(dim=-1)        # CUDA kernel launch
```

Hver iteration launcher 2-3 CUDA kernels. Med 512 iterationer × 8 lag × 24 batch = ~100,000 kernel launches per forward pass. Overhead dominerer.

### 3.4 Hvorfor `mamba-ssm` ikke virker

```
pip install mamba-ssm → fejler:
- Kræver custom CUDA kernels (selective_scan_cuda)
- Kræver nvcc (CUDA compiler) der matcher PyTorch version
- Kræver build isolation deaktiveret
- Vores Docker container har ikke de korrekte CUDA build tools
```

### 3.5 Forsøgte løsninger

| Løsning | Status | Problem |
|---------|--------|---------|
| Sekventiel Python loop | Virker men 4-10x langsomt | CUDA kernel launch overhead |
| Parallel associative scan (Blelloch) | Fejler | In-place ops bryder autograd |
| `torch.compile` | Afbrudt | Warmup tog >5 min, uafprøvet |
| Reducere batch (4x128) | Virker (1.1s) | Ikke sammenligneligt med baseline |
| `mamba-ssm` pakke | Fejler | CUDA build tools mangler |

### 3.6 Fallback: Gradient-safe sequential scan

Vi endte med denne implementering der virker men er langsom:

```python
def _selective_scan(self, x, dt, A, B, C):
    dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
    dBx = torch.einsum("bld,bln,bld->bldn", dt, B, x)
    
    # Gradient-safe: float32, ingen in-place
    h = torch.zeros(batch, d_inner, d_state, dtype=torch.float32)
    ys = []
    for t in range(seq_len):
        h = dA_f[:, t] * h + dBx_f[:, t]
        ys.append((h * C_f[:, t]).sum(dim=-1))
    return torch.stack(ys, dim=1).to(dtype=x.dtype)
```

---

## 4. Spørgsmål til Ekstern Review

### 4.1 Arkitektur-spørgsmål

1. **Er SSD den rigtige vej for micro-laget?** Conv1d virker overraskende godt (ZORBLAX-retrieval). Er Mamba-2 overkill for et micro-lag der primært skal håndtere lokal kontekst?

2. **Feedback-loopen**: Vi sender macro-output tilbage til micro via en enkelt gate (0.119). Er der smartere måder at integrere hierarkisk information?

3. **Anchor-kompression**: GatedDeltaAnchor med stride=16. Er 16 for aggressivt? Ville 8 eller 32 være bedre?

### 4.2 Implementerings-spørgsmål

4. **Parallel scan**: Vi forsøgte Blelloch-style associative scan men in-place ops bryder autograd. Hvordan løser man dette korrekt i PyTorch?

5. **torch.compile**: Kan det hjælpe med den sekventielle scan? Vi nåede aldrig at få det til at virke (warmup >5 min).

6. **Flash-linear-attention**: Er der pakker der implementerer Mamba-2 med flash-attention-lignende CUDA kernels der VIRKER i PyTorch 2.12?

### 4.3 Trænings-spørgsmål

7. **Mamba-2 vs Conv1d baseline**: Hvis vi kun kan køre 300 steps med Mamba-2 (langsomt), er det overhovedet en fair sammenligning med 15K steps Conv1d?

8. **Hybrid tilgang**: Kunne vi bruge Conv1d i de første 4 lag og Mamba-2 i de sidste 4? Sparer halvdelen af scan-overhead.

9. **KV-cache for Mamba-2**: Vores step() funktion opdaterer SSM-state O(1) per token. Er dette korrekt for inference, eller mangler vi noget?

### 4.4 Videnskabelige spørgsmål

10. **Hvad mangler vi?** Vi har 45M params, 750M tokens, Conv1d baseline med ZORBLAX-retrieval. Hvad er det næste vigtige eksperiment?

---

## 5. Filer og Struktur

```
src/staffmini_fha/
  model.py          Hovedmodel (807 linjer)
  config.py         Konfiguration (105 linjer)
  data.py           Dataset loader

configs/
  model_config_v14_2_gated_delta_anchor_50m.yaml   Conv1d baseline
  model_config_v14_2_mamba2_lite_50m.yaml           Mamba-2 variant
  train_config_300.yaml                             300 steps
  train_config_3000.yaml                            3000 steps
  train_config_20k_phase{A,B,C}.yaml                15K steps (3 faser)

scripts/
  train.py          Træningsscript
  generate.py       Inference med caching
  eval_causal_suite.py   Loss-eval ved forskellige sekvenslængder
  niah_test.py      Needle-in-a-Haystack test

outputs/
  v14_2_conv1d_baseline_3k/    Conv1d 3K steps
  v14_2_baseline_20k/          Conv1d 15K steps
  v14_2_mamba2_300/            (ikke gennemført)
```

---

## 6. Kontakt

Repo: https://github.com/Staffyeahh/Staffmini-FHA.git  
Model: 45-46M params, 8 lag, d_model=384  
Træning: Cosmopedia 100M tokens, bf16, RTX 5060 Ti 16GB
