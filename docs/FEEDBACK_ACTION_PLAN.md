# FHA Feedback-Handling Plan

Kilde: Teknisk review af V14 FHA kode (juni 2026).

---

## Oversigt

| # | Opgave | Sværhed | Impact | Status |
|---|--------|---------|--------|--------|
| 1 | RoPE i MacroAnchorTransformer | Medium | Høj — nødvendigt for lang kontekst | TODO |
| 2 | Oprydning af MoE-variabler | Lav | Medium — kodekvalitet | TODO |
| 3 | Inference med caching (generate) | Svær | Høj — nødvendigt for præstation | TODO |

---

## Opgave 1: RoPE i MacroAnchorTransformer

### Problem
`FractalHybridForCausalLM` bruger absolutte positions-embeddings (`nn.Embedding(max_position_embeddings, d_model)`). Modellen fejler hvis sekvensen er længere end `max_position_embeddings` (4096). Det strider mod FHA's kerneargument om uendelig kontekst.

### Løsning
Indføj RoPE (Rotary Position Embeddings) i MacroAnchorTransformer's Q/K-projektioner. Fjern de absolutte embeddings fra topmodellen.

### Ændringer

**config.py** — Tilføj felter:
```python
# i fha-sektionen:
self.fha_use_rope = bool(fha.get("use_rope", True))
self.fha_rope_theta = float(fha.get("rope_theta", attention_layout.get("rope_theta", 10000.0)))
```

**model.py** — Ny klasse `RoPE`:
```python
class RotaryPositionEmbedding(nn.Module):
    """Standard RoPE — roterer Q og K i par af dimensioner."""
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        # q, k: [batch, heads, seq, head_dim]
        # positions: [batch, seq]
        freqs = torch.einsum("bi,d->bid", positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [batch, seq, dim]
        cos = emb.cos().to(dtype=q.dtype)
        sin = emb.sin().to(dtype=q.dtype)
        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)
        return q_rot, k_rot

    @staticmethod
    def _apply_rotary(x, cos, sin):
        d = x.shape[-1]
        x1, x2 = x[..., :d//2], x[..., d//2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return x * cos + rotated * sin
```

**model.py** — `MacroAnchorTransformer`:
- Tilføj `self.rope = RotaryPositionEmbedding(self.head_dim, config.fha_rope_theta)` hvis `config.fha_use_rope`
- I `forward()`: beregn positions og kald `self.rope(q, k, positions)` før SDPA

**model.py** — `FractalHybridForCausalLM`:
- Fjern `self.position_embeddings = nn.Embedding(...)`
- Fjern positions-additionen i `forward()`
- Opdater `_init_weights`

**configs/*.yaml** — Tilføj:
```yaml
fha:
  use_rope: true
  rope_theta: 10000
```

### Verifikation
- Kør `smoke_test.py` — logits.shape og loss skal være uændrede
- Kør med `seq_len=8192` — skal ikke fejle
- Sammenlign param-antal (færre nu)

### Faldgruber
- RoPE roterer kun halvdelen af head_dim — sørg for at head_dim er lige
- Breaking change for alle eksisterende checkpoints

---

## Opgave 2: Oprydning af MoE-variabler

### Problem
`forward()` returnerer ~12 MoE-felter der altid er None/zero. Arv fra Mixtral/Megatron.

### Løsning
Fjern alle MoE-felter fra `_Out`-dicten. Behold kun FHA-relevante felter.

### Ændringer

**model.py** — Ny return-dict:
```python
return _Out({
    "loss": loss,
    "logits": logits,
    "lm_loss": lm_loss,
    "lm_loss_per_token": lm_loss_per_token if return_per_token_losses else None,
    "fha_feedback_gate": torch.stack(feedback_gates).mean() if feedback_gates else zero,
    "fha_anchor_entropy": torch.stack(anchor_entropies).mean() if anchor_entropies else zero,
    "fha_anchor_prediction_loss": anchor_prediction_loss,
    "fha_anchor_prediction_weight": logits.new_tensor(float(self.config.fha_anchor_prediction_weight)),
})
```

Fjern også `get_expert_weight_norms_by_layer()`.

### Verifikation
- Søg i repoen efter `moe_`, `mtp_`, `router_` for afhængigheder
- Kør `smoke_test.py`

---

## Opgave 3: Inference med caching

### Problem
`generate.py` kører hele sekvensen forfra for hvert token. Ingen caching.

### Løsning — tre cache-niveauer

#### 3a. Micro-cache (CausalDepthwiseMixer)
Tilføj `step()`-metode der kun bruger de seneste K tokens via en buffer.

#### 3b. Anchor-buffer (FractalHybridBlock)
Tilføj `step()`-metode der akkumulerer tokens og udløser anchor ved stride-grænsen.

#### 3c. Macro KV-cache (MacroAnchorTransformer)
Tilføj `step()`-metode med klassisk KV-cache for attention.

#### 3d. Ny generate-funktion
- `model.prefill(ids)` — Kør prompt, returner caches
- `model.step(token, caches)` — Ét token, returner logits + opdaterede caches

### Cache-struktur per lag:
```python
@dataclass
class BlockCache:
    micro_cache: torch.Tensor | None
    anchor_buffer: torch.Tensor | None
    anchor_count: int
    macro_kv_cache: list | None
    prev_macro: torch.Tensor | None
```

### Verifikation
- Output skal matche brute-force versionen
- Mål latency: 10-50x speedup forventet
- Test med seq_len > 4096

### Faldgruber
- Causal masking i SDPA med KV-cache: `is_causal=True` er forkert for gamle keys
- bf16 accumulation i lange cache-kæder kan akkumulere fejl

---

## Anbefalet rækkefølge

1. **Opgave 2 (MoE-rydning)** — 30 min, ingen risiko
2. **Opgave 1 (RoPE)** — 2-3 timer, breaking change
3. **Opgave 3 (Inference/caching)** — 4-6 timer, afhænger af RoPE

Opgave 1 og 2 er uafhængige. Opgave 3 bør komme efter RoPE.

## Breaking changes

- Alle eksisterende checkpoints bliver ugyldige efter RoPE
- Output-dict ændres efter MoE-rydning
- generate.py API ændres (ny prefill/step interface)

Overvej at lave en legacy-gren før merging.
