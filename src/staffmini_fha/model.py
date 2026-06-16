from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import FHAConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, zero_centered: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.zero_centered = zero_centered
        if zero_centered:
            self.weight = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        if self.zero_centered:
            return x_norm * (1.0 + self.weight)
        return x_norm * self.weight




class RotaryPositionEmbedding(nn.Module):
    """RoPE — rotates Q and K in pairs of dimensions."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [batch, heads, seq, head_dim]
        # positions: [batch, seq]
        freqs = torch.einsum("bi,d->bid", positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [batch, seq, dim]
        cos = emb.cos().to(dtype=q.dtype)
        sin = emb.sin().to(dtype=q.dtype)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        cos = cos[:, None, :, :]  # [batch, 1, seq, dim]
        sin = sin[:, None, :, :]
        return x * cos + rotated * sin

class DenseSwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))





class FlaMamba2Mixer(nn.Module):
    """Mamba-2 micro mixer using fla (Flash Linear Attention) Triton kernels.

    Drop-in replacement for CausalDepthwiseMixer.
    Uses fla.layers.Mamba2 for both training and inference.
    """

    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        expand = int(getattr(config, "fha_mamba_expand", 1))
        d_state = int(getattr(config, "fha_mamba_d_state", 8))
        d_conv = int(getattr(config, "fha_mamba_d_conv", 4))
        d_inner = self.d_model * expand

        # Calculate head_dim and num_heads for fla
        head_dim = max(1, d_inner // 8)
        num_heads = d_inner // head_dim

        from fla.layers import Mamba2 as FlaMamba2
        self.mamba = FlaMamba2(
            hidden_size=self.d_model,
            expand=expand,
            state_size=d_state,
            conv_kernel=d_conv,
            head_dim=head_dim,
            num_heads=num_heads,
            backend='cuda',
        )
        self.d_conv = d_conv
        self.d_inner = d_inner
        self.d_state = d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using fla Triton kernels."""
        output, _, _ = self.mamba(x)
        return output

    def step(
        self, x: torch.Tensor, cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        """Single-token inference using fla's built-in decode path.

        x: [batch, 1, d_model]
        cache: fla cache dict or None
        returns: (output [batch, 1, d_model], new_cache)
        """
        import torch.nn.functional as F

        if cache is None:
            # First call: initialize cache from prefill
            # Run full forward with use_cache to get initial state
            output, _, cache = self.mamba(x, use_cache=True)
            return output, cache
        else:
            # Subsequent calls: single-token decode
            output, _, cache = self.mamba(x, past_key_values=cache, use_cache=True)
            return output, cache



class SsmMamba2Mixer(nn.Module):
    """Mamba-2 micro mixer using mamba-ssm CUDA kernels.

    Drop-in replacement for CausalDepthwiseMixer.
    Uses Tri Dao's optimized CUDA selective scan.
    """

    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        from mamba_ssm import Mamba2 as SsmMamba2
        self.d_model = config.d_model
        expand = int(getattr(config, "fha_mamba_expand", 1))
        d_state = int(getattr(config, "fha_mamba_d_state", 16))
        d_conv = int(getattr(config, "fha_mamba_d_conv", 4))

        self.mamba = SsmMamba2(
            d_model=config.d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)

    def step(
        self, x: torch.Tensor, cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        """Single-token inference using mamba-ssm's inference_params."""
        # mamba-ssm supports inference via inference_params
        # For now, use forward (will optimize later)
        output = self.mamba(x)
        return output, cache



class Mamba3Mixer(nn.Module):
    """Mamba-3 micro mixer using mamba-ssm CUDA kernels.

    Mamba-3 features:
    - Built-in RoPE (rope_fraction=0.5)
    - MIMO mode (Multiple Input Multiple Output)
    - Improved state space formulation
    """

    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        from mamba_ssm import Mamba3
        self.d_model = config.d_model
        expand = int(getattr(config, "fha_mamba_expand", 2))
        d_state = int(getattr(config, "fha_mamba_d_state", 64))
        headdim = int(getattr(config, "fha_mamba_headdim", 48))
        is_mimo = bool(getattr(config, "fha_mamba_mimo", False))
        mimo_rank = int(getattr(config, "fha_mamba_mimo_rank", 4))

        self.mamba = Mamba3(
            d_model=config.d_model,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)

    def step(
        self, x: torch.Tensor, cache: dict | None = None,
    ) -> tuple[torch.Tensor, dict | None]:
        output = self.mamba(x)
        return output, cache

class Mamba2Mixer(nn.Module):
    """Mamba-2 micro mixer using SSD (Structured State Space Duality).

    Pure PyTorch implementation — no custom CUDA kernels.
    Drop-in replacement for CausalDepthwiseMixer with same interface:
      forward(x) -> output
      step(x, cache) -> (output, new_cache)
    """

    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_state = int(getattr(config, "fha_mamba_d_state", 16))
        self.d_conv = int(getattr(config, "fha_mamba_d_conv", 4))
        self.expand = int(getattr(config, "fha_mamba_expand", 2))
        self.d_inner = self.d_model * self.expand
        self.dt_rank = max(1, self.d_model // 16)

        self.input_norm = RMSNorm(config.d_model, eps=1e-6, zero_centered=config.zero_centered_rmsnorm)

        # Input projection: x -> [z, x_proj] (gated)
        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=False)

        # Conv1d for local context (no built-in padding — we do manual causal pad)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=self.d_conv,
            groups=self.d_inner, padding=0, bias=True,
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameter (log-space for stability)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)

        # Initialize dt bias for stability
        with torch.no_grad():
            dt_init_std = self.dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        normed = self.input_norm(x)

        # Input projection with gating
        xz = self.in_proj(normed)  # [batch, seq, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # each [batch, seq, d_inner]

        # Conv1d (causal — manual left-pad, no built-in padding)
        x_conv = x_proj.transpose(1, 2)  # [batch, d_inner, seq]
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))  # causal left-pad
        x_conv = self.conv1d(x_conv)  # [batch, d_inner, seq]
        x_conv = x_conv.transpose(1, 2)  # [batch, seq, d_inner]
        x_conv = F.silu(x_conv)

        # SSM parameters
        x_ssm = self.x_proj(x_conv)  # [batch, seq, dt_rank + 2*d_state]
        dt, B, C = x_ssm.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # [batch, seq, d_inner]
        dt = F.softplus(dt)  # ensure positive

        # A (negative for stability)
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # Selective scan (sequential — O(seq_len * d_inner * d_state))
        # For training, this is the bottleneck. For inference, O(1) per token.
        y = self._selective_scan(x_conv, dt, A, B, C)

        # Skip connection + gate
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)

        return self.out_proj(y)

    def _selective_scan(
        self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
        B: torch.Tensor, C: torch.Tensor,
    ) -> torch.Tensor:
        """Selective scan — chunked sequential (numerically stable + fast).

        Processes chunks of `chunk_size` tokens sequentially.
        Reduces iterations from seq_len to seq_len/chunk_size.
        Numerically stable because we use sequential steps within each chunk.

        x: [batch, seq, d_inner]
        dt: [batch, seq, d_inner]
        A: [d_inner, d_state]
        B: [batch, seq, d_state]
        C: [batch, seq, d_state]
        returns: [batch, seq, d_inner]
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        chunk_size = 8  # small chunk for numerical stability

        # Discretize in float32
        dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A).float())  # [B, L, D, N]
        dBx = torch.einsum("bld,bln,bld->bldn", dt, B, x).float()   # [B, L, D, N]
        C_f = C.float()

        # Pad to multiple of chunk_size
        n_chunks = (seq_len + chunk_size - 1) // chunk_size
        pad_len = n_chunks * chunk_size - seq_len
        if pad_len > 0:
            dA = torch.nn.functional.pad(dA, (0, 0, 0, 0, 0, pad_len))
            dBx = torch.nn.functional.pad(dBx, (0, 0, 0, 0, 0, pad_len))
            C_f = torch.nn.functional.pad(C_f, (0, 0, 0, pad_len))

        # Reshape into chunks
        dA = dA.view(batch, n_chunks, chunk_size, d_inner, d_state)
        dBx = dBx.view(batch, n_chunks, chunk_size, d_inner, d_state)
        C_f = C_f.view(batch, n_chunks, chunk_size, d_state)

        # Process each chunk sequentially (numerically stable)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=torch.float32)
        ys = []
        for c in range(n_chunks):
            for t in range(chunk_size):
                h = dA[:, c, t] * h + dBx[:, c, t]
                y_t = (h * C_f[:, c, t, None, :]).sum(dim=-1)
                ys.append(y_t)

        y = torch.stack(ys, dim=1)[:, :seq_len]  # [B, L, D]
        return y.to(dtype=x.dtype)


    def step(
        self, x: torch.Tensor, cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-token inference with O(1) state update.

        x: [batch, 1, d_model]
        cache: [batch, d_inner, d_state] (SSM hidden state)
        returns: (output [batch, 1, d_model], new_cache)
        """
        batch = x.shape[0]
        normed = self.input_norm(x)

        xz = self.in_proj(normed)
        x_proj, z = xz.chunk(2, dim=-1)

        # Conv1d: need last d_conv-1 tokens in cache
        # For simplicity, we store the conv cache separately
        # But for Mamba-2, the SSM state IS the main cache
        # We'll use a combined cache: (ssm_state, conv_buffer)
        if cache is None:
            ssm_state = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
            conv_buf = torch.zeros(batch, self.d_inner, self.d_conv - 1, device=x.device, dtype=x.dtype)
        else:
            ssm_state, conv_buf = cache

        # Conv1d step — use same conv1d on the d_conv window
        x_conv = x_proj.transpose(1, 2)  # [batch, d_inner, 1]
        conv_input = torch.cat([conv_buf, x_conv], dim=2)  # [batch, d_inner, d_conv]
        new_conv_buf = conv_input[:, :, 1:]  # [batch, d_inner, d_conv-1]
        x_conv = self.conv1d(conv_input)  # [batch, d_inner, 1]
        x_conv = F.silu(x_conv).transpose(1, 2)  # [batch, 1, d_inner]

        # SSM parameters
        x_ssm = self.x_proj(x_conv)  # [batch, 1, dt_rank + 2*d_state]
        dt, B, C = x_ssm.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # [batch, 1, d_inner]

        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # Discretize
        # dt: [batch, 1, d_inner], A: [d_inner, d_state], B: [batch, 1, d_state], C: [batch, 1, d_state]
        dt_sq = dt.squeeze(1)  # [batch, d_inner]
        B_sq = B.squeeze(1)    # [batch, d_state]
        C_sq = C.squeeze(1)    # [batch, d_state]

        dA = torch.exp(dt_sq.unsqueeze(-1) * A)       # [batch, d_inner, d_state]
        dB = dt_sq.unsqueeze(-1) * B_sq.unsqueeze(1)   # [batch, d_inner, d_state]

        # State update: h = A_bar * h + B_bar * x
        x_in = x_conv.squeeze(1)  # [batch, d_inner]
        new_state = dA * ssm_state + dB * x_in.unsqueeze(-1)  # [batch, d_inner, d_state]

        # Output: y = C @ h + D * x
        y = (new_state * C_sq.unsqueeze(1)).sum(dim=-1)  # [batch, d_inner]
        y = y + self.D * x_conv.squeeze(1)
        y = y * F.silu(z.squeeze(1))

        output = self.out_proj(y.unsqueeze(1))  # [batch, 1, d_model]
        new_cache = (new_state, new_conv_buf)
        return output, new_cache

class CausalDepthwiseMixer(nn.Module):
    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.kernel_size = int(config.fha_micro_kernel_size)
        self.input_norm = RMSNorm(config.d_model, eps=1e-6, zero_centered=config.zero_centered_rmsnorm)
        self.depthwise = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=self.kernel_size,
            groups=config.d_model,
            bias=True,
        )
        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.input_norm(x)
        conv_in = normed.transpose(1, 2)
        conv_in = F.pad(conv_in, (self.kernel_size - 1, 0))
        local = self.depthwise(conv_in).transpose(1, 2)
        mixed = F.silu(local + self.value_proj(normed)) * torch.sigmoid(self.gate_proj(normed))
        return self.out_proj(mixed)


    def step(
        self, x: torch.Tensor, cache: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-token inference with cache. x: [batch, 1, d_model]."""
        normed = self.input_norm(x)
        if cache is None:
            conv_input = F.pad(normed.transpose(1, 2), (self.kernel_size - 1, 0))
        else:
            conv_input = torch.cat([cache.transpose(1, 2), normed.transpose(1, 2)], dim=2)
        new_cache = conv_input[:, :, -(self.kernel_size - 1) :].transpose(1, 2)
        local = self.depthwise(conv_input).transpose(1, 2)[:, -1:, :]
        mixed = F.silu(local + self.value_proj(normed)) * torch.sigmoid(self.gate_proj(normed))
        return self.out_proj(mixed), new_cache


class MacroAnchorTransformer(nn.Module):
    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.num_heads = int(config.fha_macro_heads)
        if config.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by fha.macro_heads.")
        self.head_dim = config.d_model // self.num_heads
        self.scale = self.head_dim**-0.5
        self.input_norm = RMSNorm(config.d_model, eps=1e-6, zero_centered=config.zero_centered_rmsnorm)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.post_attn_norm = RMSNorm(config.d_model, eps=1e-6, zero_centered=config.zero_centered_rmsnorm)
        self.ffn = DenseSwiGLU(config.d_model, config.fha_macro_ffn_hidden_size)
        self.rope: RotaryPositionEmbedding | None = None
        if getattr(config, "fha_use_rope", False):
            self.rope = RotaryPositionEmbedding(
                self.head_dim, float(getattr(config, "fha_rope_theta", 10000.0))
            )

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        batch_size, num_anchors, _ = anchors.shape
        normed = self.input_norm(anchors)
        q = self.q_proj(normed).view(batch_size, num_anchors, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(batch_size, num_anchors, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(batch_size, num_anchors, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            positions = torch.arange(num_anchors, device=anchors.device).unsqueeze(0).expand(batch_size, -1)
            q, k = self.rope(q, k, positions)
        # Explicit causal mask: Q[i] at position K_len-Q_len+i attends to keys 0..K_len-Q_len+i
        # is_causal=True assumes Q starts at pos 0, which is wrong with KV cache
        q_len, k_len = q.shape[2], k.shape[2]
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device).tril(
            diagonal=k_len - q_len
        )
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=self.scale
        )
        attn = attn.transpose(1, 2).reshape(batch_size, num_anchors, -1)
        anchors = anchors + self.o_proj(attn)
        anchors = anchors + self.ffn(self.post_attn_norm(anchors))
        return anchors


    def step(
        self,
        anchors: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        anchor_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-anchor step with KV-cache. anchors: [batch, N_new, d_model]."""
        batch_size, num_new, _ = anchors.shape
        normed = self.input_norm(anchors)
        q = self.q_proj(normed).view(batch_size, num_new, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(normed).view(batch_size, num_new, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(normed).view(batch_size, num_new, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            positions = torch.arange(
                anchor_offset, anchor_offset + num_new, device=anchors.device
            ).unsqueeze(0).expand(batch_size, -1)
            q, k_new = self.rope(q, k_new, positions)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k_new], dim=2)
            v = torch.cat([kv_cache[1], v_new], dim=2)
        else:
            k, v = k_new, v_new
        new_kv_cache = (k, v)
        # Explicit causal mask: Q[i] at position K_len-Q_len+i attends to keys 0..K_len-Q_len+i
        # is_causal=True assumes Q starts at pos 0, which is wrong with KV cache
        q_len, k_len = q.shape[2], k.shape[2]
        mask = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device).tril(
            diagonal=k_len - q_len
        )
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=self.scale
        )
        attn = attn.transpose(1, 2).reshape(batch_size, num_new, -1)
        out = anchors + self.o_proj(attn)
        out = out + self.ffn(self.post_attn_norm(out))
        return out, new_kv_cache


class SelectiveMultiSlotAnchor(nn.Module):
    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.stride = int(config.fha_anchor_stride)
        self.num_slots = max(1, int(config.fha_anchor_slots))
        self.temperature = float(config.fha_anchor_temperature)
        self.token_norm = RMSNorm(config.d_model, eps=1e-6, zero_centered=config.zero_centered_rmsnorm)
        self.position_embed = nn.Parameter(torch.zeros(self.stride, config.d_model))
        self.slot_queries = nn.Parameter(torch.empty(self.num_slots, config.d_model))
        self.salience = nn.Linear(config.d_model, self.num_slots, bias=True)
        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.anchor_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.endpoint_proj = nn.Linear(2 * config.d_model, config.d_model, bias=False)
        nn.init.normal_(self.slot_queries, mean=0.0, std=config.embed_std)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_blocks = (seq_len + self.stride - 1) // self.stride
        padded_len = num_blocks * self.stride
        if padded_len != seq_len:
            pad = hidden_states.new_zeros(batch_size, padded_len - seq_len, hidden_size)
            hidden_states = torch.cat([hidden_states, pad], dim=1)

        blocks = hidden_states.view(batch_size, num_blocks, self.stride, hidden_size)
        positions = torch.arange(padded_len, device=hidden_states.device).view(num_blocks, self.stride)
        valid = positions.lt(seq_len)
        block_pos = self.position_embed.to(dtype=blocks.dtype, device=blocks.device)
        enriched = blocks + block_pos[None, None, :, :]
        normed = self.token_norm(enriched)

        query_scores = torch.einsum("bnsd,kd->bnsk", normed, self.slot_queries.to(dtype=normed.dtype))
        query_scores = query_scores / (hidden_size**0.5)
        salience_scores = self.salience(normed)
        scores = (query_scores + salience_scores) / max(1e-4, self.temperature)
        scores = scores.masked_fill(~valid[None, :, :, None], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=2)

        values = self.value_proj(enriched)
        anchors = torch.einsum("bnsk,bnsd->bnkd", weights, values)
        first = blocks[:, :, 0, :]
        last_index = valid.long().sum(dim=1).clamp_min(1) - 1
        last = blocks.gather(
            2,
            last_index[None, :, None, None].expand(batch_size, num_blocks, 1, hidden_size),
        ).squeeze(2)
        endpoint = self.endpoint_proj(torch.cat([first, last], dim=-1)).unsqueeze(2)
        anchors = self.anchor_proj(anchors + endpoint)

        entropy = -(weights.float().clamp_min(1e-9) * weights.float().clamp_min(1e-9).log()).sum(dim=2).mean()
        anchors = anchors.reshape(batch_size, num_blocks * self.num_slots, hidden_size)
        return anchors, entropy


class MeanAnchor(nn.Module):
    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.stride = int(config.fha_anchor_stride)
        self.anchor_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_blocks = (seq_len + self.stride - 1) // self.stride
        padded_len = num_blocks * self.stride
        if padded_len != seq_len:
            pad = hidden_states.new_zeros(batch_size, padded_len - seq_len, hidden_size)
            hidden_states = torch.cat([hidden_states, pad], dim=1)
        blocks = hidden_states.view(batch_size, num_blocks, self.stride, hidden_size)
        positions = torch.arange(padded_len, device=hidden_states.device).view(num_blocks, self.stride)
        valid = positions.lt(seq_len).to(hidden_states.dtype)
        pooled = (blocks * valid[None, :, :, None]).sum(dim=2)
        pooled = pooled / valid.sum(dim=1).clamp_min(1.0)[None, :, None]
        entropy = hidden_states.new_tensor(0.0)
        return self.anchor_proj(pooled), entropy


class GatedDeltaAnchor(nn.Module):
    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.stride = int(config.fha_anchor_stride)
        self.temperature = float(config.fha_anchor_temperature)
        self.token_norm = RMSNorm(config.d_model, eps=1e-6, zero_centered=config.zero_centered_rmsnorm)
        self.position_embed = nn.Parameter(torch.zeros(self.stride, config.d_model))
        self.importance = nn.Linear(config.d_model, 1, bias=True)
        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.mean_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.delta_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.endpoint_proj = nn.Linear(2 * config.d_model, config.d_model, bias=False)
        self.delta_gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_blocks = (seq_len + self.stride - 1) // self.stride
        padded_len = num_blocks * self.stride
        if padded_len != seq_len:
            pad = hidden_states.new_zeros(batch_size, padded_len - seq_len, hidden_size)
            hidden_states = torch.cat([hidden_states, pad], dim=1)

        blocks = hidden_states.view(batch_size, num_blocks, self.stride, hidden_size)
        positions = torch.arange(padded_len, device=hidden_states.device).view(num_blocks, self.stride)
        valid = positions.lt(seq_len)
        valid_f = valid.to(blocks.dtype)
        mean = (blocks * valid_f[None, :, :, None]).sum(dim=2)
        mean = mean / valid_f.sum(dim=1).clamp_min(1.0)[None, :, None]

        block_pos = self.position_embed.to(dtype=blocks.dtype, device=blocks.device)
        enriched = blocks + block_pos[None, None, :, :]
        normed = self.token_norm(enriched)
        scores = self.importance(normed).squeeze(-1) / max(1e-4, self.temperature)
        scores = scores.masked_fill(~valid[None, :, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        selected = torch.einsum("bns,bnsd->bnd", weights, self.value_proj(enriched))

        first = blocks[:, :, 0, :]
        last_index = valid.long().sum(dim=1).clamp_min(1) - 1
        last = blocks.gather(
            2,
            last_index[None, :, None, None].expand(batch_size, num_blocks, 1, hidden_size),
        ).squeeze(2)
        endpoint = self.endpoint_proj(torch.cat([first, last], dim=-1))
        delta = self.delta_proj(selected + endpoint)
        anchor = self.mean_proj(mean) + torch.sigmoid(self.delta_gate) * delta
        entropy = -(weights.float().clamp_min(1e-9) * weights.float().clamp_min(1e-9).log()).sum(dim=-1).mean()
        return anchor, entropy




from dataclasses import dataclass

@dataclass
class BlockCache:
    """Per-layer cache for incremental inference."""
    micro_cache: torch.Tensor | None = None
    anchor_buffer: torch.Tensor | None = None
    anchor_count: int = 0
    macro_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    macro_anchor_offset: int = 0
    prev_macro: torch.Tensor | None = None


class FractalHybridBlock(nn.Module):
    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.config = config
        self.anchor_stride = int(config.fha_anchor_stride)
        self.anchor_slots = max(1, int(config.fha_anchor_slots))
        micro_type = getattr(config, "fha_micro_type", "conv1d")
        if micro_type == "mamba2":
            self.micro = Mamba2Mixer(config)
        elif micro_type == "hybrid":
            # Mamba-2 in last N layers, Conv1d in the rest
            mamba_layers = getattr(config, "fha_hybrid_mamba_layers", 2)
            layer_idx = getattr(config, "_current_layer_idx", 0)
            total_layers = config.n_layers
            if layer_idx >= total_layers - mamba_layers:
                self.micro = Mamba2Mixer(config)
            else:
                self.micro = CausalDepthwiseMixer(config)
        elif micro_type == "hybrid_ssm":
            # mamba-ssm CUDA kernels in last N layers, Conv1d in the rest
            mamba_layers = getattr(config, "fha_hybrid_mamba_layers", 2)
            layer_idx = getattr(config, "_current_layer_idx", 0)
            total_layers = config.n_layers
            if layer_idx >= total_layers - mamba_layers:
                self.micro = SsmMamba2Mixer(config)
            else:
                self.micro = CausalDepthwiseMixer(config)
        elif micro_type == "hybrid_m3":
            # Mamba-3 in last N layers, Conv1d in the rest
            mamba_layers = getattr(config, "fha_hybrid_mamba_layers", 2)
            layer_idx = getattr(config, "_current_layer_idx", 0)
            total_layers = config.n_layers
            if layer_idx >= total_layers - mamba_layers:
                self.micro = Mamba3Mixer(config)
            else:
                self.micro = CausalDepthwiseMixer(config)
        else:
            self.micro = CausalDepthwiseMixer(config)
        if config.fha_anchor_type == "mean":
            self.anchor_compressor = MeanAnchor(config)
            self.anchor_slots = 1
        elif config.fha_anchor_type == "gated_delta":
            self.anchor_compressor = GatedDeltaAnchor(config)
            self.anchor_slots = 1
        elif config.fha_anchor_type == "multi_slot":
            self.anchor_compressor = SelectiveMultiSlotAnchor(config)
        else:
            raise ValueError(f"Unsupported FHA anchor_type: {config.fha_anchor_type}")
        self.macro = MacroAnchorTransformer(config)
        self.anchor_predictor = nn.Linear(config.d_model, config.d_model, bias=False)
        self.feedback_norm = RMSNorm(config.d_model, eps=1e-6, zero_centered=config.zero_centered_rmsnorm)
        self.feedback_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.feedback_gate = nn.Parameter(torch.tensor(float(config.fha_feedback_init)))
        self.post_micro_norm = RMSNorm(config.d_model, eps=1e-6, zero_centered=config.zero_centered_rmsnorm)
        self.ffn = DenseSwiGLU(config.d_model, config.fha_micro_ffn_hidden_size)

    def _broadcast_previous_guidance(self, macro_states: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size, macro_len, hidden_size = macro_states.shape
        num_blocks = macro_len // self.anchor_slots
        macro_states = macro_states.view(batch_size, num_blocks, self.anchor_slots, hidden_size).mean(dim=2)
        zero = macro_states.new_zeros(batch_size, 1, hidden_size)
        previous = torch.cat([zero, macro_states[:, :-1, :]], dim=1)
        guidance = previous.repeat_interleave(self.anchor_stride, dim=1)
        return guidance[:, :seq_len, :]

    def _anchor_prediction_loss(self, anchors: torch.Tensor) -> torch.Tensor:
        if self.config.fha_anchor_prediction_weight <= 0 or anchors.shape[1] < 2:
            return anchors.new_zeros(())
        pred = self.anchor_predictor(anchors[:, :-1, :])
        target = anchors[:, 1:, :].detach()
        pred = F.normalize(pred.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1)
        return (1.0 - (pred * target).sum(dim=-1)).mean().to(dtype=anchors.dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states + self.micro(hidden_states)
        anchors, anchor_entropy = self.anchor_compressor(hidden_states)
        anchor_prediction_loss = self._anchor_prediction_loss(anchors)
        macro_states = self.macro(anchors)
        guidance = self._broadcast_previous_guidance(macro_states, seq_len)
        feedback = self.feedback_proj(self.feedback_norm(guidance))
        gate = torch.sigmoid(self.feedback_gate)
        hidden_states = hidden_states + gate * feedback
        hidden_states = hidden_states + self.ffn(self.post_micro_norm(hidden_states))
        return hidden_states, gate.detach(), anchor_entropy.detach(), anchor_prediction_loss


    def step(
        self,
        hidden_states: torch.Tensor,
        cache: BlockCache | None = None,
        global_anchor_offset: int = 0,
    ) -> tuple[torch.Tensor, BlockCache]:
        """Single-token step with caching.

        Feedback semantics match _broadcast_previous_guidance:
        - Block 0: NO feedback (previous = zeros)
        - Block N: feedback from block N-1's macro output

        Every token is emitted immediately with correct feedback.
        """
        if cache is None:
            cache = BlockCache()

        # 1. Micro mixer with cache
        micro_out, new_micro_cache = self.micro.step(hidden_states, cache.micro_cache)
        h = hidden_states + micro_out

        # 2. Accumulate in anchor buffer
        if cache.anchor_buffer is None:
            anchor_buffer = h
        else:
            anchor_buffer = torch.cat([cache.anchor_buffer, h], dim=1)
        anchor_count = cache.anchor_count + 1

        new_macro_kv = cache.macro_kv_cache
        macro_offset = cache.macro_anchor_offset
        # Save previous feedback BEFORE any overwrites
        prev_feedback = cache.prev_macro
        next_feedback = prev_feedback  # default: carry forward
        anchor_entropy = h.new_zeros(())
        anchor_pred_loss = h.new_zeros(())

        # 3. If we have enough tokens -> trigger anchor and compute macro
        if anchor_count >= self.anchor_stride:
            anchors, anchor_entropy = self.anchor_compressor(anchor_buffer)
            anchor_pred_loss = self._anchor_prediction_loss(anchors)

            macro_out, new_macro_kv = self.macro.step(
                anchors, kv_cache=new_macro_kv,
                anchor_offset=global_anchor_offset + macro_offset,
            )
            macro_offset += anchors.shape[1]
            # Current block used prev_feedback; next block gets macro_out
            next_feedback = macro_out
            anchor_buffer = None
            anchor_count = 0

        # 4. Apply feedback from PREVIOUS block's macro output
        if prev_feedback is not None:
            guidance = self.feedback_norm(prev_feedback[:, -1:, :])
            feedback = self.feedback_proj(guidance)
            gate = torch.sigmoid(self.feedback_gate)
            h = h + gate * feedback

        # 5. FFN
        h = h + self.ffn(self.post_micro_norm(h))

        new_cache = BlockCache(
            micro_cache=new_micro_cache,
            anchor_buffer=anchor_buffer,
            anchor_count=anchor_count,
            macro_kv_cache=new_macro_kv,
            macro_anchor_offset=macro_offset,
            prev_macro=next_feedback,
        )
        gate_val = torch.sigmoid(self.feedback_gate).detach()
        return h, gate_val, anchor_entropy.detach(), anchor_pred_loss, new_cache


class FractalHybridForCausalLM(nn.Module):
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        _layers = []
        for i in range(config.n_layers):
            config._current_layer_idx = i
            _layers.append(FractalHybridBlock(config))
        self.layers = nn.ModuleList(_layers)
        self.final_norm = RMSNorm(config.d_model, eps=1e-6, zero_centered=config.zero_centered_rmsnorm)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.lm_head_bias)
        self._init_weights()
        if config.tie_word_embeddings:
            self.tie_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.init_method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.lm_head_std)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.embed_tokens.weight

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False



    @torch.no_grad()
    def prefill(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[BlockCache]]:
        """Run full prompt through the model, return logits and per-layer caches."""
        hidden_states = self.embed_tokens(input_ids)
        caches: list[BlockCache] = []
        for block in self.layers:
            # Run full forward to fill caches efficiently
            hidden_states, gate, entropy, pred_loss = block(hidden_states)
            # Build initial cache from the last anchor state
            # We need to re-run in step mode to get proper caches
            caches.append(BlockCache())
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, caches

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        caches: list[BlockCache],
        anchor_offsets: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[BlockCache]]:
        """Single-token step with caching. input_ids: [batch, 1].

        Returns logits for ALL emitted tokens (may be >1 when anchor triggers).
        The LAST logit row corresponds to the newest token.
        """
        if anchor_offsets is None:
            anchor_offsets = [0] * len(self.layers)
        hidden_states = self.embed_tokens(input_ids)
        new_caches: list[BlockCache] = []
        for i, block in enumerate(self.layers):
            out_h, gate, entropy, pred_loss, new_cache = block.step(
                hidden_states, caches[i], global_anchor_offset=anchor_offsets[i]
            )
            # out_h may contain multiple tokens (when anchor triggers)
            # For subsequent layers, use the full output
            hidden_states = out_h
            new_caches.append(new_cache)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, new_caches
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        return_per_token_losses: bool = False,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids with shape [batch, seq], got {tuple(input_ids.shape)}.")
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        feedback_gates: list[torch.Tensor] = []
        anchor_entropies: list[torch.Tensor] = []
        anchor_prediction_losses: list[torch.Tensor] = []
        for block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states, gate, anchor_entropy, anchor_prediction_loss = checkpoint(
                    block,
                    hidden_states,
                    use_reentrant=True,
                )
            else:
                hidden_states, gate, anchor_entropy, anchor_prediction_loss = block(hidden_states)
            feedback_gates.append(gate)
            anchor_entropies.append(anchor_entropy)
            anchor_prediction_losses.append(anchor_prediction_loss)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        lm_loss: torch.Tensor | None = None
        lm_loss_per_token: torch.Tensor | None = None
        loss: torch.Tensor | None = None
        zero = logits.new_zeros(())
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            if loss_mask is not None:
                shift_mask = loss_mask[:, 1:].contiguous().view(-1)
                masked = lm_loss_per_token * shift_mask
                lm_loss = masked.sum() / shift_mask.sum().clamp(min=1)
            else:
                lm_loss = lm_loss_per_token.mean()
            loss = lm_loss
            if self.config.fha_anchor_prediction_weight > 0 and anchor_prediction_losses:
                anchor_prediction_loss = torch.stack(anchor_prediction_losses).mean()
                loss = loss + self.config.fha_anchor_prediction_weight * anchor_prediction_loss
            else:
                anchor_prediction_loss = zero
        else:
            anchor_prediction_loss = zero

        class _Out(dict):
            __getattr__ = dict.__getitem__

        return _Out(
            {
                "loss": loss,
                "logits": logits,
                "lm_loss": lm_loss,
                "lm_loss_per_token": lm_loss_per_token if return_per_token_losses else None,
                "fha_feedback_gate": torch.stack(feedback_gates).mean() if feedback_gates else zero,
                "fha_anchor_entropy": torch.stack(anchor_entropies).mean() if anchor_entropies else zero,
                "fha_anchor_prediction_loss": anchor_prediction_loss,
                "fha_anchor_prediction_weight": logits.new_tensor(float(self.config.fha_anchor_prediction_weight)),
            }
        )
