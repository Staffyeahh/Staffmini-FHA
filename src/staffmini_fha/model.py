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


class DenseSwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        batch_size, num_anchors, _ = anchors.shape
        normed = self.input_norm(anchors)
        q = self.q_proj(normed).view(batch_size, num_anchors, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(batch_size, num_anchors, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(batch_size, num_anchors, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=self.scale)
        attn = attn.transpose(1, 2).reshape(batch_size, num_anchors, -1)
        anchors = anchors + self.o_proj(attn)
        anchors = anchors + self.ffn(self.post_attn_norm(anchors))
        return anchors


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


class FractalHybridBlock(nn.Module):
    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.config = config
        self.anchor_stride = int(config.fha_anchor_stride)
        self.anchor_slots = max(1, int(config.fha_anchor_slots))
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


class FractalHybridForCausalLM(nn.Module):
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def __init__(self, config: FHAConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList([FractalHybridBlock(config) for _ in range(config.n_layers)])
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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.embed_std)
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.lm_head_std)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.embed_tokens.weight

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def get_expert_weight_norms_by_layer(self) -> tuple[None, None]:
        return None, None

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
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.config.max_position_embeddings}."
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.embed_tokens(input_ids) + self.position_embeddings(positions)

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
                "moe_aux_loss": zero,
                "moe_z_loss": zero,
                "moe_loss": zero,
                "mtp_loss": None,
                "mtp_loss_per_token": None,
                "router_token_counts": None,
                "router_entropy": None,
                "moe_layer_indices": None,
                "moe_token_counts_by_layer": None,
                "moe_router_entropy_by_layer": None,
                "moe_expert_output_norms_by_layer": None,
                "moe_capacity_overflow_rate_by_layer": None,
                "moe_max_to_median_norm_ratio_by_layer": None,
                "lm_loss_per_token": lm_loss_per_token if return_per_token_losses else lm_loss_per_token,
                "fha_feedback_gate": torch.stack(feedback_gates).mean() if feedback_gates else zero,
                "fha_anchor_entropy": torch.stack(anchor_entropies).mean() if anchor_entropies else zero,
                "fha_anchor_prediction_loss": anchor_prediction_loss,
                "fha_anchor_prediction_weight": logits.new_tensor(float(self.config.fha_anchor_prediction_weight)),
            }
        )
