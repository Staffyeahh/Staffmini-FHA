from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from transformers import PretrainedConfig


def _nested_get(obj: dict[str, Any], keys: list[str], default: Any) -> Any:
    curr: Any = obj
    for key in keys:
        if not isinstance(curr, dict) or key not in curr:
            return default
        curr = curr[key]
    return curr


class FHAConfig(PretrainedConfig):
    """Minimal config for the Fractal Hybrid Architecture experiments."""

    model_type = "staffmini_fha"

    def __init__(
        self,
        model_name: str = "staffmini_fha",
        vocab_size: int = 30294,
        tie_word_embeddings: bool = True,
        max_position_embeddings: int = 4096,
        d_model: int = 384,
        n_layers: int = 8,
        head_dim: int = 64,
        norm_type: str = "rmsnorm",
        zero_centered_rmsnorm: bool = True,
        residual_in_fp32: bool = False,
        lm_head_bias: bool = False,
        attention_layout: dict[str, Any] | None = None,
        ffn_layout: dict[str, Any] | None = None,
        init: dict[str, Any] | None = None,
        fha: dict[str, Any] | None = None,
        mtp: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.model_name = model_name
        self.vocab_size = int(vocab_size)
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.max_position_embeddings = int(max_position_embeddings)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.head_dim = int(head_dim)
        self.norm_type = str(norm_type)
        self.zero_centered_rmsnorm = bool(zero_centered_rmsnorm)
        self.residual_in_fp32 = bool(residual_in_fp32)
        self.lm_head_bias = bool(lm_head_bias)

        attention_layout = attention_layout or {}
        self.attention_type = str(attention_layout.get("type", "fha"))
        self.query_heads_full = int(attention_layout.get("query_heads_full", 6))
        self.query_heads_swa = int(attention_layout.get("query_heads_swa", 6))
        self.kv_heads = int(attention_layout.get("kv_heads", 6))
        self.head_wise_output_gate = bool(attention_layout.get("head_wise_output_gate", False))
        self.rope_theta = float(attention_layout.get("rope_theta", 10000.0))
        self.rope_dim_full = int(_nested_get(attention_layout, ["rope_dims", "full"], 64))
        self.rope_dim_swa = int(_nested_get(attention_layout, ["rope_dims", "swa"], 64))

        ffn_layout = ffn_layout or {}
        dense_ffn = ffn_layout.get("dense_ffn", {}) or {}
        self.first_n_dense_layers = int(ffn_layout.get("first_n_dense_layers", self.n_layers))
        self.dense_ffn_type = str(dense_ffn.get("type", "swiglu"))
        self.dense_ffn_hidden_size = int(dense_ffn.get("hidden_size", 1152))

        init = init or {}
        self.init_method = str(init.get("method", "xavier_uniform"))
        self.embed_std = float(init.get("embed_std", 0.02))
        self.lm_head_std = float(init.get("lm_head_std", 0.02))

        fha = fha or {}
        self.fha_enabled = bool(fha.get("enabled", True))
        self.fha_anchor_type = str(fha.get("anchor_type", "gated_delta"))
        self.fha_anchor_stride = int(fha.get("anchor_stride", 16))
        self.fha_anchor_slots = int(fha.get("anchor_slots", 1))
        self.fha_micro_kernel_size = int(fha.get("micro_kernel_size", 5))
        self.fha_macro_heads = int(fha.get("macro_heads", max(1, self.d_model // self.head_dim)))
        self.fha_macro_ffn_hidden_size = int(fha.get("macro_ffn_hidden_size", 768))
        self.fha_micro_ffn_hidden_size = int(fha.get("micro_ffn_hidden_size", self.dense_ffn_hidden_size))
        self.fha_feedback_init = float(fha.get("feedback_init", -2.0))
        self.fha_anchor_temperature = float(fha.get("anchor_temperature", 1.0))
        self.fha_anchor_entropy_weight = float(fha.get("anchor_entropy_weight", 0.0))
        self.fha_anchor_prediction_weight = float(fha.get("anchor_prediction_weight", 0.0))
        self.fha_use_rope = bool(fha.get("use_rope", False))
        self.fha_rope_theta = float(fha.get("rope_theta", attention_layout.get("rope_theta", 10000.0)))

        mtp = mtp or {}
        self.mtp_enabled = bool(mtp.get("enabled", False))

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "FHAConfig":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Expected YAML mapping in {path}.")
        return cls(**cfg)
