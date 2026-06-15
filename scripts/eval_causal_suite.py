from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from staffmini_fha import FHAConfig, FractalHybridForCausalLM


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def eval_length(
    model: FractalHybridForCausalLM,
    tokens: np.memmap,
    seq_len: int,
    batches: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    max_start = len(tokens) - seq_len - 1
    if max_start <= 0:
        raise ValueError(f"Token bin is too small for seq_len={seq_len}.")

    total_loss = 0.0
    total_tokens = 0
    top1 = 0
    start_time = time.perf_counter()
    for i in range(batches):
        starts = np.linspace(0, max_start, num=batch_size, dtype=np.int64)
        starts = (starts + i * 9973) % max_start
        batch_np = np.stack([np.array(tokens[s : s + seq_len], dtype=np.int64, copy=True) for s in starts])
        input_ids = torch.from_numpy(batch_np).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(input_ids=input_ids)
            logits = out["logits"][:, :-1, :].float()
            labels = input_ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")
            pred = logits.argmax(dim=-1)
        total_loss += float(loss.item())
        total_tokens += labels.numel()
        top1 += int((pred == labels).sum().item())

    elapsed = time.perf_counter() - start_time
    loss = total_loss / total_tokens
    return {
        "loss": loss,
        "ppl": math.exp(min(20.0, loss)),
        "top1": top1 / total_tokens,
        "tok_s": total_tokens / max(elapsed, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-bin", type=Path, default=Path("data/tokenized/cosmopedia_mix_5m.bin"))
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FHAConfig.from_yaml(args.model_config)
    model = FractalHybridForCausalLM(cfg).to(device)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    tokens = np.memmap(args.token_bin, mode="r", dtype=np.int32)
    results = {
        "created": datetime.now(timezone.utc).isoformat(),
        "model_config": str(args.model_config),
        "checkpoint": str(args.checkpoint),
        "token_bin": str(args.token_bin),
        "device": str(device),
        "results": {},
    }
    for seq_len in args.sequence_lengths:
        metrics = eval_length(model, tokens, seq_len, args.batches, args.batch_size, device)
        results["results"][str(seq_len)] = metrics
        print(
            f"seq={seq_len} loss={metrics['loss']:.4f} ppl={metrics['ppl']:.1f} "
            f"top1={metrics['top1'] * 100:.2f}% tok/s={metrics['tok_s']:.0f}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Causal Eval - {args.checkpoint}",
            "",
            f"- Created: `{results['created']}`",
            f"- Model config: `{args.model_config}`",
            f"- Token bin: `{args.token_bin}`",
            f"- Device: `{device}`",
            "",
            "| seq | loss | ppl | top-1 | tok/s |",
            "|---:|---:|---:|---:|---:|",
        ]
        for seq_len, metrics in results["results"].items():
            lines.append(
                f"| {seq_len} | {metrics['loss']:.4f} | {metrics['ppl']:.1f} | "
                f"{metrics['top1'] * 100:.2f}% | {metrics['tok_s']:.0f} |"
            )
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
