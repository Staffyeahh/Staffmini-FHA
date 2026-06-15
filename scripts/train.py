from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from staffmini_fha import FHAConfig, FractalHybridForCausalLM
from staffmini_fha.data import PackedTokenDataset


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return data


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def wsd_lr_factor(step: int, max_steps: int, warmup_ratio: float, stable_ratio: float) -> float:
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    stable_steps = int(max_steps * stable_ratio)
    if step < warmup_steps:
        return step / warmup_steps
    if step < warmup_steps + stable_steps:
        return 1.0
    decay_steps = max(1, max_steps - warmup_steps - stable_steps)
    return max(0.0, 1.0 - (step - warmup_steps - stable_steps) / decay_steps)


def cosine_lr_factor(step: int, max_steps: int, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
    model_cfg: FHAConfig,
    train_cfg: dict[str, Any],
) -> None:
    ckpt_dir = output_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, ckpt_dir / "optimizer.pt")
    with (ckpt_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "step": step,
                "model_config": model_cfg.to_dict(),
                "train_config": train_cfg,
            },
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Staffmini FHA causal LM.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--token-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FHAConfig.from_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = int(train_cfg.get("sequence_length", 512))
    micro_bs = int(train_cfg.get("micro_batch_size", 24))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 3))
    num_workers = int(train_cfg.get("num_workers", 0))

    dataset = PackedTokenDataset(args.token_bin, sequence_length=seq_len)
    loader = DataLoader(
        dataset,
        batch_size=micro_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    model = FractalHybridForCausalLM(cfg).to(device)
    if bool(train_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()

    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 4e-4)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
    )

    start_step = 0
    if args.resume is not None:
        ckpt_dir = args.resume
        model_path = ckpt_dir / "model.pt" if ckpt_dir.is_dir() else ckpt_dir
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
        state_path = ckpt_dir / "trainer_state.json" if ckpt_dir.is_dir() else None
        if state_path and state_path.exists():
            start_step = int(json.loads(state_path.read_text(encoding="utf-8")).get("step", 0))

    max_steps = int(train_cfg.get("max_steps", 300))
    schedule_steps = int(train_cfg.get("schedule_steps", max_steps))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    stable_ratio = float(train_cfg.get("wsd_stable_ratio", 0.70))
    lr_scheduler = str(train_cfg.get("lr_scheduler", "wsd")).lower()
    base_lr = float(train_cfg.get("learning_rate", 4e-4))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 100))

    metrics_path = args.output_dir / "metrics.jsonl"
    run_meta_path = args.output_dir / "run_meta.json"
    if start_step == 0:
        metrics_path.write_text("", encoding="utf-8")
    run_meta_path.write_text(
        json.dumps(
            {
                "model_config": cfg.to_dict(),
                "train_config": train_cfg,
                "tokens_per_step": seq_len * micro_bs * grad_accum,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Training {cfg.model_name} on {device} | params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | "
        f"seq={seq_len} micro_bs={micro_bs} accum={grad_accum}"
    )

    model.train()
    step = start_step
    tokens_seen = step * seq_len * micro_bs * grad_accum
    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=max_steps, initial=start_step, dynamic_ncols=True, desc="train")
    loader_iter = iter(loader)

    while step < max_steps:
        accum_loss = 0.0
        accum_lm = 0.0
        accum_gate = 0.0
        accum_entropy = 0.0
        accum_pred = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True).long()
            labels = input_ids
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(input_ids=input_ids, labels=labels)
                loss = out["loss"] / grad_accum
            loss.backward()
            accum_loss += float(out["loss"].detach().float().item())
            accum_lm += float(out["lm_loss"].detach().float().item())
            accum_gate += float(out["fha_feedback_gate"].detach().float().item())
            accum_entropy += float(out["fha_anchor_entropy"].detach().float().item())
            accum_pred += float(out["fha_anchor_prediction_loss"].detach().float().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        step += 1
        factor = (
            wsd_lr_factor(step, schedule_steps, warmup_ratio, stable_ratio)
            if lr_scheduler == "wsd"
            else cosine_lr_factor(step, schedule_steps, warmup_ratio)
        )
        lr = base_lr * factor
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += seq_len * micro_bs * grad_accum
        pbar.update(1)

        if step == 1 or step % log_every == 0:
            elapsed = time.perf_counter() - start_time
            tok_s = tokens_seen / max(elapsed, 1e-9)
            metrics = {
                "step": step,
                "loss": accum_loss / grad_accum,
                "lm_loss": accum_lm / grad_accum,
                "lr": lr,
                "tokens_seen": tokens_seen,
                "tok_s": tok_s,
                "fha_feedback_gate": accum_gate / grad_accum,
                "fha_anchor_entropy": accum_entropy / grad_accum,
                "fha_anchor_prediction_loss": accum_pred / grad_accum,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            eta = (max_steps - step) * elapsed / max(1, step - start_step)
            print(
                f"step {step:6d}/{max_steps} loss={metrics['loss']:.4f} lm={metrics['lm_loss']:.4f} "
                f"lr={lr:.2e} tok/s={tok_s:,.0f} fha(gate={metrics['fha_feedback_gate']:.3f} "
                f"ent={metrics['fha_anchor_entropy']:.3f} pred={metrics['fha_anchor_prediction_loss']:.3f}) "
                f"eta={format_seconds(eta)}"
            )

        if step % save_every == 0 or step == max_steps:
            save_checkpoint(model, optimizer, step, args.output_dir, cfg, train_cfg)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    pbar.close()
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"Done. Peak VRAM: {peak_gb:.2f} GB")
    print(f"Final checkpoint: {args.output_dir / f'step_{max_steps:07d}' / 'model.pt'}")


if __name__ == "__main__":
    main()
