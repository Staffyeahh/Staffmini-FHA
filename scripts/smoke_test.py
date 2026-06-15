from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from staffmini_fha import FHAConfig, FractalHybridForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = FHAConfig.from_yaml(args.config)
    model = FractalHybridForCausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len))
    out = model(input_ids=input_ids, labels=input_ids)
    print(f"logits: {tuple(out['logits'].shape)}")
    print(f"loss: {float(out['loss'].item()):.4f}")
    print(f"lm_loss: {float(out['lm_loss'].item()):.4f}")
    print(f"fha_anchor_prediction_loss: {float(out['fha_anchor_prediction_loss'].item()):.4f}")


if __name__ == "__main__":
    main()
