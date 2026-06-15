from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from staffmini_fha import FHAConfig, FractalHybridForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = FHAConfig.from_yaml(args.config)
    model = FractalHybridForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total_params: {total:,} ({total / 1e6:.2f}M)")
    print(f"trainable_params: {trainable:,} ({trainable / 1e6:.2f}M)")


if __name__ == "__main__":
    main()
