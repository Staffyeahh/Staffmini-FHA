from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    checkpoint = repo / "outputs" / "v14_2_gated_delta_300" / "step_0000300" / "model.pt"
    config = repo / "configs" / "model_config_v14_2_gated_delta_anchor_50m.yaml"
    out_dir = repo / "outputs" / "v14_2_gated_delta_300"
    cmd = [
        sys.executable,
        str(repo / "scripts" / "eval_causal_suite.py"),
        "--model-config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--token-bin",
        str(repo / "data" / "tokenized" / "cosmopedia_mix_5m.bin"),
        "--sequence-lengths",
        "512",
        "1024",
        "2048",
        "--batches",
        "50",
        "--batch-size",
        "8",
        "--output-json",
        str(out_dir / "eval_causal_300.json"),
        "--output-md",
        str(out_dir / "eval_causal_300.md"),
    ]
    subprocess.run(cmd, cwd=repo, check=True)


if __name__ == "__main__":
    main()
