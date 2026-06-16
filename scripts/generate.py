from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from staffmini_fha import FHAConfig, FractalHybridForCausalLM
from staffmini_fha.model import BlockCache


def load_model(config_path: Path, checkpoint: Path, device: torch.device) -> FractalHybridForCausalLM:
    cfg = FHAConfig.from_yaml(config_path)
    model = FractalHybridForCausalLM(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    return model


@torch.no_grad()
def generate(
    model: FractalHybridForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    # Prefill: run entire prompt through model with caching
    with amp_ctx:
        caches = [BlockCache() for _ in range(len(model.layers))]
        for t in range(ids.shape[1]):
            logits, caches = model.step(ids[:, t:t+1], caches)

    generated = ids

    # Decode: one token at a time with caching
    tok_start = time.perf_counter()
    for _ in range(max_new_tokens):
        last_logits = logits[:, -1, :].float()

        if temperature <= 0:
            next_token = last_logits.argmax(dim=-1, keepdim=True)
        else:
            last_logits = last_logits / temperature
            if top_k > 0:
                cutoff = torch.topk(last_logits, min(top_k, last_logits.size(-1)))[0][:, -1:]
                last_logits = torch.where(last_logits < cutoff, torch.full_like(last_logits, float("-inf")), last_logits)
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

        with amp_ctx:
            logits, caches = model.step(next_token, caches)

    tok_elapsed = time.perf_counter() - tok_start
    n_new = generated.shape[1] - ids.shape[1]
    if tok_elapsed > 0 and n_new > 0:
        print(f"\n[{n_new} tokens in {tok_elapsed:.2f}s = {n_new/tok_elapsed:.1f} tok/s]", file=sys.stderr)

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/staffyeahh_32k"))
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(args.tokenizer))
    model = load_model(args.model_config, args.checkpoint, device)
    print(generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k))


if __name__ == "__main__":
    main()
