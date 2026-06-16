from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import torch, torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from staffmini_fha import FHAConfig, FractalHybridForCausalLM

NEEDLES = [
    {"id": "password", "insert": "The secret password is ZORBLAX.",
     "question": "What is the secret password?", "answer": "ZORBLAX", "wrong": "QUANTUM"},
    {"id": "number", "insert": "The lucky number is 7749.",
     "question": "What is the lucky number?", "answer": "7749", "wrong": "1234"},
    {"id": "city", "insert": "The hidden city is called Aethermoor.",
     "question": "What is the hidden city called?", "answer": "Aethermoor", "wrong": "Luminaris"},
    {"id": "color", "insert": "The enchanted crystal glows ultraviolet.",
     "question": "What color does the crystal glow?", "answer": "ultraviolet", "wrong": "crimson"},
    {"id": "name", "insert": "The ancient dragon is named Flamethrax.",
     "question": "What is the ancient dragon named?", "answer": "Flamethrax", "wrong": "Stormwing"},
]

HAYSTACK = [
    "The weather today is mild with a gentle breeze from the east.",
    "Many researchers have studied the effects of climate on agriculture.",
    "In the old library, shelves of books lined every wall from floor to ceiling.",
    "The train departed on schedule, arriving at its destination by noon.",
    "Scientists discovered a new species of butterfly in the tropical rainforest.",
    "The conference room was filled with delegates from various countries.",
    "A gentle stream flowed through the meadow, reflecting the afternoon sun.",
    "The architect presented her plans for the new community center.",
    "Birds sang in the trees as the morning dew glistened on the grass.",
    "The museum attracted thousands of visitors every summer season.",
    "He carefully examined the manuscript, looking for clues about its origin.",
    "The orchestra performed a beautiful symphony at the grand concert hall.",
    "Farmers harvested their crops as autumn painted the landscape in gold.",
    "The ship sailed across the harbor under clear blue skies.",
    "Students gathered in the courtyard to discuss their research projects.",
    "The chef prepared a magnificent feast for the evening celebration.",
    "Mountains rose in the distance, their peaks covered with fresh snow.",
    "The detective studied the evidence carefully before making his report.",
    "Children played in the park while their parents watched from nearby.",
    "The garden bloomed with roses, tulips, and sunflowers in every color.",
]

def load_model(config_path, checkpoint, device):
    cfg = FHAConfig.from_yaml(config_path)
    model = FractalHybridForCausalLM(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    if device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    return model

def build_prompt(tok, target_len, needle, pos):
    needle_enc = tok.encode(needle["insert"])
    q_enc = tok.encode("\n" + needle["question"] + "\n")
    a_enc = tok.encode(" " + needle["answer"])
    w_enc = tok.encode(" " + needle["wrong"])
    budget = max(10, target_len - len(needle_enc) - len(q_enc) - max(len(a_enc), len(w_enc)) - 5)
    hay = ""
    i = 0
    while len(tok.encode(hay)) < budget:
        hay += HAYSTACK[i % len(HAYSTACK)] + " "
        i += 1
    hay_tokens = tok.encode(hay)[:budget]
    p = max(0, min(int(len(hay_tokens) * pos), len(hay_tokens)))
    all_toks = hay_tokens[:p] + needle_enc + hay_tokens[p:] + q_enc
    return all_toks, a_enc[0], w_enc[0]

@torch.no_grad()
def run_niah(model, tok, seq_lengths, device):
    positions = [0.0, 0.5, 1.0]
    pos_label = {0.0: "start", 0.5: "mid", 1.0: "end"}
    results = []

    for seq_len in seq_lengths:
        for needle in NEEDLES:
            for pos in positions:
                prompt_toks, ans_tok, wrong_tok = build_prompt(tok, seq_len, needle, pos)
                input_ids = torch.tensor([prompt_toks], device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model(input_ids=input_ids)
                last_logit = out["logits"][0, -1, :].float()
                lp = F.log_softmax(last_logit, dim=-1)
                correct_lp = float(lp[ans_tok].item())
                wrong_lp = float(lp[wrong_tok].item())
                delta = correct_lp - wrong_lp
                ok = correct_lp > wrong_lp
                results.append({
                    "seq_len": seq_len, "needle": needle["id"],
                    "position": pos, "correct_lp": correct_lp,
                    "wrong_lp": wrong_lp, "delta": delta, "ok": ok,
                })
                s = "OK" if ok else "FAIL"
                print(f"  seq={seq_len:4d} {needle['id']:10s} {pos_label[pos]:5s} "
                      f"correct={correct_lp:.3f} wrong={wrong_lp:.3f} delta={delta:+.3f} [{s}]")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/staffyeahh_32k"))
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(str(args.tokenizer))
    model = load_model(args.model_config, args.checkpoint, device)

    print(f"NIAH | {args.checkpoint.name} | seqs={args.sequence_lengths}\n")
    t0 = time.perf_counter()
    results = run_niah(model, tok, args.sequence_lengths, device)
    elapsed = time.perf_counter() - t0

    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} ({passed/total*100:.1f}%) | {elapsed:.1f}s")
    for sl in args.sequence_lengths:
        sub = [r for r in results if r["seq_len"] == sl]
        p = sum(1 for r in sub if r["ok"])
        print(f"  seq={sl}: {p}/{len(sub)} ({p/len(sub)*100:.0f}%)")
    for pos in [0.0, 0.5, 1.0]:
        sub = [r for r in results if r["position"] == pos]
        p = sum(1 for r in sub if r["ok"])
        lbl = {0.0:"start",0.5:"mid",1.0:"end"}[pos]
        print(f"  {lbl}: {p}/{len(sub)} ({p/len(sub)*100:.0f}%)")

    output = {"created": datetime.now(timezone.utc).isoformat(),
              "checkpoint": str(args.checkpoint), "total": total,
              "passed": passed, "retrieval_rate": passed/total if total else 0,
              "results": results}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2))
        print(f"\nJSON: {args.output_json}")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        lbl = {0.0:"start",0.5:"mid",1.0:"end"}
        lines = [f"# NIAH — {args.checkpoint.name}", "",
                 f"Retrieval: **{passed/total*100:.1f}%** ({passed}/{total})", "",
                 "| Seq | Needle | Pos | Correct | Wrong | Delta | |",
                 "|---:|:---|:---:|---:|---:|---:|:---:|"]
        for r in results:
            s = "OK" if r["ok"] else "FAIL"
            lines.append(f"| {r['seq_len']} | {r['needle']} | {lbl[r['position']]} | "
                         f"{r['correct_lp']:.3f} | {r['wrong_lp']:.3f} | {r['delta']:+.3f} | {s} |")
        args.output_md.write_text("\n".join(lines) + "\n")
        print(f"MD:   {args.output_md}")

if __name__ == "__main__":
    main()
