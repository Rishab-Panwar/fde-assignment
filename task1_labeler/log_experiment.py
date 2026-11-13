"""Score a prediction file on dev / val / all and append one row to experiments.md.

Usage:
    python log_experiment.py --id E1 --pred out/v1_predictions.jsonl --change "Baseline" [--hypothesis "..."]

Reads <pred>.meta.json (written by run.py) for model, settings, tokens, cost and runtime.
Validation IDs come from data/split.json and are never used for tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WEIGHTS = {"disposition": 15, "sub_disposition": 20, "lead_quality": 15, "next_call_at": 20}
LOG = Path(__file__).with_name("experiments.md")
HEADER = (
    "| ID | Change | Model / effort | Prompt | Dev score /70 (n=34) | **Val score /70 (n=16)** | All /70 | "
    "Disp | Sub | LQ | Next call | Exact | Tokens in/out | Cost $ | Time s | Keep? |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def load(path):
    return {r["transcript_id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def score(gold, pred, ids):
    c = dict.fromkeys(WEIGHTS, 0)
    exact = 0
    for tid in ids:
        g, p = gold[tid], pred.get(tid, {})
        ok = {
            "disposition": p.get("disposition") == g["disposition"],
            "sub_disposition": p.get("disposition") == g["disposition"] and p.get("sub_disposition") == g["sub_disposition"],
            "lead_quality": p.get("lead_quality") == g["lead_quality"],
            "next_call_at": p.get("next_call_at", "MISSING") == g["next_call_at"],
        }
        for f, v in ok.items():
            c[f] += v
        exact += all(ok.values())
    n = len(ids)
    return {"points": sum(c[f] / n * w for f, w in WEIGHTS.items()), "acc": {f: c[f] / n for f in c},
            "exact": exact / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--change", required=True, help="what is different from the previous experiment")
    ap.add_argument("--keep", default="?", help="yes / no / ? (decided after looking at val)")
    ap.add_argument("--gold", default="data/gold.jsonl")
    ap.add_argument("--split", default="data/split.json")
    args = ap.parse_args()

    gold, pred = load(args.gold), load(args.pred)
    split = json.load(open(args.split))
    meta_path = Path(args.pred).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    s_dev, s_val, s_all = (score(gold, pred, ids) for ids in (split["dev"], split["val"], list(gold)))

    a = s_all["acc"]
    row = (f"| {args.id} | {args.change} | {meta.get('model', '?')} / {meta.get('reasoning_effort', '-')} | "
           f"{meta.get('prompt_version', '?')} | {s_dev['points']:.1f} | **{s_val['points']:.1f}** | {s_all['points']:.1f} | "
           f"{a['disposition']:.0%} | {a['sub_disposition']:.0%} | {a['lead_quality']:.0%} | {a['next_call_at']:.0%} | "
           f"{s_all['exact']:.0%} | {meta.get('prompt_tokens', 0):,}/{meta.get('completion_tokens', 0):,} | "
           f"{meta.get('est_cost_usd', 0):.3f} | {meta.get('wall_seconds', 0):.0f} | {args.keep} |\n")

    text = LOG.read_text(encoding="utf-8") if LOG.exists() else "# Experiment log\n\n## Results\n"
    if "| ID | Change |" not in text:
        text = text.rstrip("\n") + ("\n\nAccuracy columns are on all 50; decisions use the **Val** column. "
                                    "Score = assignment weights (disp 15, sub 20, LQ 15, next call 20).\n\n" + HEADER)
        LOG.write_text(text, encoding="utf-8")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(row)
    print(row.strip())
    for name, s in (("dev", s_dev), ("val", s_val), ("all", s_all)):
        acc = "  ".join(f"{k[:4]}={v:.0%}" for k, v in s["acc"].items())
        print(f"{name:4s} n={s['n']:2d}  score={s['points']:.1f}/70  {acc}  exact={s['exact']:.0%}")


if __name__ == "__main__":
    main()
