"""Compare prediction files against a reference run, with an overfitting guard.

Usage:
    python compare.py --ref out/v4x3_predictions.jsonl out/N_v4x3_seed10.jsonl out/X2_v5x3.jsonl

Reports the weighted label score (/70) and exact match on:
  - all 50 calls
  - val: the 16 frozen validation calls (never used to write fixes)
  - untouched: every call NOT used to derive any prompt rule (val + dev calls outside SOURCE_CALLS).
    This is the decision set: improvements on the source calls alone are not trusted (overfitting).
Then lists per-call, per-field wins and regressions versus the reference.
"""
from __future__ import annotations

import argparse
import json

# Dev calls whose transcripts were read to derive prompt rules (v2/v4/v5). Never val calls.
SOURCE_CALLS = {"AVT-001", "AVT-013", "AVT-020", "AVT-024", "AVT-030", "AVT-032", "AVT-034", "AVT-038",
                "AVT-040", "AVT-041", "AVT-042", "AVT-046", "AVT-049"}
WEIGHTS = {"disposition": 15, "sub_disposition": 20, "lead_quality": 15, "next_call_at": 20}


def load(path):
    return {r["transcript_id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def correct(g, p):
    return {
        "disposition": p["disposition"] == g["disposition"],
        "sub_disposition": p["disposition"] == g["disposition"] and p["sub_disposition"] == g["sub_disposition"],
        "lead_quality": p["lead_quality"] == g["lead_quality"],
        "next_call_at": p["next_call_at"] == g["next_call_at"],
    }


def score(gold, pred, ids):
    pts = exact = 0.0
    for t in ids:
        ok = correct(gold[t], pred[t])
        pts += sum(WEIGHTS[f] for f, v in ok.items() if v) / len(ids)
        exact += all(ok.values())
    return pts, exact / len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--gold", default="data/gold.jsonl")
    ap.add_argument("--split", default="data/split.json")
    args = ap.parse_args()
    gold = load(args.gold)
    val = set(json.load(open(args.split))["val"])
    sets = {"all": sorted(gold), "val": sorted(val), "untouched": sorted(t for t in gold if t not in SOURCE_CALLS)}
    ref = load(args.ref)

    print(f"{'run':38s}" + "".join(f"{k + f' (n={len(v)})':>22s}" for k, v in sets.items()))
    for path in [args.ref] + args.preds:
        p = load(path)
        cells = []
        for ids in sets.values():
            s, e = score(gold, p, ids)
            cells.append(f"{s:6.1f}/70 {e:5.0%} ex")
        print(f"{path.split('/')[-1][:38]:38s}" + "".join(f"{c:>22s}" for c in cells))

    for path in args.preds:
        p = load(path)
        print(f"\n--- {path} vs {args.ref}")
        for t in sorted(gold):
            a, b = correct(gold[t], ref[t]), correct(gold[t], p[t])
            for f in WEIGHTS:
                if a[f] != b[f]:
                    tag = "SOURCE" if t in SOURCE_CALLS else ("VAL" if t in val else "dev")
                    print(f"  {'WIN ' if b[f] else 'LOSS'} {tag:6s} {t} {f:16s} gold={gold[t][f]} | new={p[t][f]}")


if __name__ == "__main__":
    main()
