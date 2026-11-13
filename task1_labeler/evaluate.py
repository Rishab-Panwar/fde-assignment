"""Score predictions against gold labels using the assignment's scoring contract.

Usage: python evaluate.py --gold data/gold.jsonl --pred out/predictions.jsonl [--ids AVT-001,AVT-002]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

WEIGHTS = {"disposition": 15, "sub_disposition": 20, "lead_quality": 15, "next_call_at": 20}


def load(path: str) -> dict:
    return {r["transcript_id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ids", default="", help="comma-separated subset of IDs to score")
    args = ap.parse_args()
    gold, pred = load(args.gold), load(args.pred)
    ids = [i for i in (args.ids.split(",") if args.ids else gold)]

    correct = Counter()
    exact = 0
    errors = []
    for tid in ids:
        g, p = gold[tid], pred.get(tid)
        if p is None or set(p) != {"transcript_id", *WEIGHTS}:
            errors.append((tid, "MISSING/BAD KEYS", "", ""))
            continue
        ok = {
            "disposition": p["disposition"] == g["disposition"],
            "sub_disposition": p["disposition"] == g["disposition"] and p["sub_disposition"] == g["sub_disposition"],
            "lead_quality": p["lead_quality"] == g["lead_quality"],
            "next_call_at": p["next_call_at"] == g["next_call_at"],
        }
        for f, v in ok.items():
            correct[f] += v
            if not v:
                errors.append((tid, f, g[f], p[f]))
        exact += all(ok.values())

    n = len(ids)
    print(f"Records: {n}")
    score = 0.0
    for f, w in WEIGHTS.items():
        acc = correct[f] / n
        score += acc * w
        print(f"  {f:16s} {correct[f]:3d}/{n}  {acc:6.1%}")
    print(f"  {'exact match':16s} {exact:3d}/{n}  {exact / n:6.1%}")
    print(f"  label score (of 70): {score:.1f}")
    print("\nErrors (id | field | gold | pred):")
    for e in errors:
        print("  " + " | ".join(str(x) for x in e))


if __name__ == "__main__":
    main()
