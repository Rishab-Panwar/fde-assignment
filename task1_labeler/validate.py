"""Validate a predictions JSONL against the assignment's output/scoring contract.

Usage: python validate.py --input data/calls.jsonl --pred out/predictions.jsonl
Exit 0 if valid, 1 otherwise. Checks: parseable JSON per line, exactly the five keys, every input ID
present exactly once, allowed disposition / valid parent-child pair / allowed lead quality, and
next_call_at either JSON null or an ISO timestamp with seconds and +05:30 that is after call_started_at.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime

from combine import ALLOWED, LEAD_QUALITIES

KEYS = {"transcript_id", "disposition", "sub_disposition", "lead_quality", "next_call_at"}
TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+05:30$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    calls = {json.loads(l)["transcript_id"]: json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()}
    problems, seen = [], {}
    for n, line in enumerate(open(args.pred, encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            p = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: not valid JSON ({e})")
            continue
        tid = p.get("transcript_id")
        if set(p) != KEYS:
            problems.append(f"{tid}: keys {sorted(p)} != {sorted(KEYS)}")
        seen[tid] = seen.get(tid, 0) + 1
        d, s, lq, nxt = p.get("disposition"), p.get("sub_disposition"), p.get("lead_quality"), p.get("next_call_at")
        if d not in ALLOWED:
            problems.append(f"{tid}: invalid disposition {d!r}")
        elif s not in ALLOWED[d]:
            problems.append(f"{tid}: invalid pair {d!r} / {s!r}")
        if lq not in LEAD_QUALITIES:
            problems.append(f"{tid}: invalid lead_quality {lq!r}")
        if nxt is not None:
            if not isinstance(nxt, str) or not TS.match(nxt):
                problems.append(f"{tid}: next_call_at {nxt!r} is not null or YYYY-MM-DDTHH:MM:SS+05:30")
            elif tid in calls and datetime.fromisoformat(nxt) <= datetime.fromisoformat(calls[tid]["call_started_at"]):
                problems.append(f"{tid}: next_call_at {nxt} is not after call_started_at")
    for tid in calls:
        if seen.get(tid, 0) == 0:
            problems.append(f"{tid}: missing")
    for tid, c in seen.items():
        if c > 1:
            problems.append(f"{tid}: duplicated {c} times")
        if tid not in calls:
            problems.append(f"{tid}: not in input (extra ID)")

    if problems:
        print(f"INVALID: {len(problems)} problem(s)")
        for p in problems:
            print("  " + p)
        return 1
    print(f"VALID: {len(calls)} records, all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
