"""Deterministic stratified dev/validation split (by disposition/sub_disposition).

Validation IDs are frozen BEFORE any prompt iteration and never inspected while
tuning; they give an honest estimate of performance on unseen calls.
Usage: python split.py data/gold.jsonl data/split.json
"""
import json
import sys
from collections import defaultdict

gold = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
groups = defaultdict(list)
for g in gold:
    groups[(g["disposition"], g["sub_disposition"])].append(g["transcript_id"])
val = []
for key in sorted(groups):
    ids = sorted(groups[key])
    val += ids[1::3]  # every 3rd record of each class (classes of size 1 stay in dev)
dev = sorted(g["transcript_id"] for g in gold if g["transcript_id"] not in val)
json.dump({"dev": dev, "val": sorted(val)}, open(sys.argv[2], "w"), indent=1)
print(f"dev={len(dev)} val={len(val)}")
