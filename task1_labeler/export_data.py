"""Export the workbook's '50 Labeled Calls' tab to input-only JSONL and gold JSONL.

Usage: python export_data.py <workbook.xlsx> <out_dir>
"""
import json
import sys
from pathlib import Path

import openpyxl


def main(xlsx: str, out_dir: str) -> None:
    ws = openpyxl.load_workbook(xlsx, read_only=True)["50 Labeled Calls"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calls.jsonl", "w", encoding="utf-8") as fi, open(out / "gold.jsonl", "w", encoding="utf-8") as fg:
        for r in rows[1:]:
            rec = dict(zip(header, r))
            if not rec.get("transcript_id"):
                continue
            fi.write(json.dumps({k: rec[k] for k in ("transcript_id", "call_started_at", "timezone", "transcript")}, ensure_ascii=False) + "\n")
            nxt = rec["next_call_at"]
            fg.write(json.dumps({
                "transcript_id": rec["transcript_id"],
                "disposition": rec["disposition"],
                "sub_disposition": rec["sub_disposition"],
                "lead_quality": rec["lead_quality"],
                "next_call_at": None if nxt in (None, "null") else nxt,
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
