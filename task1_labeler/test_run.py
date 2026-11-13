"""Runner robustness tests with a FAKE Groq client (no network, no API key).
Run: python -m unittest test_run -v"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run

GOOD = {  # a valid v4 extraction: follow-up, no blocker, nothing scheduled
    "patient_summary": "", "meaningful_dialogue": True, "invalid_reason": "none", "invalid_evidence": "",
    "refusal": "none", "refusal_evidence": "",
    "appointment": {"evidence": "", "confirmed": False, "modality": "clinic", "is_reschedule": False,
                    "day": {"kind": "none", "n": None, "weekday": "none", "day_of_month": None, "month": None, "year": None},
                    "time": {"kind": "none", "hour": None, "minute": None, "ampm": "unspecified"}},
    "primary_blocker": "cost_concern", "blocker_evidence": "", "information_review_pending": False,
    "readiness": "considering", "defers_beyond_30_days": False,
    "callback": {"evidence": "", "exists": False, "acknowledged": False,
                 "day": {"kind": "none", "n": None, "weekday": "none", "day_of_month": None, "month": None, "year": None},
                 "time": {"kind": "none", "hour": None, "minute": None, "ampm": "unspecified"}},
    "vague_postponement": True, "vague_evidence": "",
}


def run_main(input_path, out_path, fake, samples=3):
    argv = ["run.py", "--input", str(input_path), "--output", str(out_path), "--cache-dir", "",
            "--samples", str(samples), "--workers", "2"]
    with mock.patch.object(run, "call_groq", fake), mock.patch.object(sys, "argv", argv), \
            mock.patch.dict(os.environ, {"GROQ_API_KEY": "fake"}):
        try:
            run.main()
            return 0
        except SystemExit as e:
            return e.code


class TestQuotaDegradation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.inp, self.out = d / "in.jsonl", d / "pred.jsonl"
        with open(self.inp, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({"transcript_id": f"T{i}", "call_started_at": "2026-11-09T14:00:00+05:30",
                                    "timezone": "Asia/Kolkata", "transcript": f"Agent: hi\nCustomer: knee pain {i}"}) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def preds(self):
        return [json.loads(l) for l in open(self.out, encoding="utf-8")]

    def test_healthy_run_exit_0(self):
        code = run_main(self.inp, self.out, lambda m, k, s=None: (GOOD, {"prompt_tokens": 1, "completion_tokens": 1}))
        self.assertEqual(code, 0)
        self.assertEqual(len(self.preds()), 10)

    def test_quota_mid_run_still_writes_every_record(self):
        calls = {"n": 0}

        def fake(m, k, s=None):  # quota runs out after 13 calls: round 1 (10) completes, round 2 partial
            calls["n"] += 1
            if calls["n"] > 13:
                raise run.QuotaExhausted("tokens per day")
            return GOOD, {"prompt_tokens": 1, "completion_tokens": 1}

        code = run_main(self.inp, self.out, fake)
        self.assertEqual(code, 2)                      # flagged as degraded
        p = self.preds()
        self.assertEqual(len(p), 10)                   # but every record written
        self.assertTrue(all(x["sub_disposition"] == "Cost Concern" for x in p))  # real answers, not fallback
        meta = json.loads(self.out.with_suffix(".meta.json").read_text())
        self.assertTrue(meta["quota_exhausted"])
        self.assertEqual(meta["errors"], 0)            # no record fell back: each got >= 1 real sample
        self.assertEqual(meta["min_samples_completed"], 1)

    def test_quota_before_any_call_writes_fallbacks(self):
        def fake(m, k, s=None):
            raise run.QuotaExhausted("tokens per day")

        code = run_main(self.inp, self.out, fake)
        self.assertEqual(code, 2)
        self.assertEqual(len(self.preds()), 10)        # still a complete, valid file
        meta = json.loads(self.out.with_suffix(".meta.json").read_text())
        self.assertEqual(meta["errors"], 10)

    def test_config_error_aborts(self):
        def fake(m, k, s=None):
            raise run.FatalAPIError("HTTP 401 invalid api key")

        with self.assertRaises(run.FatalAPIError):
            run_main(self.inp, self.out, fake)


if __name__ == "__main__":
    unittest.main()
