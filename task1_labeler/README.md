# Task 1: Call transcript labeler

Reads a pain-clinic call transcript and predicts four labels: **disposition**, **sub-disposition**, **lead quality** and **next call time**, following the rules in the assignment workbook.

**Result on the 50 supplied calls:** 64.2 / 70 points, with all four labels right on 78% of calls. On 16 calls held back from all tuning: 68.8 / 70.

## How to run

```bash
# Python 3.10+ (tested on 3.12.10). No packages to install.
export GROQ_API_KEY=gsk_...        # Windows PowerShell: $env:GROQ_API_KEY="gsk_..."   (or put it in .env)
python run.py --input data/calls.jsonl --output out/predictions.jsonl
```

- **Input:** one JSON object per line with `transcript_id`, `call_started_at`, `timezone` and `transcript`. Nothing else is read.
- **Output:** one JSON object per line, in input order, with exactly `transcript_id`, `disposition`, `sub_disposition`, `lead_quality` and `next_call_at`. `next_call_at` is either a timestamp like `2026-10-15T18:00:00+05:30` or `null`.
- **Check the output format** (optional): `python validate.py --input data/calls.jsonl --pred out/predictions.jsonl`
- **Time and cost:** about 70 seconds and $0.24 for 50 calls on a paid Groq account.

**Optional settings:**
- `--samples 1` runs each call once instead of three times. It's cheaper and slightly less accurate (63.2 / 70 on average).
- `--workers N` sets how many calls run in parallel (default 4).
- `--prompt prompts_v1` (or `_v2`, `_v3`, `_v4`) reruns an earlier experiment.

**First, create `data/` from the workbook** (not included in the repo, since it's the company's data): `python export_data.py "<workbook>.xlsx" data` (needs `openpyxl==3.1.5`). This produces exactly the files used for every result here.

**Run on just some calls:** `--ids AVT-001,AVT-013` processes only those transcripts from the input file.

## How it works

```
transcript → AI model answers 10 questions (×3) → Python applies the rules → majority vote → 4 labels
```

1. **The AI model reads the call and answers 10 factual questions,** quoting the words that support each answer. For example: *Was a date and time for the consultation agreed? What is the main reason they didn't book? Did they agree to a callback, and when?* It never picks the final labels or works out dates.
2. **Python turns those answers into labels** using the workbook's rules: which outcome takes priority, which label pairs are allowed, what makes a lead High / Medium / Low, and when the next call should be.
3. **Python also does all date calculations.** For example, "tomorrow evening" on a 14 Oct call becomes `2026-10-15T18:00:00+05:30`, using the workbook's definitions (evening = 18:00, "next week" = Monday, and so on).
4. **Each call is run 3 times and the labels are decided by majority vote.** Running the same call twice can give different answers, even with randomness turned off: 11 of 50 calls changed between two identical runs. Voting removes most of that variation.

**Why this split:** the AI model is good at understanding messy, translated Hindi-English conversations. It's less reliable at strict rule-following and date arithmetic, so code does those.

## Model and settings

| Setting | Value |
|---|---|
| Model | `openai/gpt-oss-120b` on Groq (Groq's strongest production model that supports strict JSON output) |
| Temperature / top_p | 0 / 1 |
| Runs per call | 3 (seeds 7, 8, 9), majority vote |
| Reasoning effort | medium |
| Max output tokens | 8,192 |
| Output format | Strict JSON schema enforced by Groq (`prompts.py`), parsed with `json.loads` |
| Timeout | 120 seconds per request |
| Dependencies | Python standard library only |

## When something goes wrong

- **Temporary errors** (rate limits, server errors, malformed output): retried up to 8 times with increasing waits, respecting Groq's `retry-after`.
- **Model ran out of output space:** the retry doubles the space (up to 32k tokens).
- **Transcript too long for the account's limit:** retried with filler lines like "Hello." and "Okay." removed.
- **Daily token quota used up:** the run stops calling the API but **still writes predictions for every call**. Each call gets its first answer before any call gets a second, so every call reached has a real answer. The rest get a flagged fallback, and the exit code is `2`.
- **Invalid key or model:** stops immediately, exit code `1`.

A Groq free account (200k tokens/day) can't finish 3 runs of 50 calls in a day. Use `--samples 1` there.

## Files

| File | What it is |
|---|---|
| `run.py` | Main script: calls the model, retries, voting, writes output |
| `prompts.py` | The final prompt and output schema. `prompts_v1.py`–`prompts_v4.py` are earlier versions, kept to reproduce each experiment. |
| `combine.py` | The labelling rules and the vote |
| `timeresolve.py` | Turns spoken times ("day after tomorrow at 4") into timestamps |
| `validate.py` | Checks an output file has the required format |
| `evaluate.py`, `compare.py`, `log_experiment.py` | Scoring against the correct labels |
| `analysis.py` | Regenerates every number quoted in the docs from saved results |
| `export_data.py`, `split.py` | Workbook export; the fixed 34 / 16 tuning / held-back split |
| `test_*.py` | 67 tests: `python -m unittest test_combine test_timeresolve test_run` |
| `robustness/` | 12 made-up calls covering outcomes missing from the 50 (wrong number, under-15 patient, "don't call me", online booking, …). Result: 12/12. |
| `out/` | Saved results of every experiment. `out/predictions.jsonl` is the submission. |
| `ERROR_ANALYSIS.md` | Results, remaining errors, what was tried, cost |
| `experiments.md` | Every experiment, including rejected ones, and the command that reproduces each number |

## Rules followed

- The model only sees the four input fields. The correct labels are used only for scoring.
- No answers are looked up or hand-corrected. The prompt contains general rules only, plus a few short example phrases from the tuning calls (none from the held-back calls).
- Results are cached locally (`.cache/`, not included) so identical requests aren't repeated. A fresh run gives very similar but not byte-identical results, because of the run-to-run variation described above.
- No API keys are included.
- Built with help from Claude Code (an AI coding assistant). The submitted pipeline uses only the Groq model above.
