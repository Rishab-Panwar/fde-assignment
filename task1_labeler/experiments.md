# Experiment log

Goal: find the best model + strategy in as few full runs as possible.

**Rules followed**
- Few full runs. Cheap targeted checks on the failing dev calls first.
- Decide on the **Val** column: 16 calls frozen in `data/split.json` before any tuning, never used to write fixes. Dev (34 calls) is where errors are read.
- Every fix must be a general rule from the policy sheet, never a patch for one call ID.

**Final system: E4.** Prompt v4 (`prompts.py`) + 3-sample self-consistency vote. **64.2/70 on all 50, 68.8/70 on Val, 78% exact match.**

## Full runs (all 50 calls)

Score = assignment weights (disposition 15, sub-disposition 20, lead quality 15, next call 20). Accuracy columns are on all 50.

| ID | Change | Prompt | Dev /70 (n=34) | **Val /70 (n=16)** | All /70 | Disp | Sub | LQ | Next call | Exact | Tokens in / out | Cost $ | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | Baseline: 10-question evidence extraction + Python rules | v1 | 62.1 | **67.5** | 63.8 | 98% | 80% | 98% | 92% | 76% | 245,558 / 71,152 | 0.080 | baseline |
| E2 | Structural redesign: model lists all blockers / callback proposals / refusal facts with line numbers, Python selects; compact prompt | v3 | 56.9 | **48.8** | 54.3 | 92% | 68% | 94% | 64% | 50% | 240,060 / 73,540 | 0.080 | **rejected** |
| E3 | v1 + 3 targeted edits (explicit callback acknowledgement; escort/companion → travel; any other pending treatment → current treatment) | v4 | 61.5 | **68.8** | 63.8 | 98% | 84% | 98% | 88% | 76% | 251,958 / 70,263 | 0.080 | kept (prompt) |
| E1-rerun | **Noise check:** E1 re-run with nothing changed | v1 | 59.7 | **70.0** | 63.0 | 98% | 80% | 98% | 88% | 72% | 245,558 / 69,656 | 0.079 | measurement |
| **E4** | **E3 prompt + self-consistency: 3 samples (seeds 7, 8, 9), hierarchical majority vote** | v4 | **62.1** | **68.8** | **64.2** | 98% | 84% | 98% | 90% | **78%** | 755,874 / 211,111 | 0.240 | **final** |

## Round 2: can E4 be beaten? (cost no longer the constraint; overfitting is)

Overfitting guard: **source calls** = the 13 dev calls whose transcripts were read to write prompt rules (001, 013, 020, 024, 030, 032, 034, 038, 040, 041, 042, 046, 049). **Untouched** = the other 37 (all 16 val + 21 dev). Gains on source calls alone are not trusted.

| ID | Change | All /70 | Val /70 | Untouched /70 | Decision |
|---|---|---|---|---|---|
| N | E4 re-run with seeds 10–12 (noise floor of the voted system) | 63.5 | 68.8 | 68.0 | noise ≈ 0.7 all / 1.5 untouched |
| X1 | E4 + `reasoning_effort=high` | 63.5 | 68.8 | 69.1 | **rejected**: no gain; 7× wall time (778 s), 4.8× output tokens; reasoning exhausted the 8k output budget on 4 calls (led to the budget-escalation retry fix) |
| X2 | Prompt v5 = v4 + 5 rule-sheet edits (P1 final-step blocker, P2 schedule check = travel / insurance check = cost, P3 broader vague postponement, P4 later specific callback wins, P5 frustration ≠ refusal), seeds 7–9 | 65.9 | 68.8 | 68.5 | looked +1.7 better, so reseeded… |
| X2-reseed | Same v5, seeds 10–12 | 61.9 | 63.1 | 64.2 | …and lost 4.0 points: lost confirmed appointments on 037 and 039 |

**Per-sample comparison (6 independent samples per prompt, mean score of each individual sample, far less noisy than comparing two voted runs):**

| Set | v4 mean ± sd | v5 mean ± sd | v5 − v4 |
|---|---|---|---|
| Source (13) | 50.0 ± 1.6 | 56.6 ± 1.9 | **+6.6** |
| Untouched (37) | 67.8 ± 0.8 | 65.9 ± 1.9 | **−1.9** |
| Val (16) | 67.6 ± 2.3 | 64.9 ± 3.2 | **−2.7** |
| All 50 | 63.2 ± 0.6 | 63.5 ± 1.4 | +0.3 |

Per field (all 50): next call +4.7 pts (P3 worked), but disposition −1.3 and lead quality −2.3. v5 made the model doubt real bookings: AVT-037 was confirmed in 6/6 v4 samples but only 3/6 v5 samples ("Don't confirm it, I can come before or after that as well"), and AVT-039 dropped from 5/6 to 3/6. **v5 is textbook overfitting: it improves the calls its rules were written from and degrades the rest. Rejected; E4 stays final.**

## Targeted checks (subsets, not full runs)

| Check | Calls | Result | Decision |
|---|---|---|---|
| v2 wording fixes on the 10 dev calls E1 got wrong | 10 dev | 49.0/70 on that subset. Fixed 001, 030, 041 fully and the blocker for 013 and 040; new misses on 032, 034 (Information Requested over-triggered) and 038 (blocker dropped); 042 and 049 unchanged. | Wording-only fixes unstable; tried structural v3 next |

## Key findings

1. **Runs are not repeatable at temperature 0 + fixed seed.** Re-running E1 unchanged flipped labels on **11/50 calls**, moving the score by 0.8 points overall and 2.5 on Val. Single-run differences under ~2.5 points are therefore noise. Most flips sit in the borderline judgements: "is this a vague postponement?" (next call +1 vs +3 days) and "which blocker is primary?".
2. **Self-consistency removes most of that noise.** With 3 votes, disposition is unanimous in 49/50 calls, lead quality in 48/50, sub-disposition in 42/50 and next call in 40/50. E4 is the best and most stable configuration.
3. **The structural redesign (v3) failed clearly:** −9.5 overall, −18.7 on Val, well beyond the noise. Listing deferrals as blockers made the model stop flagging vague postponements (next-call accuracy 92% → 64%). Splitting refusal into sub-facts flipped answers both ways.
4. **Most remaining E4 errors are consistent, not random.** 8 of the 15 remaining field errors are unanimous across all 3 samples (calls 020, 032, 034, 040, 042). These are borderline judgement calls. Targeting them with more prompt edits (v2) caused regressions elsewhere, so we stopped rather than overfit 50 calls.

## Not run, with reasons
- **Few-shot examples:** with ~50 labelled calls, and the remaining errors being borderline judgements, few-shot excerpts risk teaching surface patterns. The rule-sheet boundary examples are already encoded in the prompt and the Python rules.
- **Model comparison:** gpt-oss-120b is Groq's strongest production model with strict JSON-schema support. gpt-oss-20b is smaller, and qwen3.8-27b is preview-only and may be withdrawn before evaluation.
- **5+ samples / reasoning_effort=high:** diminishing returns versus cost. 3 samples already make disposition and lead quality near-unanimous.

## Reproducing every number in this log and in ERROR_ANALYSIS.md

All commands run offline from the saved run files in `out/` (predictions, per-sample debug files, metadata): no API key, no cache.

| Claim | Command |
|---|---|
| Score rows (dev / val / all, per field, exact) | `python log_experiment.py --id X --pred out/<run>.jsonl --change "..."` (or `python evaluate.py --gold data/gold.jsonl --pred out/<run>.jsonl`) |
| Round-2 table incl. *untouched* column, per-call wins/losses | `python compare.py --ref out/v4x3_predictions.jsonl out/N_v4x3_seed10.jsonl out/X1_v4x3_high.jsonl out/X2_v5x3.jsonl out/v5_seeds10-12.jsonl` |
| v2 subset check (10 dev calls) | `python evaluate.py --gold data/gold.jsonl --pred out/v2_deverrors.jsonl --ids AVT-001,AVT-013,AVT-030,AVT-032,AVT-034,AVT-038,AVT-040,AVT-041,AVT-042,AVT-049` |
| Noise: 11/50 calls change between identical E1 runs | `python analysis.py noise out/v1_predictions.jsonl out/v1_rerun_predictions.jsonl` |
| E4 vote unanimity; 15 remaining errors, 8 unanimous | `python analysis.py agreement out/v4x3_predictions.debug.jsonl` |
| v5 reseed / 5-vote / 6-vote views | `python analysis.py views out/X2_v5x6.debug.jsonl`, then `compare.py` as above |
| v4 vs v5 per-sample means, per field, AVT-037/039 booking shares | `python analysis.py persample --a out/v4x3_predictions.debug.jsonl out/N_v4x3_seed10.debug.jsonl --b out/X2_v5x6.debug.jsonl` |
| Per-sample votes on specific calls (e.g. 014, 022) | `python analysis.py samples AVT-014:sub_disposition AVT-022:lead_quality --files out/v4x3_predictions.debug.jsonl out/N_v4x3_seed10.debug.jsonl out/X2_v5x3.debug.jsonl` |
| Do-Not-Contact rule changes 0 of 300 real-call samples | `python analysis.py dnc-check out/v4x3_predictions.debug.jsonl out/N_v4x3_seed10.debug.jsonl` |
| Tokens / cost / wall time per run | `python analysis.py cost out/*.meta.json` |
| Robustness 12/12 | see README (`robustness/`) |

**Data:** `data/calls.jsonl` and `data/gold.jsonl` aren't in the repo (company data). Recreate them first with `python export_data.py "<workbook>.xlsx" data`; the export is byte-identical to the files used here.

The v2 subset check re-runs on just its 10 calls with `--ids AVT-001,AVT-013,AVT-030,AVT-032,AVT-034,AVT-038,AVT-040,AVT-041,AVT-042,AVT-049 --prompt prompts_v2 --samples 1`.

Re-*generating* a run (rather than re-scoring it) calls Groq: `python run.py --input data/calls.jsonl --output out/<run>.jsonl --prompt <prompts_vN> --samples <k> [--seed S] [--reasoning-effort high]`. Because hosted LLM output varies between runs even at temperature 0 (see the noise finding), a fresh re-run reproduces these numbers statistically, not bit-for-bit. The saved run files are the exact record.

**Code-change safety check.** After any later edit to `combine.py` / `run.py`, re-label every saved run with the current code: `python analysis.py regression-check v1_predictions v1_rerun_predictions v2_deverrors v3_predictions v4_predictions v4x3_predictions N_v4x3_seed10 X1_v4x3_high X2_v5x3 X2_v5x6 predictions`. It must report 0 changed records (it did after the final edits: the Do-Not-Contact rule, quota-safe sampling rounds, output-budget retry). Exact offline replay of the final predictions: `GROQ_API_KEY=invalid python run.py --input data/calls.jsonl --output out/replay.jsonl` (with `.cache/` present), which makes 0 API calls and is byte-identical to `out/v4x3_predictions.jsonl`.
