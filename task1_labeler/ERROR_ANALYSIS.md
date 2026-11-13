# Error analysis

## 1. Final results (E4: prompt v4, gpt-oss-120b, 3-sample majority vote)

| Split | n | Disposition | Sub-disposition | Lead quality | Next call | Exact match | Label score /70 |
|---|---|---|---|---|---|---|---|
| Dev (used for error reading) | 34 | 97% | 79% | 97% | 85% | 71% | 62.1 |
| **Validation (frozen, never tuned on)** | 16 | 100% | 94% | 100% | 100% | 94% | **68.8** |
| All 50 | 50 | 98% | 84% | 98% | 90% | 78% | 64.2 |

Score uses the assignment weights (disposition 15, sub-disposition 20, lead quality 15, next call 20). Sub-disposition only counts when the disposition is also right.

**Caveat on the validation number.** 16 calls is small, and the validation split happens to contain fewer borderline calls than dev. Re-running an identical configuration moved validation by 2.5 points (section 4). Our honest expectation for unseen calls is **about 62–66 / 70**, closer to the all-50 figure than to validation. Five validation calls (004, 017, 023, 044, 047) were read while designing the date logic, before the split existed. No prompt edit was based on a validation call.

## 2. What still goes wrong (15 field errors on 11 calls)

| Pattern | Calls | Example | Why it's hard |
|---|---|---|---|
| **Choosing the primary blocker** when several are raised | 013, 018, 020, 032, 034, 038, 049 | AVT-049: "I'll speak to my office first and discuss at home… I'll have you talk with my insurance person". Gold: Cost (insurance). Predicted: Family. | Prospects give 2–3 reasons. Gold follows the *final* main one, often implicitly. In 4 of these 7 the 3 samples disagree with each other. |
| **Vague postponement or not?** (decides +3 vs +1 days) | 024, 034, 046 | "Send it, then I will talk at home" or "let me know on WhatsApp": is that a postponement? | Genuinely borderline. Samples split 2–1 on all three, and the majority landed on the wrong side. |
| **"Next week" callback overridden by an earlier vague reference** | 040 | "I'll call after Diwali" … Agent: "Should I call next week?" Customer: "Yes, call in between". Gold: next Monday. | The model consistently keeps the earlier vague reference. |
| **Frustration read as refusal** | 042 | "Only when you tell me will I want it. You're not telling me. Forget it." Gold: Follow-up / Treatment Concern / Not Assessable. Predicted: Not Interested. | Accounts for 4 of the 15 field errors on its own. Unanimous across samples and every prompt version. |

**No errors in the deterministic layer.** Every remaining error traces back to an LLM extraction answer. When the extraction is right, the Python rules and date resolution produce the correct label every time (61 unit tests, incl. the rule sheet's boundary examples EX1–EX6).

## 3. Iterations and what each one taught us

| Step | Change | All /70 | Val /70 | Outcome |
|---|---|---|---|---|
| E1 | Baseline: 10 evidence questions + Python rules | 63.8 | 67.5 | Strong start. 10 dev errors, almost all "which blocker" and "callback acknowledged?" |
| v2 (10-call check) | Wording fixes for each dev error pattern | n/a | n/a | Fixed 3 calls fully but caused 3 new misses. **Wording tweaks shift borderline judgements unpredictably.** |
| E2 | Structural redesign: model lists *every* blocker / callback proposal / refusal fact with line numbers, Python picks the final one | 54.3 | 48.8 | **Rejected.** Listing deferrals as blockers made the model stop flagging postponements (next-call accuracy 92% → 64%). A sound idea on paper that made the model's task harder. |
| E3 | v1 + only the 3 edits that worked in v2 without side effects (explicit callback "okay"; escort need = travel; any other pending treatment = current treatment) | 63.8 | 68.8 | Fixed the targeted calls (001, 040, 041), but other calls flipped. That prompted the noise check. |
| E1-rerun | **Same as E1, nothing changed** | 63.0 | 70.0 | **11/50 calls changed label.** Run-to-run noise (≈ ±2.5 on val) was as large as the effects we were measuring. |
| **E4** | E3 + 3-sample majority vote | **64.2** | **68.8** | Best and most stable. Disposition unanimous on 49/50, lead quality on 48/50. **Final.** |

**Round 2: trying to beat E4 without overfitting.** We split the 50 calls into the 13 *source* calls whose transcripts had been read to write prompt rules, and 37 *untouched* calls. Three challengers:

| Challenger | Result | Decision |
|---|---|---|
| E4 re-seeded (noise floor of the voted system) | ±0.7 all, ±1.5 untouched | reference |
| `reasoning_effort=high` | no gain; 7× slower, ~5× output tokens; exhausted the output budget on 4 calls | rejected (and led to a retry fix: on schema failure the output budget doubles, 8k → 16k → 32k) |
| Prompt v5: five more rule-sheet edits for the remaining error patterns | first seed set +1.7 (looked like a win); **re-seeded: −2.3** | **rejected**: averaged over 6 samples per prompt, **+6.6 on source calls but −1.9 on untouched and −2.7 on val**. Broadening "vague postponement" made the model doubt real bookings (AVT-037 confirmed in 3/6 samples vs 6/6). |

**Biggest lesson:** at temperature 0 with a fixed seed, hosted LLM outputs still vary between runs on borderline judgements. Before comparing prompt variants, measure the noise floor. Then reduce variance with self-consistency rather than chasing single-run differences with more prompt edits. Compare prompt variants by the **mean score of many individual samples** on calls **not used to write the rules**. A single voted run on all 50 made v5 look like a +1.7 win when it was a regression on unseen-like calls.

**Deliberately not done:**
- **More prompt edits for the remaining errors.** Most are stable judgement calls on borderline cases (8 of 15 field errors are unanimous across all 3 samples, on 5 calls), some with arguable gold labels. With 50 calls, further tuning would overfit.
- **Model comparison.** See README: gpt-oss-120b is Groq's strongest production model with strict JSON support.

## 3b. Robustness test: outcomes absent from the 50 calls

The 50 labelled calls contain no Invalid Lead, Do Not Contact, Unable to Assess, Online Consultation or Rescheduled examples, but the hidden set may. We wrote 12 short synthetic transcripts with one clear answer each under the rule sheet (`robustness/make_cases.py`) and ran the **final pipeline unchanged**. They are tests only: never in the prompt and never used for tuning. The cases:
- wrong number, internal test, patient aged 12, outside service area with online care refused, no pain need
- "remove my number", explicit refusal, a silent call
- confirmed online consultation, rescheduled missed appointment, "call me in 2 hours" (seconds preserved), and a booking 8 days out (must be Medium, not High)

| Run | Result | Action |
|---|---|---|
| First run | 10/12 exact; disposition, lead quality and next call 100% | Two sub-label mismatches investigated |
| "Please remove my number and do not call me again" | All 3 samples **quoted** the stop-contact sentence as evidence, but 2/3 still picked Explicit Refusal | **Real gap.** Fixed deterministically in `combine.py`: a refusal whose own evidence contains an explicit stop-contact request becomes Do Not Contact (rule sheet: DNC takes precedence). No prompt change. Re-labelling all 300 cached real-call samples: **0 changes**. |
| "Busy now, call me in 2 hours" | Model: Travel Constraint (3/3) | **Our test label was wrong**: real gold AVT-044 (busy at the shop, "call me this evening") is Travel Constraint. Test corrected, model unchanged. |
| After fix | **12/12 exact** | Final predictions for the 50 calls unchanged |

## 4. Runtime, tokens, cost

| Run | Calls to Groq | Prompt tokens | Completion tokens | Est. cost | Wall time |
|---|---|---|---|---|---|
| Single run (1 sample × 50) | 50 | ~246k | ~71k | ~$0.08 | ~55 s (Dev tier) |
| **Final E4 (3 samples × 50)** | 150 | 755,874 | 211,111 | **$0.24** | ~70 s (Dev tier) |
| `reasoning_effort=high` (X1, rejected) | 150 | 752,907 | 1,026,525 | $0.73 | ~13 min |

Per-run figures for every experiment come from the saved run metadata: `python analysis.py cost out/*.meta.json`.

Pricing: gpt-oss-120b on Groq, $0.15 / M input, $0.60 / M output. **Per call:** about 5k input + 1.4k output tokens per sample, so about $0.005 per call with 3 samples.

**Free-tier note.** Groq's free tier allows 8k tokens/minute and 200k tokens/day per organization, so one 50-call sample needs more than a day's quota. Development hit both limits. The runner now handles them:
- daily-quota exhaustion stops new requests but still writes predictions: sampling runs in rounds (one sample for every call first), so every call reached has a real answer; a rerun resumes from the cache (tested with a fake API in `test_run.py`)
- oversized requests (413) are retried with filler lines removed
- schema-validation hiccups are retried per record instead of aborting the batch

## 5. If this were a real deployment: next steps
1. **Clarify the two ambiguous policy boundaries with the clinic team:** "send me details, I'll talk at home" (vague postponement or not?) and frustration vs refusal. Those two account for most remaining errors, and the fix is a policy clarification more than a model change.
2. **Route low-agreement calls to a human:** with 3 samples, a 2–1 split on sub-disposition or next call is a free confidence signal (15/50 calls had at least one split field).
3. **Grow the labelled set** beyond 50 calls before tuning further. Then few-shot excerpts or a second-pass verifier on just the blocker question become safe to evaluate.
