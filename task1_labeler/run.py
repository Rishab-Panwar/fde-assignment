"""Runner: input JSONL (transcript_id, call_started_at, timezone, transcript) -> prediction JSONL.

Usage:
    python run.py --input data/calls.jsonl --output out/predictions.jsonl

Reads GROQ_API_KEY from the environment. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from combine import combine, vote
from prompts import EXTRACTION_SCHEMA, PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE
NUMBER_LINES = False  # set per prompt module by set_prompt() (only prompts_v3 numbers lines)
from timeresolve import parse_call_time

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
GEN = {
    "temperature": 0,
    "top_p": 1,
    "seed": 7,
    "max_completion_tokens": 8192,
    "reasoning_effort": "medium",
}
TIMEOUT_S = 120
MAX_RETRIES = 8  # 429 rate limits honour retry-after, so several patient attempts are cheap
PRICE_IN, PRICE_OUT = 0.15, 0.60  # USD per 1M tokens, openai/gpt-oss-120b on Groq

FALLBACK_EXTRACTION = {  # used only if every retry fails; most common class, never crashes the batch
    "meaningful_dialogue": True, "invalid_reason": "none", "refusal": "none",
    "appointment": {"confirmed": False}, "primary_blocker": "none", "information_review_pending": False,
    "readiness": "considering", "defers_beyond_30_days": False, "callback": {"exists": False},
    "vague_postponement": False,
}


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines). Real environment variables take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


FILLER = {"hello", "hello?", "hi", "yes", "yeah", "okay", "ok", "hmm", "mhmm", "haan", "ji", "right", "sure",
          "alright", "yes yes", "okay okay", "okay sir", "okay ma'am", "yes sir", "yes ma'am", "hello hello"}


def is_filler(line: str) -> bool:
    """A line whose whole utterance is a greeting/backchannel ("Agent: Hello.", "Customer: Okay, okay.")."""
    _, sep, text = line.partition(":")
    words = "".join(ch for ch in (text if sep else line).lower() if ch.isalpha() or ch in " '?").split()
    return 0 < len(words) <= 3 and " ".join(words).replace("?", "") in FILLER | {w.rstrip("?") for w in FILLER}


def number_lines(transcript: str, compact: bool = False) -> str:
    """Preprocessing: prefix each line with its ORIGINAL position [n] so the model can cite it.
    compact=True (only used when a request exceeds the provider's size limit) drops pure filler
    lines; numbering keeps original positions and no other wording is changed."""
    lines = transcript.splitlines()
    return "\n".join(f"[{i}] {line}" for i, line in enumerate(lines, 1) if not (compact and is_filler(line)))


def build_messages(rec: dict, compact: bool = False) -> list:
    call_dt = parse_call_time(rec["call_started_at"])
    if NUMBER_LINES or compact:
        transcript = number_lines(rec["transcript"], compact)
    else:
        transcript = rec["transcript"]
    user = USER_TEMPLATE.format(
        transcript_id=rec["transcript_id"],
        call_started_at=rec["call_started_at"],
        weekday=call_dt.strftime("%A"),
        timezone=rec.get("timezone") or "Asia/Kolkata",
        transcript=transcript,
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


PRICES = {  # USD per 1M tokens (input, output), Groq price list
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}


def set_prompt(module_name: str) -> None:
    """Select a frozen prompt module (prompts = latest, prompts_v1 = baseline). Part of the cache key."""
    global SYSTEM_PROMPT, USER_TEMPLATE, PROMPT_VERSION, EXTRACTION_SCHEMA, NUMBER_LINES
    mod = importlib.import_module(module_name)
    SYSTEM_PROMPT, USER_TEMPLATE = mod.SYSTEM_PROMPT, mod.USER_TEMPLATE
    PROMPT_VERSION, EXTRACTION_SCHEMA = mod.PROMPT_VERSION, mod.EXTRACTION_SCHEMA
    NUMBER_LINES = getattr(mod, "NUMBER_LINES", False)


def set_model(model: str, reasoning_effort: str) -> None:
    """Select model/effort for this run. Both are part of the cache key."""
    global MODEL, PRICE_IN, PRICE_OUT
    MODEL = model
    if reasoning_effort == "none":
        GEN.pop("reasoning_effort", None)
    else:
        GEN["reasoning_effort"] = reasoning_effort
    PRICE_IN, PRICE_OUT = PRICES.get(model, (0.0, 0.0))


class FatalAPIError(RuntimeError):
    """Configuration errors (bad key, unknown model, rejected schema): abort the whole run loudly."""


class QuotaExhausted(FatalAPIError):
    """HTTP 429 on tokens-per-day: no further requests can succeed today. The run stops requesting but
    still writes predictions from the samples completed so far (see main: sampling in rounds)."""


class RequestTooLarge(RuntimeError):
    """HTTP 413: this record's request exceeds the account's per-request token limit. Not retryable as-is."""


def call_groq(messages: list, api_key: str, seed: int | None = None) -> tuple[dict, dict]:
    gen = dict(GEN) if seed is None else dict(GEN, seed=seed)
    body = {
        "model": MODEL,
        "messages": messages,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "call_evidence", "strict": True, "schema": EXTRACTION_SCHEMA}},
        **gen,
    }
    last_err = None
    for attempt in range(MAX_RETRIES):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(GROQ_URL, data=data, method="POST", headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
            "User-Agent": "fde-labeler/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            return json.loads(content), payload.get("usage", {})
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", "replace")  # classify on the FULL body, truncate only for logs
            last_err = f"HTTP {e.code}: {body_txt[:300]}"
            # 400 json_validate_failed / "does not match the expected schema" = the model produced
            # invalid output for this record -> retry. Any other 4xx (auth, unknown model,
            # malformed request schema) is a configuration error -> abort the run loudly.
            if e.code == 413:
                raise RequestTooLarge(last_err)
            if e.code == 429 and "tokens per day" in body_txt:
                # daily quota exhausted: retrying for minutes cannot succeed. Stop requesting; predictions
                # are still written from completed samples, and a rerun resumes from the cache.
                raise QuotaExhausted("Daily token quota (TPD) exhausted for this Groq organization. "
                                     + body_txt[:200])
            per_record = e.code == 400 and ("json_validate_failed" in body_txt
                                            or "does not match the expected schema" in body_txt)
            if e.code not in (408, 409, 429, 500, 502, 503, 504) and not per_record:
                raise FatalAPIError(last_err)
            if per_record:
                # most often the reasoning used the whole output budget before the JSON was written:
                # give this request more room (8k -> 16k -> 32k). Not part of the cache key.
                body["max_completion_tokens"] = min(32768, body.get("max_completion_tokens", 8192) * 2)
            wait = float(e.headers.get("retry-after") or 0) or min(60, 2 ** attempt * 2)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            last_err = repr(e)
            wait = min(60, 2 ** attempt * 2)
        time.sleep(wait + random.random())
    raise RuntimeError(f"Groq call failed after {MAX_RETRIES} attempts: {last_err}")


def cache_key(messages: list, seed: int | None = None) -> str:
    gen = dict(GEN) if seed is None else dict(GEN, seed=seed)
    h = hashlib.sha256()
    h.update(json.dumps({"model": MODEL, "gen": gen, "v": PROMPT_VERSION, "schema": EXTRACTION_SCHEMA,
                         "messages": messages}, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:24]


def extract_once(rec: dict, api_key: str, cache_dir: Path | None, seed: int) -> dict:
    """One LLM extraction for one record with one seed (cached by model+settings+seed+prompt+transcript)."""
    messages = build_messages(rec)
    usage, cached, error = {}, False, None
    cpath = cache_dir / f"{cache_key(messages, seed)}.json" if cache_dir else None
    if cpath and cpath.exists():
        blob = json.loads(cpath.read_text(encoding="utf-8"))
        extraction, usage, cached = blob["extraction"], blob["usage"], True
    else:
        try:
            try:
                extraction, usage = call_groq(messages, api_key, seed)
            except RequestTooLarge:
                # degrade gracefully: same prompt, filler lines removed, original line numbers kept
                print(f"{rec['transcript_id']}: request too large, retrying with compacted transcript",
                      file=sys.stderr)
                extraction, usage = call_groq(build_messages(rec, compact=True), api_key, seed)
                usage = dict(usage, compacted=True)
            if cpath:
                cpath.write_text(json.dumps({"extraction": extraction, "usage": usage}, ensure_ascii=False),
                                 encoding="utf-8")
        except FatalAPIError:
            raise
        except Exception as e:  # transient failure after all retries: keep the batch alive, flag loudly
            extraction, error = FALLBACK_EXTRACTION, str(e)
    return {"seed": seed, "extraction": extraction, "usage": usage, "cached": cached, "error": error}


def process(rec: dict, api_key: str, cache_dir: Path | None, samples: int = 1) -> dict:
    """Run `samples` independent extractions for one record, then finalize (single-record helper)."""
    base_seed = GEN["seed"]
    return finalize(rec, [extract_once(rec, api_key, cache_dir, base_seed + i) for i in range(samples)])


def finalize(rec: dict, runs: list) -> dict:
    """Label each sample deterministically (combine) and majority-vote them (combine.vote).
    A record with no completed sample gets the flagged fallback answer."""
    if not runs:
        runs = [{"seed": None, "extraction": FALLBACK_EXTRACTION, "usage": {}, "cached": False,
                 "error": "no sample completed (daily quota exhausted before this record was reached)"}]
    ok_runs = [r for r in runs if not r["error"]] or runs  # never vote on fallback answers if any real run exists
    labelled = [combine(r["extraction"], rec["call_started_at"]) for r in ok_runs]
    final = vote(labelled) if len(labelled) > 1 else labelled[0]
    pred = {"transcript_id": rec["transcript_id"], "disposition": final["disposition"],
            "sub_disposition": final["sub_disposition"], "lead_quality": final["lead_quality"],
            "next_call_at": final["next_call_at"]}
    usage = {k: sum(r["usage"].get(k, 0) for r in runs) for k in ("prompt_tokens", "completion_tokens")}
    errors = [r["error"] for r in runs if r["error"]]
    debug = {"transcript_id": rec["transcript_id"],
             "samples": [{"seed": r["seed"], "labels": {k: l[k] for k in pred if k != "transcript_id"},
                          "trace": l["_trace"], "extraction": r["extraction"]} for r, l in zip(ok_runs, labelled)],
             "vote_trace": final.get("_vote", ""), "usage": usage,
             "cached": all(r["cached"] for r in runs),
             # a record is only an error if NO sample succeeded (otherwise the vote used real answers)
             "error": "; ".join(errors) if len(errors) == len(runs) else None,
             "sample_errors": errors}
    return {"pred": pred, "debug": debug}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-dir", default=".cache", help="'' disables the response cache")
    ap.add_argument("--model", default=MODEL, help=f"Groq model ID (default {MODEL})")
    ap.add_argument("--reasoning-effort", default=GEN["reasoning_effort"],
                    help="low|medium|high for reasoning models; 'none' to omit the parameter")
    ap.add_argument("--prompt", default="prompts",
                    help="prompt module: prompts (final = v4) or frozen prompts_v1 / prompts_v2 / prompts_v3 / prompts_v4")
    ap.add_argument("--samples", type=int, default=3,
                    help="independent extractions per call (seeds 7, 8, ...) combined by majority vote")
    ap.add_argument("--seed", type=int, default=GEN["seed"],
                    help="base seed; sample i uses seed+i (default 7 -> seeds 7, 8, 9)")
    ap.add_argument("--ids", default="",
                    help="optional comma-separated transcript_ids: process only these records from --input")
    args = ap.parse_args()
    GEN["seed"] = args.seed
    set_prompt(args.prompt)
    set_model(args.model, args.reasoning_effort)

    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set (export it, or put GROQ_API_KEY=... in task1_labeler/.env)")
    recs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        missing = set(wanted) - {r["transcript_id"] for r in recs}
        if missing:
            sys.exit(f"--ids not found in {args.input}: {sorted(missing)}")
        recs = [r for r in recs if r["transcript_id"] in set(wanted)]
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Sample in ROUNDS: seed 7 for every record, then seed 8 for every record, ... If the provider's daily
    # quota runs out mid-run, every record reached so far already has at least one sample, so the output
    # degrades to single-run quality instead of losing whole records. Predictions are always written.
    runs_by_id = {r["transcript_id"]: [] for r in recs}
    quota_msg = None
    for k in range(args.samples):
        if quota_msg:
            break
        seed = GEN["seed"] + k
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(extract_once, r, api_key, cache_dir, seed): r["transcript_id"] for r in recs}
            done = 0
            for f in as_completed(futs):
                tid = futs[f]
                try:
                    run = f.result()
                except QuotaExhausted as e:
                    quota_msg = quota_msg or str(e)
                    continue
                runs_by_id[tid].append(run)
                done += 1
                status = "ERROR " + run["error"][:80] if run["error"] else ("cached" if run["cached"] else "ok")
                print(f"[sample {k + 1}/{args.samples} seed {seed}] [{done}/{len(recs)}] {tid} {status}",
                      file=sys.stderr)
        if quota_msg:
            print(f"STOPPED REQUESTING: {quota_msg[:160]}", file=sys.stderr)
    results = {tid: finalize(next(r for r in recs if r["transcript_id"] == tid), runs)
               for tid, runs in runs_by_id.items()}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp, open(out.with_suffix(".debug.jsonl"), "w", encoding="utf-8") as fd:
        for r in recs:  # preserve input order
            res = results[r["transcript_id"]]
            fp.write(json.dumps(res["pred"], ensure_ascii=False) + "\n")
            fd.write(json.dumps(res["debug"], ensure_ascii=False) + "\n")

    pt = sum(results[t]["debug"]["usage"].get("prompt_tokens", 0) for t in results)
    ct = sum(results[t]["debug"]["usage"].get("completion_tokens", 0) for t in results)
    errs = sum(1 for t in results if results[t]["debug"]["error"])
    sample_errs = sum(len(results[t]["debug"]["sample_errors"]) for t in results)
    completed = [len([r for r in runs_by_id[t] if not r["error"]]) for t in runs_by_id]
    meta = {"records": len(recs), "samples_per_record": args.samples, "errors": errs,
            "failed_samples": sample_errs, "min_samples_completed": min(completed),
            "quota_exhausted": bool(quota_msg), "wall_seconds": round(time.time() - t0, 1),
            "prompt_tokens": pt, "completion_tokens": ct,
            "est_cost_usd": round(pt / 1e6 * PRICE_IN + ct / 1e6 * PRICE_OUT, 4),
            "prompt_version": PROMPT_VERSION, "model": MODEL, **GEN}
    print(json.dumps(meta), file=sys.stderr)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if errs or quota_msg or min(completed) < args.samples:
        print(f"WARNING: degraded run. {errs} record(s) used the fallback answer; fewest completed samples "
              f"for a record: {min(completed)}/{args.samples}"
              + ("; daily quota exhausted (rerun later: completed samples are cached)" if quota_msg else "")
              + f". Predictions were still written to {out}; details in {out.with_suffix('.debug.jsonl')}",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
