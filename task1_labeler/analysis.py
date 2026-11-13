"""Regenerate every analysis number quoted in ERROR_ANALYSIS.md / experiments.md from saved run files.

Works offline from out/*.jsonl and out/*.debug.jsonl (the per-sample extractions and labels each run
saved), with no API key and no cache.

    python analysis.py noise      out/v1_predictions.jsonl out/v1_rerun_predictions.jsonl
    python analysis.py agreement  out/v4x3_predictions.debug.jsonl
    python analysis.py views      out/X2_v5x6.debug.jsonl            # writes v5 seed-subset votes
    python analysis.py persample  --a out/v4x3_predictions.debug.jsonl out/N_v4x3_seed10.debug.jsonl \
                                  --b out/X2_v5x6.debug.jsonl
    python analysis.py samples    AVT-014:sub_disposition AVT-022:lead_quality AVT-037:disposition \
                                  --files out/v4x3_predictions.debug.jsonl out/N_v4x3_seed10.debug.jsonl out/X2_v5x6.debug.jsonl
    python analysis.py dnc-check  out/v4x3_predictions.debug.jsonl out/N_v4x3_seed10.debug.jsonl
    python analysis.py cost       out/*.meta.json
    python analysis.py regression-check v1_predictions v1_rerun_predictions v2_deverrors v3_predictions \
                                  v4_predictions v4x3_predictions N_v4x3_seed10 X1_v4x3_high X2_v5x3 X2_v5x6 predictions
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from combine import combine, vote
from compare import SOURCE_CALLS, WEIGHTS, correct

FIELDS = list(WEIGHTS)


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def gold():
    return {r["transcript_id"]: r for r in jl("data/gold.jsonl")}


def val_ids():
    return set(json.load(open("data/split.json"))["val"])


def samples_by_call(paths):
    """{transcript_id: [per-sample label dicts in seed order]} merged across debug files."""
    out = {}
    for p in paths:
        for d in jl(p):
            out.setdefault(d["transcript_id"], []).extend(s["labels"] for s in d["samples"])
    return out


def points(g, p):
    return sum(WEIGHTS[f] for f, ok in correct(g, p).items() if ok)


# ----------------------------------------------------------------------------------------- commands
def cmd_noise(a, b):
    """How many labels change between two runs of the same configuration."""
    A = {r["transcript_id"]: r for r in jl(a)}
    B = {r["transcript_id"]: r for r in jl(b)}
    diff = [(t, f, A[t][f], B[t][f]) for t in A for f in FIELDS if A[t][f] != B[t][f]]
    print(f"fields changed: {len(diff)} | calls affected: {len({t for t, *_ in diff})} / {len(A)}")
    for t, f, x, y in diff:
        print(f"  {t} {f}: {x} -> {y}")


def cmd_agreement(debug):
    """Sample agreement in a voted run + which remaining errors are unanimous."""
    D = jl(debug)
    G = gold()
    print(f"calls where all samples agree on all 4 fields: "
          f"{sum(all(s['labels'] == d['samples'][0]['labels'] for s in d['samples']) for d in D)}/{len(D)}")
    for f in FIELDS:
        n = sum(len({str(s["labels"][f]) for s in d["samples"]}) == 1 for d in D)
        print(f"  {f:16s} unanimous in {n}/{len(D)}")
    preds = {r["transcript_id"]: r for r in jl(debug.replace(".debug.jsonl", ".jsonl"))}
    errs = [(d["transcript_id"], f) for d in D for f in FIELDS
            if preds[d["transcript_id"]][f] != G[d["transcript_id"]][f]]
    by_id = {d["transcript_id"]: d for d in D}
    unan = [(t, f) for t, f in errs if len({str(s["labels"][f]) for s in by_id[t]["samples"]}) == 1]
    print(f"remaining field errors: {len(errs)} on {len({t for t, _ in errs})} calls; "
          f"unanimous across samples: {len(unan)} on {sorted({t for t, _ in unan})}")


def cmd_views(debug):
    """Vote subsets of a 6-sample run (seeds 7..12): reseed check and 5-/6-vote variants."""
    D = jl(debug)
    views = {"v5_seeds10-12": [3, 4, 5], "v5_vote5_seeds7-11": [0, 1, 2, 3, 4], "v5_vote6": [0, 1, 2, 3, 4, 5]}
    for name, idx in views.items():
        with open(f"out/{name}.jsonl", "w", encoding="utf-8") as f:
            for d in D:
                v = vote([dict(d["samples"][i]["labels"], _trace=[]) for i in idx])
                f.write(json.dumps({"transcript_id": d["transcript_id"], **{k: v[k] for k in FIELDS}}) + "\n")
        print(f"wrote out/{name}.jsonl (samples {idx})")
    print("score them with: python compare.py --ref out/v4x3_predictions.jsonl out/v5_*.jsonl")


def cmd_persample(a_paths, b_paths):
    """Mean +- sd of the score of each individual sample: far less noisy than comparing voted runs."""
    G, val = gold(), val_ids()
    A, B = samples_by_call(a_paths), samples_by_call(b_paths)
    k = min(len(A[next(iter(A))]), len(B[next(iter(B))]))
    sets = {"all": list(G), "val": [t for t in G if t in val],
            "untouched": [t for t in G if t not in SOURCE_CALLS], "source": [t for t in G if t in SOURCE_CALLS]}

    def per_sample(S, ids):
        return [sum(points(G[t], S[t][i]) for t in ids) / len(ids) for i in range(k)]

    print(f"{k} samples per prompt.  A = {a_paths}  B = {b_paths}")
    for name, ids in sets.items():
        a, b = per_sample(A, ids), per_sample(B, ids)
        print(f"  {name:9s} n={len(ids):2d}  A {st.mean(a):6.2f} ± {st.stdev(a):4.2f}   "
              f"B {st.mean(b):6.2f} ± {st.stdev(b):4.2f}   B-A {st.mean(b) - st.mean(a):+6.2f}")
    print("per-field accuracy on all 50 (mean over samples):")
    for f in FIELDS:
        fa = st.mean(sum(correct(G[t], A[t][i])[f] for t in G) / len(G) for i in range(k))
        fb = st.mean(sum(correct(G[t], B[t][i])[f] for t in G) / len(G) for i in range(k))
        print(f"  {f:16s} A {fa:6.1%}   B {fb:6.1%}   {fb - fa:+.1%}")
    print("gold-booked calls not labelled Appointment Scheduled in every sample:")
    for t in (t for t in G if G[t]["disposition"] == "Appointment Scheduled"):
        na = sum(s["disposition"] == "Appointment Scheduled" for s in A[t][:k])
        nb = sum(s["disposition"] == "Appointment Scheduled" for s in B[t][:k])
        if na != k or nb != k:
            print(f"  {t}: A {na}/{k}   B {nb}/{k}")


def cmd_samples(specs, files):
    """Show every sample's value for chosen call:field pairs across runs."""
    G = gold()
    for spec in specs:
        t, f = spec.split(":")
        print(f"{t} {f}  gold={G[t][f]}")
        for p in files:
            d = next(d for d in jl(p) if d["transcript_id"] == t)
            print(f"   {Path(p).name:34s} {[s['labels'][f] for s in d['samples']]}")


def cmd_dnc_check(paths):
    """Re-apply the CURRENT combine.py to every saved real-call extraction; count label changes."""
    calls = {r["transcript_id"]: r for r in jl("data/calls.jsonl")}
    checked = changed = 0
    for p in paths:
        for d in jl(p):
            for s in d["samples"]:
                new = combine(s["extraction"], calls[d["transcript_id"]]["call_started_at"])
                checked += 1
                if any(new[k] != s["labels"][k] for k in FIELDS):
                    changed += 1
                    print(f"  CHANGED {d['transcript_id']} seed {s['seed']}")
    print(f"real-call samples re-labelled with current rules: {checked}, changed: {changed}")


def cmd_regression_check(names):
    """Re-label every saved run's per-sample extractions with the CURRENT combine.py / vote() and
    compare with the predictions saved when the run was made. 0 changes = later code edits did not
    alter any saved result. Handles both debug formats (pre-sampling: one 'extraction' per record)."""
    calls = {r["transcript_id"]: r for r in jl("data/calls.jsonl")}
    total = 0
    for name in names:
        saved = {r["transcript_id"]: r for r in jl(f"out/{name}.jsonl")}
        diff = 0
        for d in jl(f"out/{name}.debug.jsonl"):
            exts = [s["extraction"] for s in d["samples"]] if "samples" in d else [d["extraction"]]
            labs = [combine(e, calls[d["transcript_id"]]["call_started_at"]) for e in exts]
            new = vote(labs) if len(labs) > 1 else labs[0]
            if any(new[f] != saved[d["transcript_id"]][f] for f in FIELDS):
                diff += 1
                print(f"  CHANGED {name} {d['transcript_id']}")
        total += diff
        print(f"  {name:24s} records {len(saved):2d}  changed: {diff}")
    print(f"TOTAL records changed by current code: {total}")


def cmd_cost(metas):
    """Per-run tokens / cost / wall time from run metadata (cached samples count at their original cost)."""
    print(f"{'run':34s} {'samples':>7s} {'in tok':>11s} {'out tok':>10s} {'cost $':>7s} {'wall s':>7s}")
    for m in metas:
        x = json.load(open(m))
        print(f"{Path(m).name.replace('.meta.json', ''):34s} {x.get('samples_per_record', 1):7d} "
              f"{x['prompt_tokens']:11,d} {x['completion_tokens']:10,d} {x['est_cost_usd']:7.3f} {x['wall_seconds']:7.1f}")
    print("note: wall time is the time of the invocation that wrote the file (a replay from cache shows ~0 s).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("noise"); s.add_argument("a"); s.add_argument("b")
    s = sub.add_parser("agreement"); s.add_argument("debug")
    s = sub.add_parser("views"); s.add_argument("debug")
    s = sub.add_parser("persample"); s.add_argument("--a", nargs="+", required=True); s.add_argument("--b", nargs="+", required=True)
    s = sub.add_parser("samples"); s.add_argument("specs", nargs="+"); s.add_argument("--files", nargs="+", required=True)
    s = sub.add_parser("dnc-check"); s.add_argument("paths", nargs="+")
    s = sub.add_parser("cost"); s.add_argument("metas", nargs="+")
    s = sub.add_parser("regression-check"); s.add_argument("names", nargs="+",
                                                           help="run names in out/, e.g. v4x3_predictions")
    a = ap.parse_args()
    {"noise": lambda: cmd_noise(a.a, a.b), "agreement": lambda: cmd_agreement(a.debug),
     "views": lambda: cmd_views(a.debug), "persample": lambda: cmd_persample(a.a, a.b),
     "samples": lambda: cmd_samples(a.specs, a.files), "dnc-check": lambda: cmd_dnc_check(a.paths),
     "cost": lambda: cmd_cost(a.metas), "regression-check": lambda: cmd_regression_check(a.names)}[a.cmd]()


if __name__ == "__main__":
    main()
