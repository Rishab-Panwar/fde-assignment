"""Evaluate the rule-based gender detector and compare it with a classical ML baseline.

Usage:
    python evaluate.py --input ../task1_labeler/data/calls.jsonl --reference reference_labels.jsonl

1. Rules (gender.py) vs the manually annotated reference labels.
2. Classical ML baseline, for comparison only (not part of the shipped solution):
   TF-IDF (word 1-2 grams) + logistic regression, leave-one-out cross-validation
   (train on 49 calls, predict the held-out one, repeat). Two variants:
     a) full transcript text
     b) same, with address terms masked (sir, ma'am, madam, Mr, Mrs, ...), which tests whether
        a model finds any gender signal BEYOND the honorifics the rules already use.
   Majority-class baseline reported for reference.
Calls whose reference label is "mixed" (two callers of different gender) are excluded from accuracy;
for the rules we report separately whether they abstained ("unknown") on them.
Requires scikit-learn (evaluation only; gender.py itself is standard library).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from gender import classify

HONORIFICS = re.compile(r"\b(sir|ma'?am|maam|madam|mam|mr|mrs|ms|miss|didi|bhaiya|brother|sister|aunty|uncle|ji)\b\.?", re.I)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def loo_tfidf(texts, labels):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline

    preds = []
    for train, test in LeaveOneOut().split(texts):
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit([texts[i] for i in train], [labels[i] for i in train])
        preds.append(model.predict([texts[test[0]]])[0])
    return preds


def acc(pred, gold):
    return sum(p == g for p, g in zip(pred, gold)) / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()

    calls = {r["transcript_id"]: r for r in load(args.input)}
    ref = {r["transcript_id"]: r for r in load(args.reference)}
    ids = [t for t in ref if ref[t]["gender"] in ("male", "female")]
    mixed = [t for t in ref if ref[t]["gender"] == "mixed"]
    gold = [ref[t]["gender"] for t in ids]
    print(f"Reference: {len(ref)} calls = {Counter(r['gender'] for r in ref.values())}; "
          f"scored on {len(ids)} single-speaker calls\n")

    # 1. rules
    rules = {t: classify(calls[t]) for t in ref}
    rp = [rules[t]["gender"] for t in ids]
    print(f"RULES (gender.py)             accuracy {acc(rp, gold):6.1%}  "
          f"({sum(p == g for p, g in zip(rp, gold))}/{len(ids)}), abstained on {rp.count('unknown')}")
    for cert in ("clear", "weak"):
        sub = [t for t in ids if ref[t]["certainty"] == cert]
        if sub:
            a = acc([rules[t]["gender"] for t in sub], [ref[t]["gender"] for t in sub])
            print(f"   on '{cert}' reference calls (n={len(sub)}): {a:.1%}")
    for t in mixed:
        print(f"   mixed-speaker call {t}: rules -> {rules[t]['gender']} (abstaining is the desired behaviour)")
    for t, p, g in zip(ids, rp, gold):
        if p != g:
            print(f"   ERROR {t}: rules={p} reference={g} | {ref[t]['basis']}")

    # throughput of the shipped classifier (single process, one core)
    import time
    recs = list(calls.values())
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < 2:
        for r in recs:
            classify(r)
            n += 1
    avg_chars = sum(len(r["transcript"]) for r in recs) / len(recs)
    print(f"\nSPEED: {n / (time.perf_counter() - t0):.0f} calls/second, one process "
          f"(avg transcript {avg_chars:.0f} chars)")

    # 2. classical ML baseline (comparison only)
    texts = [calls[t]["transcript"] for t in ids]
    majority = Counter(gold).most_common(1)[0][0]
    print(f"\nMAJORITY CLASS ('{majority}')     accuracy {acc([majority] * len(gold), gold):6.1%}")
    try:
        pa = loo_tfidf(texts, gold)
        pb = loo_tfidf([HONORIFICS.sub(" ", x) for x in texts], gold)
    except ImportError:
        print("scikit-learn not installed: skipping the ML baseline")
        return
    print(f"TF-IDF + LogReg (LOO)          accuracy {acc(pa, gold):6.1%}")
    print(f"TF-IDF + LogReg, honorifics masked     {acc(pb, gold):6.1%}")
    for name, p in (("TF-IDF", pa), ("TF-IDF masked", pb)):
        wrong = [t for t, x, g in zip(ids, p, gold) if x != g]
        print(f"   {name} errors ({len(wrong)}): {', '.join(wrong)}")


if __name__ == "__main__":
    main()
