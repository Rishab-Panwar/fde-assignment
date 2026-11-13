"""Deterministic combination of extracted evidence into the four final labels.

Implements outcome precedence (D6), valid parent/child pairs, lead-quality order
(LQ) and next_call_at rules (F2-F9). Pure functions: easy to unit-test and inspect.
"""
from __future__ import annotations

import re

from timeresolve import day_offset, plus_days_11am, resolve_callback

ALLOWED = {
    "Appointment Scheduled": ["Clinic Visit", "Online Consultation", "Rescheduled"],
    "Follow-up Required": ["Specific Callback", "Vague Deferral", "Family Discussion", "Cost Concern",
                           "Travel Constraint", "Current Treatment", "Treatment Concern",
                           "Information Requested", "Further Discussion"],
    "Not Interested": ["Explicit Refusal", "Do Not Contact"],
    "Invalid Lead": ["Wrong Contact", "No Pain Need", "Outside Service Area", "Policy Exclusion", "Test Enquiry"],
    "Unable to Assess": ["Incomplete Conversation"],
}
LEAD_QUALITIES = ["High", "Medium", "Low", "Not Qualified", "Not Assessable"]

INVALID_MAP = {
    "test_enquiry": "Test Enquiry",
    "wrong_contact": "Wrong Contact",
    "no_pain_need": "No Pain Need",
    "policy_exclusion": "Policy Exclusion",
    "outside_service_area": "Outside Service Area",
}
BLOCKER_MAP = {
    "family_discussion": "Family Discussion",
    "cost_concern": "Cost Concern",
    "travel_constraint": "Travel Constraint",
    "current_treatment": "Current Treatment",
    "treatment_concern": "Treatment Concern",
}

SUPPRESSED = {"Invalid Lead", "Not Interested", "Unable to Assess"}

# SUB: "Do Not Contact ... takes precedence over Explicit Refusal". The model sometimes quotes a
# stop-contacting request as its refusal evidence but still picks explicit_refusal (found by the
# synthetic robustness test, robustness/). Applied only to the model's OWN refusal evidence.
DNC_RE = re.compile(
    r"(remove|delete|take off)\b.{0,25}\b(my )?(number|contact|name)"
    r"|\b(don'?t|do not|never|stop)\s+(ever\s+)?(call|calling|contact|contacting|message|messaging|disturb)"
    r"|\b(call|phone)\s+(mat|na)\s+(karo|karna|kijiye)",
    re.I)
BLOCKER_ORDER = ["family_discussion", "cost_concern", "travel_constraint", "current_treatment", "treatment_concern"]


def select_blocker(blockers: list, trace: list) -> str:
    """Rule SUB tie-breaking: explicitly primary open reason, else the LATEST open reason,
    remaining ties by the sheet's order (Family, Cost, Travel, Current Treatment, Treatment Concern)."""
    open_ = [b for b in blockers or [] if b.get("still_open_at_end") and b.get("type") in BLOCKER_ORDER]
    if not open_:
        return "none"
    pool = [b for b in open_ if b.get("stated_as_main")] or open_
    last = max(b.get("line") or 0 for b in pool)
    cands = {b["type"] for b in pool if (b.get("line") or 0) == last}
    choice = min(cands, key=BLOCKER_ORDER.index)
    trace.append(f"blocker: open={[(b['type'], b.get('line')) for b in open_]} -> {choice}")
    return choice


def select_callback(proposals: list, trace: list) -> dict:
    """Rules F3/F9: the LAST proposal that was acknowledged and not cancelled/replaced later."""
    valid = [p for p in proposals or [] if p.get("acknowledged") and not p.get("cancelled_or_replaced_later")]
    if not valid:
        return {}
    final = max(valid, key=lambda p: p.get("line") or 0)
    trace.append(f"callback: final valid proposal at line {final.get('line')}: {final.get('evidence', '')[:60]}")
    return {"exists": True, "acknowledged": True, "day": final.get("day"), "time": final.get("time")}


def normalize(ex: dict, trace: list) -> dict:
    """Map a v3 (list-based) extraction onto the flat decision inputs used below.
    v1/v2 extractions are already flat and pass through unchanged."""
    if "blockers" not in ex:
        return ex
    ref = ex.get("refusal") or {}
    words = ref.get("words", "none")
    if words == "do_not_contact" and not ref.get("reversed_later"):
        refusal = "do_not_contact"
    elif words == "refusal" and not ref.get("conditional_or_frustration") and not ref.get("reversed_later"):
        refusal = "explicit_refusal"
    else:
        refusal = "none"
    if words != "none":
        trace.append(f"refusal words={words} conditional={ref.get('conditional_or_frustration')} "
                     f"reversed={ref.get('reversed_later')} -> {refusal}")
    info = ex.get("information_request") or {}
    vague = ex.get("vague_postponement") or {}
    flat = dict(ex)
    flat.update({
        "refusal": refusal,
        "primary_blocker": select_blocker(ex.get("blockers"), trace),
        "information_review_pending": bool(info.get("asked") and info.get("is_final_next_step")),
        "callback": select_callback(ex.get("callback_proposals"), trace),
        "vague_postponement": bool(vague.get("present")) if isinstance(vague, dict) else bool(vague),
    })
    return flat


def _majority(values: list):
    """Most common value; ties go to the value seen first (i.e. the lowest-seed sample)."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    return next(v for v in values if counts[v] == best)


def vote(labelled: list) -> dict:
    """Self-consistency vote over several independently labelled samples of the same call.

    Hierarchical so the result is always a coherent, valid record:
      1. majority disposition;
      2. sub_disposition, lead_quality and next_call_at by majority ONLY among samples that agree
         with that disposition (a sub-label can never come from a different parent);
      3. next_call_at forced to null for suppressed dispositions (F2).
    """
    disp = _majority([l["disposition"] for l in labelled])
    agree = [l for l in labelled if l["disposition"] == disp]
    out = {
        "disposition": disp,
        "sub_disposition": _majority([l["sub_disposition"] for l in agree]),
        "lead_quality": _majority([l["lead_quality"] for l in agree]),
        "next_call_at": None if disp in SUPPRESSED else _majority([l["next_call_at"] for l in agree]),
        "_trace": [],
    }
    assert out["sub_disposition"] in ALLOWED[disp]
    out["_vote"] = " | ".join(f"{k}: {[l[k] for l in labelled]}" for k in
                              ("disposition", "sub_disposition", "lead_quality", "next_call_at"))
    return out


def combine(ex: dict, call_started_at: str) -> dict:
    """ex = LLM extraction (see prompts.EXTRACTION_SCHEMA). Returns the four labels plus a trace."""
    trace = []
    ex = normalize(ex, trace)
    appt = ex.get("appointment") or {}
    cbk = ex.get("callback") or {}

    callback_at = None
    if cbk.get("exists") and cbk.get("acknowledged"):
        callback_at = resolve_callback(cbk.get("day"), cbk.get("time"), call_started_at)
        trace.append(f"callback resolved -> {callback_at}")

    # ---- disposition (D6 precedence) ----
    invalid = ex.get("invalid_reason", "none")
    refusal = ex.get("refusal", "none")
    if refusal == "explicit_refusal" and DNC_RE.search(ex.get("refusal_evidence") or ""):
        refusal = "do_not_contact"
        trace.append("refusal evidence contains a stop-contact request -> do_not_contact (SUB precedence)")
    if invalid in INVALID_MAP:
        disp, sub = "Invalid Lead", INVALID_MAP[invalid]
    elif refusal in ("do_not_contact", "explicit_refusal"):
        disp = "Not Interested"
        sub = "Do Not Contact" if refusal == "do_not_contact" else "Explicit Refusal"
    elif appt.get("confirmed"):
        disp = "Appointment Scheduled"
        if appt.get("is_reschedule"):
            sub = "Rescheduled"
        elif appt.get("modality") == "online":
            sub = "Online Consultation"
        else:
            sub = "Clinic Visit"
    elif ex.get("meaningful_dialogue"):
        disp = "Follow-up Required"
        blocker = ex.get("primary_blocker", "none")
        if blocker in BLOCKER_MAP:
            sub = BLOCKER_MAP[blocker]
        elif callback_at:
            sub = "Specific Callback"
        elif ex.get("information_review_pending"):
            sub = "Information Requested"
        elif ex.get("vague_postponement"):
            sub = "Vague Deferral"
        else:
            sub = "Further Discussion"
    else:
        disp, sub = "Unable to Assess", "Incomplete Conversation"

    # ---- lead quality (LQ order: Not Qualified, Not Assessable, Low, High, Medium) ----
    readiness = ex.get("readiness", "considering")
    offset = None
    if disp == "Appointment Scheduled":
        offset = day_offset(appt.get("day"), appt.get("time"), call_started_at)
        trace.append(f"appointment offset={offset}")
    if disp == "Invalid Lead":
        lq = "Not Qualified"
    elif disp == "Unable to Assess" or readiness == "not_assessable":
        lq = "Not Assessable"
    elif disp == "Not Interested" or readiness == "exploratory_only" or (
            ex.get("defers_beyond_30_days") and disp != "Appointment Scheduled"):
        lq = "Low"
    elif disp == "Appointment Scheduled" and (offset is None or 0 <= offset <= 6):
        lq = "High"
    else:
        lq = "Medium"

    # ---- next_call_at (F2-F9) ----
    if disp in SUPPRESSED:
        nxt = None
    elif callback_at:
        nxt = callback_at
    elif disp == "Appointment Scheduled":
        nxt = None
    elif ex.get("vague_postponement"):
        nxt = plus_days_11am(call_started_at, 3)
    else:
        nxt = plus_days_11am(call_started_at, 1)

    assert sub in ALLOWED[disp] and lq in LEAD_QUALITIES
    return {"disposition": disp, "sub_disposition": sub, "lead_quality": lq, "next_call_at": nxt, "_trace": trace}
