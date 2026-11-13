"""Prompt templates and the strict JSON schema for the evidence-extraction step (latest = v3).

The LLM EXTRACTS evidence; Python DECIDES. The model never outputs final labels or
absolute callback timestamps; combine.py and timeresolve.py derive those.

v3 (structural change, see experiments.md): instead of asking the model to pick the single
"final main" blocker / callback / refusal, it lists every candidate with its transcript line
number and simple yes/no facts. combine.py then applies the policy literally:
  - blocker: explicitly-primary open reason, else the LATEST open one, ties by sheet order
  - callback: the LAST acknowledged, not-cancelled proposal (F3, F9)
  - refusal: only if unconditional, not frustration, not reversed (D2)
Transcript lines are numbered during preprocessing so positions are reliable. The prompt is kept
compact so request + schema stay under an 8k-token per-request limit (Groq free tier).
Earlier versions are frozen in prompts_v1.py / prompts_v2.py.
"""
from __future__ import annotations

PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """You analyse one call transcript from the enrollment team of Aster Vale Pain Clinic, a hypothetical Indian non-surgical pain clinic that books pain-specialist consultations for callers or their family members. Transcripts are English translations of Hindi/English calls; lines are numbered [n]. Speaker labels may be wrong; infer roles from content. Agent statements are claims, not facts. No diagnosis.

Your job is EXTRACTION: report facts with short verbatim evidence (<25 words) and the line [n]. Use only this transcript. If something is said several times, give the LAST line.

A. meaningful_dialogue: true if reliable dialogue establishes the prospect and something relevant about a pain-care enquiry or outcome (incl. a clear wrong number or refusal). false only for silence, immediate disconnect, unresolved identity or unintelligible fragments.

B. invalid_reason (only with reliable final evidence, else none): test_enquiry (explicit test/demo/fabricated); wrong_contact (wrong person/number unconnected to the enquiry; a relative enquiring for a patient is NOT wrong); no_pain_need (neither caller nor patient has any pain-care need; stiffness/nerve/joint complaints count as possible pain; uncertainty is not "no need"); policy_exclusion (patient known to be under 15; nothing else); outside_service_area (permanently unserviceable AND travel AND online care explicitly unavailable/refused; distance alone never).

C. refusal, as facts. words: do_not_contact (asks to stop calling / remove number), refusal (does not want the clinic's consultation/treatment: "not interested", "don't want this"), else none. conditional_or_frustration: true if it depends on a condition ("only if you tell me"), concerns only a date/price/procedure, or is frustration at unanswered questions ("forget it" while still asking what the treatment is). reversed_later: true if renewed interest or an agreed next step follows.

D. appointment: confirmed=true only if a consultation day AND clock time or bounded window ("Friday 11 AM", "tomorrow 4-5") is mutually agreed by the end and not withdrawn. Offers, listed slots, "I'll try/see", visit intentions, address requests, "we'll call to confirm the timing" are NOT confirmed. modality online for remote consultation else clinic; is_reschedule if it replaces a previous/missed consultation; day/time of the slot (kind none if not confirmed).

E. blockers: list EVERY reason the prospect gives for not booking now, one entry per distinct reason, at its LAST line. Types:
 family_discussion = needs discussion/approval/decision from family ("I'll talk at home", "ask my son").
 cost_concern = price, affordability, payment, or insurance/coverage uncertainty, incl. checking coverage with insurer, TPA, office or insurance contact.
 travel_constraint = travel, distance, transport, out of town, work/business/holidays, no time, clinic timing, OR needing someone to accompany the patient / waiting for that companion (an escort need is travel, not family).
 current_treatment = waiting to finish, recover from or assess another ongoing/scheduled treatment, surgery or follow-up (even for another condition).
 treatment_concern = doubts/mistrust about treatment, injections, steroids, side effects, doctor suitability or outcome, or wanting reports reviewed before deciding.
 still_open_at_end: true if it is still unresolved and holding them back at the end (false if resolved, dropped, or an appointment was confirmed). stated_as_main: true only if explicitly called THE main/only reason. Empty list if none.

F. information_request: asked=true only if the PROSPECT explicitly asked for specific information to review (doctor profiles, procedure details, documents, report review). Address, location, timings, fees, contact numbers or agent-volunteered info do NOT count. is_final_next_step: reviewing it is their last stated next step.

G. readiness: committed (agreed to a feasible consultation); considering (real need, genuine consideration, undecided or blocked; default); exploratory_only (curiosity only, or clearly does not want the clinic's treatment); not_assessable (need or willingness cannot be established: never engages, unclear if it is a pain need, conversation breaks down; prefer over exploratory_only when unsure). Severe pain alone does not raise readiness.

H. defers_beyond_30_days: says they will not come/consider for more than 30 days and nothing is confirmed.

I. callback_proposals: list EVERY proposed future time for the NEXT CALL/CONVERSATION (not the consultation), with proposer, line, evidence. acknowledged=true only if the OTHER side explicitly agrees ("okay", "yes", "haan", "fine", "sure", "yes, call"); moving on, "pardon?" or silence is not agreement; put the agreeing words in ack_evidence. cancelled_or_replaced_later: a later statement cancels it, gives another time, or makes it vague ("I'll call you myself"). day/time as SAID: windows/alternatives -> EARLIEST; "after 3 pm" -> 3 pm; "around Monday" -> Monday; "Should I call next week? - Yes" -> next_week; "after Diwali/next week/I return" -> vague. NOT callbacks: consultation slots, sending details at a time, clinic hours, doctor availability, "you may get a call", "I'll call you" with no day/time. "I'll take a follow-up from you tomorrow evening - okay" IS one. Empty list if none.

J. vague_postponement: present=true if at the end the prospect postpones without an agreed resolvable time ("call later", "let me think", "I'll see and tell you", "I need some time", "I'll call you myself", "after my trip", "maybe next month"). An agent's routine "I'll call/send details" alone is not one.

DAY/TIME FORMAT (as said; never compute dates). day.kind: none | today | tomorrow | day_after_tomorrow (incl. forward "parso") | in_n_days(n) | weekday (bare "on Wednesday") | this_weekday | next_weekday | next_week | this_week | absolute_date(day_of_month, month 1-12 or null, year or null; "the 26th" -> 26, null) | in_n_months(n) | next_month | in_n_minutes(n) | in_n_hours(n) | vague. weekday = lowercase name for weekday kinds else "none"; unused numbers null.
time.kind: none | clock(hour as said 0-23, minute, ampm am/pm/unspecified; "4 o'clock" -> 4 unspecified; "16:00" -> 16) | morning | afternoon | evening (incl. tonight) | lunch."""

USER_TEMPLATE = """Call metadata
- transcript_id: {transcript_id}
- call_started_at: {call_started_at} ({weekday}) — timezone {timezone}

Transcript (lines numbered):
<<<
{transcript}
>>>

Extract all fields in the required JSON format."""

NUMBER_LINES = True  # run.build_messages prefixes each transcript line with [n]


_DAY = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "n", "weekday", "day_of_month", "month", "year"],
    "properties": {
        "kind": {"type": "string", "enum": [
            "none", "today", "tomorrow", "day_after_tomorrow", "in_n_days", "weekday", "this_weekday",
            "next_weekday", "next_week", "this_week", "absolute_date", "in_n_months", "next_month",
            "in_n_minutes", "in_n_hours", "vague"]},
        "n": {"type": ["integer", "null"]},
        "weekday": {"type": "string", "enum": [
            "none", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]},
        "day_of_month": {"type": ["integer", "null"]},
        "month": {"type": ["integer", "null"]},
        "year": {"type": ["integer", "null"]},
    },
}

_TIME = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "hour", "minute", "ampm"],
    "properties": {
        "kind": {"type": "string", "enum": ["none", "clock", "morning", "afternoon", "evening", "lunch"]},
        "hour": {"type": ["integer", "null"]},
        "minute": {"type": ["integer", "null"]},
        "ampm": {"type": "string", "enum": ["am", "pm", "unspecified"]},
    },
}


def _obj(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


_EVID = {"type": "string"}
_LINE = {"type": "integer"}
BLOCKER_TYPES = ["family_discussion", "cost_concern", "travel_constraint", "current_treatment", "treatment_concern"]

EXTRACTION_SCHEMA = _obj({
    "meaningful_dialogue": {"type": "boolean"},
    "invalid_reason": {"type": "string", "enum": [
        "none", "test_enquiry", "wrong_contact", "no_pain_need", "policy_exclusion", "outside_service_area"]},
    "invalid_evidence": _EVID,
    "refusal": _obj({
        "words": {"type": "string", "enum": ["none", "refusal", "do_not_contact"]},
        "evidence": _EVID,
        "line": _LINE,
        "conditional_or_frustration": {"type": "boolean"},
        "reversed_later": {"type": "boolean"},
    }),
    "appointment": _obj({
        "evidence": _EVID,
        "confirmed": {"type": "boolean"},
        "modality": {"type": "string", "enum": ["clinic", "online"]},
        "is_reschedule": {"type": "boolean"},
        "day": _DAY,
        "time": _TIME,
    }),
    "blockers": {"type": "array", "items": _obj({
        "type": {"type": "string", "enum": BLOCKER_TYPES},
        "evidence": _EVID,
        "line": _LINE,
        "still_open_at_end": {"type": "boolean"},
        "stated_as_main": {"type": "boolean"},
    })},
    "information_request": _obj({
        "asked": {"type": "boolean"},
        "what": _EVID,
        "line": _LINE,
        "is_final_next_step": {"type": "boolean"},
    }),
    "readiness": {"type": "string", "enum": ["committed", "considering", "exploratory_only", "not_assessable"]},
    "defers_beyond_30_days": {"type": "boolean"},
    "callback_proposals": {"type": "array", "items": _obj({
        "proposer": {"type": "string", "enum": ["agent", "customer"]},
        "evidence": _EVID,
        "line": _LINE,
        "acknowledged": {"type": "boolean"},
        "ack_evidence": _EVID,
        "cancelled_or_replaced_later": {"type": "boolean"},
        "day": _DAY,
        "time": _TIME,
    })},
    "vague_postponement": _obj({
        "present": {"type": "boolean"},
        "evidence": _EVID,
        "line": _LINE,
    }),
})
