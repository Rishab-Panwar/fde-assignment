"""Prompt templates and the strict JSON schema for the evidence-extraction step.

The LLM answers small evidence-backed questions. It never outputs final labels or
absolute callback timestamps; combine.py and timeresolve.py derive those.
"""
from __future__ import annotations

PROMPT_VERSION = "v5"
# v5 = v4 + five rule-sheet-grounded edits for the patterns behind the remaining E4 errors:
# P1 blocker = reason attached to the FINAL next step (SUB); P2 checking a family member's schedule
# = travel, insurance/payment checks = cost; P3 "I'll let you know"/"I'll talk at home"/event-tied
# callbacks are vague postponements (F7, F12); P4 a later acknowledged specific callback overrides an
# earlier vague one (F3, F9); P5 frustration is not refusal (D2).
# v4 = v1 (E1 baseline) + exactly three targeted edits that fixed E1 dev errors in the v2 subset
# check without side effects: (1) callbacks need an explicit agreeing reply, (2) needing an escort /
# companion is travel_constraint, (3) waiting on any other medical treatment/follow-up is
# current_treatment. v2/v3 edits that caused regressions are deliberately NOT included.

SYSTEM_PROMPT = """You analyse one outbound/inbound call transcript from the enrollment team of Aster Vale Pain Clinic, a (hypothetical) Indian non-surgical pain clinic. The team helps people book a consultation with a pain specialist, for themselves or for a family member. Transcripts are English translations of Hindi/English calls; speaker labels (Agent/Customer) may be wrong in places, so infer roles from the content. Agent statements are claims, not facts. Do not diagnose.

Your job: answer a fixed set of factual questions about the call, quoting short evidence. Judge by the FINAL operative state at the end of the call: later corrections, reversals and amendments replace earlier statements. Use only this transcript; never assume earlier history or CRM state.

Answer each question as follows.

1. meaningful_dialogue — true if the call contains reliable, intelligible dialogue that establishes who the prospect is and something relevant about a pain-care enquiry or its outcome (including a clear wrong number or refusal). false only for silence, immediate disconnects, unresolved identity or unintelligible fragments. A long but unresolved discussion of a known pain need is meaningful.

2. invalid_reason — use a non-"none" value ONLY with reliable final evidence:
   - test_enquiry: explicit internal test, demo or fabricated enquiry.
   - wrong_contact: wrong person/number who has no connection to the enquiry. A relative or caregiver enquiring for a patient is NOT wrong contact.
   - no_pain_need: reliable evidence that neither the caller nor the intended patient has any pain-care need (e.g. enquiry was for something unrelated). Stiffness, nerve or joint complaints count as possible pain needs; uncertainty is not "no need".
   - policy_exclusion: the intended patient is known to be under 15 years old. Nothing else is a policy exclusion. Missing age or duration is not exclusion.
   - outside_service_area: permanently lives outside all serviceable areas AND travel AND online/remote care are explicitly unavailable or refused. Distance alone is never invalid.
   - none: otherwise.

3. refusal — the prospect's final stance:
   - do_not_contact: explicitly asks to stop calling / remove the number.
   - explicit_refusal: unconditional final rejection of the clinic's offering ("not interested", "I don't want this, don't need it") that is not reversed later.
   - none: anything else. A rejected date, price objection, "I'll think", "not now", "I'll call you myself", irritation, or treatment elsewhere is NOT a refusal. Conditional statements ("only if you tell me...") are not refusals. Frustration such as "forget it" / "leave it" / "chhodo" said because the agent did not answer the prospect's questions is NOT a refusal: the prospect still wants information.

4. appointment — was an actual consultation slot MUTUALLY CONFIRMED by the end of the call? confirmed=true requires both a specific date/day and a clock time or bounded time window (e.g. "Friday 11 AM", "tomorrow between 4 and 5"), and the prospect agrees to it and it is not withdrawn. An offer, available slots listed, "I'll try to come", "I'll see", visit intention, address request, "agent will call to confirm the timing" or a tentative plan is NOT confirmed. Report:
   - modality: "online" for a confirmed remote/video/phone consultation, else "clinic".
   - is_reschedule: true only if the confirmed slot replaces a previously scheduled or missed consultation.
   - day / time: the confirmed consultation day and time (structured, see DAY/TIME FORMAT). Use kind "none" if not confirmed.

5. primary_blocker — if no appointment is confirmed: the prospect's FINAL main unresolved reason for not committing now (not every historical objection). Anchor on the reason attached to the prospect's LAST stated next step near the end of the call (e.g. "send it, then I will talk at home" -> family_discussion even if distance was discussed earlier); objections raised earlier (distance, price, doubts) that the prospect does not tie to the final next step do not win. One of:
   - family_discussion: needs to discuss with / get approval from family about going ahead, or a family member decides. Only checking a family member's schedule, availability or timing ("I'll ask my wife if she can manage the delay") is travel_constraint, not family_discussion.
   - cost_concern: affordability, price, payment or insurance uncertainty is unresolved. Checking insurance/payment coverage (with an insurer, TPA, employer/office or insurance contact) is cost_concern even if the prospect also says they will tell family.
   - travel_constraint: travel, distance, transport, being out of town, work/business, availability of time, or clinic timing/schedule prevents committing now; also needing a family member to accompany the patient / waiting until that companion is free or back ("I can't come alone, I'll come when my husband returns"). Needing an escort is travel_constraint, not family_discussion.
   - current_treatment: the patient is waiting to finish, recover from, or assess another ongoing or scheduled medical treatment, procedure, surgery or follow-up (even for a different condition, e.g. a cataract follow-up) before deciding.
   - treatment_concern: unresolved doubts or mistrust about the treatment, procedure, injections, steroids, side effects, doctor suitability or outcome; or the prospect wants the clinic/doctor to first review reports and say whether they can be treated.
   - none: no named blocker (e.g. just agreed to talk again, or open enquiry).
   If several apply, pick the one the prospect states as the main/final reason; if truly equal, prefer in this order: family_discussion, cost_concern, travel_constraint, current_treatment, treatment_concern. Use "none" if an appointment was confirmed.

6. information_review_pending — true only if the prospect's final next step is to review information or reports they specifically requested (doctor profile, treatment details, etc.). Merely receiving the address/location on WhatsApp does not count.

7. readiness — the intended patient's readiness to consult:
   - committed: explicitly agreed to a feasible consultation (normally the confirmed slot).
   - considering: real pain need and genuine consideration, but undecided or blocked. This is the default for an interested but undecided prospect.
   - exploratory_only: only curiosity / information gathering with no current intent to consult, or clearly does not want the clinic's treatment, or final refusal.
   - not_assessable: relevant discussion happened but the patient's need or willingness cannot be established reliably (e.g. the prospect never engages with the offer, conversation breaks down before need/intent is clear).
   Severe pain alone does not raise readiness.

8. defers_beyond_30_days — true if the prospect says they will not consider/come for more than 30 days (e.g. "after a month or two", "after 2 months") and no consultation is confirmed.

9. callback — the FINAL agreed time for the NEXT CALL/CONVERSATION (not the consultation). Rules:
   - A future callback time proposed by either side and EXPLICITLY acknowledged by the other with an agreeing word or answer ("okay", "yes", "haan", "fine", "sure", "theek hai", "call me then"). Simply moving on to another topic without agreeing is NOT acknowledgment (e.g. Agent: "If you plan Friday, we will call you Wednesday to confirm." Customer: "Pardon?" ... -> not acknowledged). A customer stating a time that the agent ignores is not enough. Set acknowledged=false when no agreeing reply exists.
   - A later acknowledged amendment replaces earlier times, and a later acknowledged specific proposal replaces an earlier vague one (e.g. "I'll call after Diwali" ... Agent: "Should I call next week?" Customer: "Yes, call" -> day kind next_week). If the callback is finally cancelled or replaced by something vague ("I'll call you myself", "later"), set exists=false.
   - A consultation slot, "I'll send details at 5", clinic hours, doctor availability, or "you may receive a call" is NOT a callback.
   - "I'll take a follow-up from you by tomorrow evening" + okay IS a callback (tomorrow, evening). A definite "will call you tomorrow to confirm" + okay is a callback.
   - If the agent says something like "I'll call you once" or "I will call" with no day or time, exists=false.
   - Windows or alternatives ("the 4th or 5th", "tomorrow or the day after", "between 4 and 5"): report the EARLIEST one. "After 3 pm" means 3 pm. "Around Monday" means Monday. "After the 25th, so shall I call on the 26th? -okay" means the 26th.
   - "After next week", "after the festival", "after I return" without an agreed day is vague: exists=false.
   Give day/time as written (structured; see DAY/TIME FORMAT), not as computed dates.

10. vague_postponement — true if, at the end, the prospect explicitly postpones the decision or next conversation without an agreed resolvable day/time: "call later", "let me think", "I'll see and let you know", "I will let you know", "send the details, I'll talk at home and tell you", "I'll call you myself / when I'm able", "after my trip/family event", "not now, maybe next month". This applies even when a blocker explains the delay. An acknowledged callback tied to an unresolved event ("I'll call you next week after Dussehra" - "okay") is ALSO a vague postponement (and not a specific callback). An agent's routine "I'll call you"/"I'll send details" alone is NOT vague postponement. Only a rejected specific callback is not a vague postponement either.

DAY/TIME FORMAT (describe what was said; do NOT compute dates):
day.kind is one of:
  none (no day stated) | today | tomorrow | day_after_tomorrow (incl. "parso" in a forward-looking context) | in_n_days (n) |
  weekday (bare weekday name, e.g. "on Wednesday") | this_weekday ("this Friday") | next_weekday ("next Tuesday") |
  next_week ("next week" with no day) | this_week ("this week" with no day; vague) |
  absolute_date (day_of_month, optional month 1-12, optional year; e.g. "the 26th" -> day_of_month 26, month null) |
  in_n_months (n) | next_month ("next month" with no day) |
  in_n_minutes (n) | in_n_hours (n) | vague (any other non-resolvable reference such as "after Diwali", "after next week")
  For weekday kinds set weekday to the lowercase English name; otherwise "none". Unused numeric fields are null.
time.kind is one of: none | clock (hour 0-23 as said, minute, ampm am/pm/unspecified) | morning | afternoon | evening (incl. tonight) | lunch.
  For "4 o'clock" give hour 4, ampm unspecified. For "16:00" give hour 16. For "5:30 in the evening" give clock hour 5 minute 30 ampm pm.

Keep every evidence field to a short verbatim quote (under 25 words) or "" when not applicable."""

USER_TEMPLATE = """Call metadata
- transcript_id: {transcript_id}
- call_started_at: {call_started_at} ({weekday}) — timezone {timezone}

Transcript:
<<<
{transcript}
>>>

Answer all questions in the required JSON format."""


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

EXTRACTION_SCHEMA = _obj({
    "patient_summary": {"type": "string"},
    "meaningful_dialogue": {"type": "boolean"},
    "invalid_reason": {"type": "string", "enum": [
        "none", "test_enquiry", "wrong_contact", "no_pain_need", "policy_exclusion", "outside_service_area"]},
    "invalid_evidence": _EVID,
    "refusal": {"type": "string", "enum": ["none", "explicit_refusal", "do_not_contact"]},
    "refusal_evidence": _EVID,
    "appointment": _obj({
        "evidence": _EVID,
        "confirmed": {"type": "boolean"},
        "modality": {"type": "string", "enum": ["clinic", "online"]},
        "is_reschedule": {"type": "boolean"},
        "day": _DAY,
        "time": _TIME,
    }),
    "primary_blocker": {"type": "string", "enum": [
        "none", "family_discussion", "cost_concern", "travel_constraint", "current_treatment", "treatment_concern"]},
    "blocker_evidence": _EVID,
    "information_review_pending": {"type": "boolean"},
    "readiness": {"type": "string", "enum": ["committed", "considering", "exploratory_only", "not_assessable"]},
    "defers_beyond_30_days": {"type": "boolean"},
    "callback": _obj({
        "evidence": _EVID,
        "exists": {"type": "boolean"},
        "acknowledged": {"type": "boolean"},
        "day": _DAY,
        "time": _TIME,
    }),
    "vague_postponement": {"type": "boolean"},
    "vague_evidence": _EVID,
})
