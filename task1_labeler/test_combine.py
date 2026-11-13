"""Unit tests for combine.py: precedence, valid pairs, lead quality and next_call_at policy.

Covers dispositions absent from the 50 labelled calls (Invalid Lead, Do Not Contact,
Unable to Assess) using hand-built extractions, i.e. synthetic tests, not training data.
Run: python -m unittest test_combine -v
"""
import copy
import unittest

from combine import ALLOWED, combine, vote

CALL = "2026-11-09T14:00:00+05:30"  # Monday
NODAY = {"kind": "none", "n": None, "weekday": "none", "day_of_month": None, "month": None, "year": None}
NOTIME = {"kind": "none", "hour": None, "minute": None, "ampm": "unspecified"}
BASE = {
    "meaningful_dialogue": True, "invalid_reason": "none", "refusal": "none",
    "appointment": {"confirmed": False, "modality": "clinic", "is_reschedule": False, "day": NODAY, "time": NOTIME},
    "primary_blocker": "none", "information_review_pending": False, "readiness": "considering",
    "defers_beyond_30_days": False,
    "callback": {"exists": False, "acknowledged": False, "day": NODAY, "time": NOTIME},
    "vague_postponement": False,
}


def ex(**kw):
    e = copy.deepcopy(BASE)
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(e.get(k), dict):
            e[k].update(v)
        else:
            e[k] = v
    return e


def day(**kw):
    return dict(NODAY, **kw)


def tm(**kw):
    return dict(NOTIME, **kw)


TOMORROW_4 = {"exists": True, "acknowledged": True, "day": day(kind="tomorrow"), "time": tm(kind="clock", hour=4)}


class TestPrecedence(unittest.TestCase):
    def test_invalid_beats_refusal_and_callback(self):
        r = combine(ex(invalid_reason="wrong_contact", refusal="explicit_refusal", callback=TOMORROW_4), CALL)
        self.assertEqual((r["disposition"], r["sub_disposition"], r["lead_quality"], r["next_call_at"]),
                         ("Invalid Lead", "Wrong Contact", "Not Qualified", None))

    def test_under_15(self):
        r = combine(ex(invalid_reason="policy_exclusion"), CALL)
        self.assertEqual(r["sub_disposition"], "Policy Exclusion")

    def test_ex5_do_not_contact_suppresses_callback(self):
        r = combine(ex(refusal="do_not_contact", callback=TOMORROW_4), CALL)
        self.assertEqual((r["disposition"], r["sub_disposition"], r["lead_quality"], r["next_call_at"]),
                         ("Not Interested", "Do Not Contact", "Low", None))

    def test_dnc_in_refusal_evidence_upgrades(self):
        for q in ("I am not interested. Please remove my number and do not call me again.",
                  "Don't call me again", "aage se call mat karo"):
            r = combine(ex(refusal="explicit_refusal", refusal_evidence=q), CALL)
            self.assertEqual(r["sub_disposition"], "Do Not Contact", q)

    def test_plain_refusal_stays_refusal(self):
        for q in ("I don't want this treatment. I am not interested.", "No, I don't need it",
                  "I will call you myself if needed"):
            r = combine(ex(refusal="explicit_refusal", refusal_evidence=q), CALL)
            self.assertEqual(r["sub_disposition"], "Explicit Refusal", q)

    def test_refusal_beats_appointment(self):
        r = combine(ex(refusal="explicit_refusal", appointment={"confirmed": True, "day": day(kind="tomorrow")}), CALL)
        self.assertEqual(r["disposition"], "Not Interested")

    def test_unable_to_assess(self):
        r = combine(ex(meaningful_dialogue=False, callback=TOMORROW_4), CALL)
        self.assertEqual((r["disposition"], r["sub_disposition"], r["lead_quality"], r["next_call_at"]),
                         ("Unable to Assess", "Incomplete Conversation", "Not Assessable", None))


class TestAppointment(unittest.TestCase):
    def test_ex3_appointment_no_callback(self):
        r = combine(ex(readiness="committed", appointment={"confirmed": True, "day": day(kind="tomorrow"),
                                                           "time": tm(kind="clock", hour=15)}), CALL)
        self.assertEqual((r["disposition"], r["sub_disposition"], r["lead_quality"], r["next_call_at"]),
                         ("Appointment Scheduled", "Clinic Visit", "High", None))

    def test_ex6_exactly_7_days_is_medium(self):
        r = combine(ex(appointment={"confirmed": True, "day": day(kind="in_n_days", n=7)}), CALL)
        self.assertEqual(r["lead_quality"], "Medium")

    def test_day_6_is_high(self):
        r = combine(ex(appointment={"confirmed": True, "day": day(kind="in_n_days", n=6)}), CALL)
        self.assertEqual(r["lead_quality"], "High")

    def test_reschedule_beats_modality(self):
        r = combine(ex(appointment={"confirmed": True, "is_reschedule": True, "modality": "online",
                                    "day": day(kind="tomorrow")}), CALL)
        self.assertEqual(r["sub_disposition"], "Rescheduled")

    def test_online(self):
        r = combine(ex(appointment={"confirmed": True, "modality": "online", "day": day(kind="tomorrow")}), CALL)
        self.assertEqual(r["sub_disposition"], "Online Consultation")

    def test_distinct_callback_coexists_with_appointment(self):  # F12
        r = combine(ex(appointment={"confirmed": True, "day": day(kind="day_after_tomorrow")},
                       callback=TOMORROW_4), CALL)
        self.assertEqual(r["next_call_at"], "2026-11-10T16:00:00+05:30")


class TestFollowUp(unittest.TestCase):
    def test_ex1_specific_callback(self):
        r = combine(ex(callback=TOMORROW_4), CALL)
        self.assertEqual((r["sub_disposition"], r["next_call_at"]), ("Specific Callback", "2026-11-10T16:00:00+05:30"))

    def test_blocker_owns_sub_but_callback_time_kept(self):  # D4
        r = combine(ex(primary_blocker="cost_concern", callback=TOMORROW_4), CALL)
        self.assertEqual((r["sub_disposition"], r["next_call_at"]), ("Cost Concern", "2026-11-10T16:00:00+05:30"))

    def test_ex2_vague_plus3(self):
        r = combine(ex(vague_postponement=True), CALL)
        self.assertEqual((r["sub_disposition"], r["next_call_at"]), ("Vague Deferral", "2026-11-12T11:00:00+05:30"))

    def test_vague_plus3_even_with_blocker(self):  # F7
        r = combine(ex(primary_blocker="family_discussion", vague_postponement=True), CALL)
        self.assertEqual((r["sub_disposition"], r["next_call_at"]), ("Family Discussion", "2026-11-12T11:00:00+05:30"))

    def test_ex4_default_plus1(self):
        r = combine(ex(), CALL)
        self.assertEqual((r["sub_disposition"], r["lead_quality"], r["next_call_at"]),
                         ("Further Discussion", "Medium", "2026-11-10T11:00:00+05:30"))

    def test_callback_before_information_requested(self):
        r = combine(ex(callback=TOMORROW_4, information_review_pending=True), CALL)
        self.assertEqual(r["sub_disposition"], "Specific Callback")

    def test_information_requested(self):
        r = combine(ex(information_review_pending=True, vague_postponement=True), CALL)
        self.assertEqual(r["sub_disposition"], "Information Requested")

    def test_unacknowledged_callback_ignored(self):  # F3
        r = combine(ex(callback=dict(TOMORROW_4, acknowledged=False)), CALL)
        self.assertEqual(r["next_call_at"], "2026-11-10T11:00:00+05:30")

    def test_past_callback_not_repaired(self):  # F6: "today morning" on a 14:00 call
        cb = {"exists": True, "acknowledged": True, "day": day(kind="today"), "time": tm(kind="morning")}
        r = combine(ex(callback=cb), CALL)
        self.assertEqual((r["sub_disposition"], r["next_call_at"]), ("Further Discussion", "2026-11-10T11:00:00+05:30"))

    def test_defer_beyond_30_days_is_low(self):
        r = combine(ex(primary_blocker="current_treatment", defers_beyond_30_days=True, vague_postponement=True), CALL)
        self.assertEqual(r["lead_quality"], "Low")

    def test_exploratory_is_low(self):
        r = combine(ex(readiness="exploratory_only"), CALL)
        self.assertEqual(r["lead_quality"], "Low")

    def test_not_assessable(self):
        r = combine(ex(readiness="not_assessable"), CALL)
        self.assertEqual((r["disposition"], r["lead_quality"]), ("Follow-up Required", "Not Assessable"))


def v3(**kw):
    """Minimal v3 (list-based) extraction."""
    e = {
        "meaningful_dialogue": True, "invalid_reason": "none",
        "refusal": {"words": "none", "evidence": "", "line": 0, "conditional_or_frustration": False,
                    "reversed_later": False},
        "appointment": copy.deepcopy(BASE["appointment"]), "blockers": [],
        "information_request": {"asked": False, "what": "", "line": 0, "is_final_next_step": False},
        "readiness": "considering", "defers_beyond_30_days": False, "callback_proposals": [],
        "vague_postponement": {"present": False, "evidence": "", "line": 0},
    }
    e.update(kw)
    return e


def blk(t, line, open_=True, main=False):
    return {"type": t, "evidence": "", "line": line, "still_open_at_end": open_, "stated_as_main": main}


def prop(line, ack=True, cancelled=False, **daykw):
    return {"proposer": "agent", "evidence": "", "line": line, "acknowledged": ack, "ack_evidence": "",
            "cancelled_or_replaced_later": cancelled, "day": day(**daykw), "time": NOTIME}


class TestV3Selection(unittest.TestCase):
    def test_latest_open_blocker_wins(self):
        r = combine(v3(blockers=[blk("travel_constraint", 20), blk("family_discussion", 60)]), CALL)
        self.assertEqual(r["sub_disposition"], "Family Discussion")

    def test_resolved_blocker_ignored(self):
        r = combine(v3(blockers=[blk("cost_concern", 80, open_=False), blk("travel_constraint", 30)]), CALL)
        self.assertEqual(r["sub_disposition"], "Travel Constraint")

    def test_stated_main_beats_recency(self):
        r = combine(v3(blockers=[blk("cost_concern", 20, main=True), blk("family_discussion", 60)]), CALL)
        self.assertEqual(r["sub_disposition"], "Cost Concern")

    def test_same_line_tie_uses_sheet_order(self):
        r = combine(v3(blockers=[blk("travel_constraint", 50), blk("family_discussion", 50)]), CALL)
        self.assertEqual(r["sub_disposition"], "Family Discussion")

    def test_last_acknowledged_callback_wins(self):
        r = combine(v3(callback_proposals=[prop(10, kind="tomorrow"), prop(40, kind="next_week")]), CALL)
        self.assertEqual(r["next_call_at"], "2026-11-16T11:00:00+05:30")

    def test_unacknowledged_later_proposal_does_not_replace(self):
        r = combine(v3(callback_proposals=[prop(10, kind="tomorrow"), prop(40, ack=False, kind="next_week")]), CALL)
        self.assertEqual(r["next_call_at"], "2026-11-10T11:00:00+05:30")

    def test_cancelled_callback_falls_back_to_vague(self):
        r = combine(v3(callback_proposals=[prop(10, cancelled=True, kind="tomorrow")],
                       vague_postponement={"present": True, "evidence": "", "line": 50}), CALL)
        self.assertEqual((r["sub_disposition"], r["next_call_at"]), ("Vague Deferral", "2026-11-12T11:00:00+05:30"))

    def test_conditional_refusal_is_not_refusal(self):
        ref = {"words": "refusal", "evidence": "forget it", "line": 30, "conditional_or_frustration": True,
               "reversed_later": False}
        r = combine(v3(refusal=ref, blockers=[blk("treatment_concern", 28)]), CALL)
        self.assertEqual((r["disposition"], r["sub_disposition"]), ("Follow-up Required", "Treatment Concern"))

    def test_unconditional_refusal(self):
        ref = {"words": "refusal", "evidence": "", "line": 30, "conditional_or_frustration": False,
               "reversed_later": False}
        self.assertEqual(combine(v3(refusal=ref), CALL)["sub_disposition"], "Explicit Refusal")

    def test_reversed_refusal_ignored(self):
        ref = {"words": "refusal", "evidence": "", "line": 30, "conditional_or_frustration": False,
               "reversed_later": True}
        self.assertEqual(combine(v3(refusal=ref), CALL)["disposition"], "Follow-up Required")

    def test_info_requested_needs_final_next_step(self):
        info = {"asked": True, "what": "doctor profiles", "line": 70, "is_final_next_step": True}
        self.assertEqual(combine(v3(information_request=info), CALL)["sub_disposition"], "Information Requested")
        info["is_final_next_step"] = False
        self.assertEqual(combine(v3(information_request=info), CALL)["sub_disposition"], "Further Discussion")


def lab(d, s, lq, n):
    return {"disposition": d, "sub_disposition": s, "lead_quality": lq, "next_call_at": n, "_trace": []}


class TestVote(unittest.TestCase):
    def test_majority_per_field(self):
        r = vote([lab("Follow-up Required", "Travel Constraint", "Medium", "A"),
                  lab("Follow-up Required", "Family Discussion", "Medium", "B"),
                  lab("Follow-up Required", "Travel Constraint", "Medium", "B")])
        self.assertEqual((r["sub_disposition"], r["next_call_at"]), ("Travel Constraint", "B"))

    def test_sub_only_from_agreeing_parent(self):
        r = vote([lab("Not Interested", "Explicit Refusal", "Low", None),
                  lab("Follow-up Required", "Treatment Concern", "Not Assessable", "X"),
                  lab("Follow-up Required", "Cost Concern", "Medium", "X")])
        self.assertEqual(r["disposition"], "Follow-up Required")
        self.assertIn(r["sub_disposition"], ("Treatment Concern", "Cost Concern"))
        self.assertEqual(r["next_call_at"], "X")

    def test_three_way_tie_uses_first_sample(self):
        r = vote([lab("Follow-up Required", "Cost Concern", "Medium", "A"),
                  lab("Follow-up Required", "Travel Constraint", "Medium", "B"),
                  lab("Follow-up Required", "Family Discussion", "Medium", "C")])
        self.assertEqual((r["sub_disposition"], r["next_call_at"]), ("Cost Concern", "A"))

    def test_suppressed_disposition_nulls_callback(self):
        r = vote([lab("Not Interested", "Explicit Refusal", "Low", None),
                  lab("Not Interested", "Explicit Refusal", "Low", None),
                  lab("Follow-up Required", "Vague Deferral", "Medium", "A")])
        self.assertIsNone(r["next_call_at"])


class TestValidity(unittest.TestCase):
    def test_every_output_pair_is_allowed(self):
        variants = [ex(invalid_reason=v) for v in ("test_enquiry", "wrong_contact", "no_pain_need",
                                                   "policy_exclusion", "outside_service_area")]
        variants += [ex(primary_blocker=b) for b in ("family_discussion", "cost_concern", "travel_constraint",
                                                     "current_treatment", "treatment_concern")]
        for v in variants:
            r = combine(v, CALL)
            self.assertIn(r["sub_disposition"], ALLOWED[r["disposition"]])


if __name__ == "__main__":
    unittest.main()
