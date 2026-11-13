"""Synthetic robustness cases for outcomes ABSENT from the 50 labelled calls.

These are TESTS ONLY: they are never added to the prompt or used to tune anything. Each case is
written to have one unambiguous answer under the policy sheet (rule IDs noted). The call time is
Monday 2026-11-09 14:00:00 IST (the rule sheet's own boundary-example date).

Usage: python robustness/make_cases.py  -> robustness/cases.jsonl (inputs) + robustness/expected.jsonl
"""
import json
from pathlib import Path

CALL = "2026-11-09T14:00:00+05:30"
HERE = Path(__file__).parent

CASES = [
    # id, transcript, expected (disposition, sub, lead_quality, next_call_at), rule
    ("SYN-01", """Agent: Hello, I am Riya calling from Aster Vale Pain Clinic. Am I speaking with Mr. Anil?
Customer: No, there is no Anil here.
Agent: Sir, we received an enquiry from this number about knee pain treatment.
Customer: I didn't make any enquiry. This is a hardware shop number. Nobody here has any pain problem, you have the wrong number.
Agent: Sorry for the trouble, sir. I will update our records.
Customer: Okay.""", ("Invalid Lead", "Wrong Contact", "Not Qualified", None), "D1 / SUB Wrong Contact"),

    ("SYN-02", """Agent: Hello, this is Neel from Aster Vale Care. You submitted a form for back pain?
Customer: Hi Neel, this is Kabir from the Aster Vale tech team. I filled that form myself just to test the new website lead form. There is no patient, it's a test entry.
Agent: Oh okay, got it. So I should close this lead?
Customer: Yes, please mark it as a test. Thanks.
Agent: Done, thank you.""", ("Invalid Lead", "Test Enquiry", "Not Qualified", None), "D1 / SUB Test Enquiry"),

    ("SYN-03", """Agent: Hello ma'am, I am Tara from Aster Vale Pain Clinic. You enquired about knee pain?
Customer: Yes, it's for my son. He has knee pain after playing football.
Agent: Okay ma'am. How old is your son?
Customer: He is 12 years old. He is in class 7.
Agent: Ma'am, our clinic currently sees patients who are 15 years or older, so we will not be able to take this consultation.
Customer: Oh, okay. I will show him to a paediatric doctor then.
Agent: Yes ma'am, that would be best. Thank you.""", ("Invalid Lead", "Policy Exclusion", "Not Qualified", None), "D1 / SUB Policy Exclusion (patient under 15)"),

    ("SYN-04", """Agent: Hello sir, Ishan here from Aster Vale Pain Clinic, regarding your back pain enquiry.
Customer: Look, I have already told your people twice. I am not interested.
Agent: Sir, just two minutes, the consultation is only...
Customer: No. Please remove my number from your list and do not call me again.
Agent: Okay sir, I will remove your number. Sorry for the disturbance.
Customer: Okay, bye.""", ("Not Interested", "Do Not Contact", "Low", None), "D2 / SUB Do Not Contact"),

    ("SYN-05", """Agent: Hello ma'am, I am Priya from Aster Vale Care. You had enquired about shoulder pain treatment.
Customer: Yes, I did enquire last week.
Agent: Ma'am, would you like to book a consultation with our pain specialist?
Customer: No. I have thought about it and I have decided I don't want this treatment. I am not interested.
Agent: Okay ma'am, may I know the reason?
Customer: I just don't want it. Thank you.
Agent: Okay ma'am, thank you for your time.""", ("Not Interested", "Explicit Refusal", "Low", None), "D2 / SUB Explicit Refusal"),

    ("SYN-06", """Agent: Hello?
Agent: Hello, am I audible?
Customer: Hello?
Agent: Hello, I am calling from Aster Vale...
Customer: Hello? Hello?
Agent: Hello?""", ("Unable to Assess", "Incomplete Conversation", "Not Assessable", None), "D5 / Incomplete Conversation"),

    ("SYN-07", """Agent: Hello sir, Mehul from Aster Vale Pain Clinic. You enquired about lower back pain?
Customer: Yes. I live in another city, so I cannot come to the clinic right now. Can I talk to the doctor online first?
Agent: Yes sir, we have an online video consultation. The doctor is available tomorrow at 5 PM. Shall I book it?
Customer: Yes, book the video consultation for tomorrow at 5 PM.
Agent: Done sir, your online consultation is confirmed for tomorrow at 5 PM. I will send the video link on WhatsApp.
Customer: Okay, thank you.""", ("Appointment Scheduled", "Online Consultation", "High", None), "D3 / SUB Online Consultation; LQ High (offset 1)"),

    ("SYN-08", """Agent: Hello ma'am, Nisha from Aster Vale Care. You had a consultation booked yesterday at 11 AM but could not come.
Customer: Yes, sorry, there was an emergency at home. I want to rebook it.
Agent: No problem ma'am. The doctor is available this Thursday at 11 AM. Shall I reschedule it for then?
Customer: Yes, Thursday 11 AM is fine.
Agent: Done ma'am, your appointment is rescheduled to Thursday at 11 AM.
Customer: Okay, thank you.""", ("Appointment Scheduled", "Rescheduled", "High", None), "D3 / SUB Rescheduled (precedence over modality); LQ High (offset 3)"),

    ("SYN-09", """Agent: Hello sir, Tushar from Aster Vale Pain Clinic. You enquired about knee pain?
Customer: Yes, yes, I want to get my knee checked. But right now I am in the middle of something.
Agent: No problem sir. When can I call you?
Customer: Call me in 2 hours, I will be free then and we can discuss properly.
Agent: Sure sir, I will call you in 2 hours.
Customer: Okay, thank you.""", ("Follow-up Required", "Travel Constraint", "Medium", "2026-11-09T16:00:00+05:30"),
     "F10 'in N hours' from call time. Sub-label first written as Specific Callback, corrected to Travel "
     "Constraint: the labelled data treats 'busy right now, call me later' as a timing constraint "
     "(AVT-044: busy at the shop, 'call me this evening' -> gold Travel Constraint)"),

    ("SYN-10", """Agent: Hello ma'am, I am Ananya from Aster Vale Pain Clinic. You enquired about neck pain?
Customer: Yes, I have had neck pain for three months.
Agent: Ma'am, the doctor is available next week on Tuesday, the 17th, at 11 AM. Would that work?
Customer: Yes, Tuesday the 17th at 11 AM is fine for me.
Agent: Great, your clinic consultation is confirmed for Tuesday the 17th at 11 AM.
Customer: Okay, thank you.""", ("Appointment Scheduled", "Clinic Visit", "Medium", None), "D3; LQ: confirmed consultation on date offset 8 -> Medium (EX6)"),

    ("SYN-11", """Agent: Hello sir, I am calling from Aster Vale Pain Clinic about your back pain enquiry.
Customer: Yes, but I live permanently in Dubai. I am not coming to India at all.
Agent: Sir, we also have an online consultation option with the doctor.
Customer: No, I don't want any online consultation. I only wanted a clinic near me in Dubai.
Agent: Sir, we don't have any centre outside India.
Customer: Then it's of no use to me. Okay.""", ("Invalid Lead", "Outside Service Area", "Not Qualified", None), "D1 / SUB Outside Service Area (travel and remote care refused)"),

    ("SYN-12", """Agent: Hello ma'am, Riya from Aster Vale Care. You filled an enquiry form?
Customer: Yes, I wanted to know about your weight loss diet plans.
Agent: Ma'am, we are a pain clinic. Do you have any pain, in your knees, back or anywhere?
Customer: No, no pain at all. I am completely fine. I only wanted a diet plan for weight loss.
Agent: Okay ma'am, we don't provide diet plans. Sorry.
Customer: Okay, no problem.""", ("Invalid Lead", "No Pain Need", "Not Qualified", None), "D1 / SUB No Pain Need"),
]


def main():
    with open(HERE / "cases.jsonl", "w", encoding="utf-8") as fi, open(HERE / "expected.jsonl", "w", encoding="utf-8") as fe:
        for cid, tr, (d, s, lq, nxt), rule in CASES:
            fi.write(json.dumps({"transcript_id": cid, "call_started_at": CALL, "timezone": "Asia/Kolkata",
                                 "transcript": tr}, ensure_ascii=False) + "\n")
            fe.write(json.dumps({"transcript_id": cid, "disposition": d, "sub_disposition": s, "lead_quality": lq,
                                 "next_call_at": nxt, "rule": rule}, ensure_ascii=False) + "\n")
    print(f"{len(CASES)} cases written")


if __name__ == "__main__":
    main()
