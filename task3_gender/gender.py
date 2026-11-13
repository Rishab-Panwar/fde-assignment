"""Rule-based customer gender identification (no transformer / no ML model).

Idea: in Indian call-centre speech the agent addresses the caller with gendered
honorifics ("sir", "ma'am", "Mr.", "Mrs.") and callers reveal gender through
relationships ("my husband", "my wife"), self-introductions ("my name is ...")
and residual Hindi verb endings ("raha hoon" / "rahi hoon"). We collect those
cues, weight them, and vote.

Speaker labels in the transcripts can be wrong, so each turn is first re-assigned
to agent/customer using content cues (agent-only phrases such as "speaking from
Aster Vale", "our doctors", "consultation fee is ..."). Honorifics count only when
spoken by the agent *to* the customer; honorifics the customer uses describe the
agent and are ignored.

Usage:
    python gender.py --input ../task1_labeler/data/calls.jsonl --output gender_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

# ---------------------------------------------------------------- role repair
# Phrases only the clinic side says. Kept clinic-specific so that a customer
# asking "are you speaking from Aster Vale?" is fine, and "I am calling from
# <my locality>" is NOT mistaken for the agent.
AGENT_CUES = re.compile(
    r"((?:i am|i'm|this is)\s+\w+(?:\s+\w+)?\s+(?:speaking|calling|talking) from (?:the )?(?:aster vale|doctor)"
    r"|(?:i am|i'm) (?:speaking|calling|talking) from (?:the )?(?:aster vale|doctor'?s? (?:team|side))"
    r"|\bour doctors?\b|\bour (?:centre|center|clinic|hospital|senior doctors)\b"
    r"|consultation (?:fee|charges?) (?:is|are)|(?:the )?doctor (?:is|will be) available"
    r"|we have (?:\d|more than|non-surgical|different|specialist)|shall i book"
    r"|i(?:'ll| will) (?:send|share) (?:you )?(?:the )?(?:location|address|details|doctor)"
    r"|book (?:a|an|your) (?:slot|appointment) for you|received a call ?back request|you had enquired)",
    re.I)
CUSTOMER_CUES = re.compile(
    r"(where is your (?:clinic|centre|center)|what(?:'s| is) the (?:fee|address|charge)|how much (?:does|will) it cost"
    r"|\bmy (?:mother|father|husband|wife|son|daughter|knee|back|leg)\b|\bi have (?:pain|been)\b|it hurts"
    r"|are you (?:speaking|calling) from)",
    re.I)


def label_of(line: str) -> tuple[str, str] | None:
    label, sep, text = line.partition(":")
    if not sep:
        return None
    return ("agent" if label.strip().lower().startswith("agent") else "customer"), text


def turn_roles(transcript: str) -> list[tuple[str, str]]:
    """[(role, text)] with role in {'agent','customer'}.

    1. If the side labelled Customer carries clearly more agent cues than the side
       labelled Agent, the labels are swapped for the whole call: flip them.
    2. Per turn, a turn with agent cues only becomes agent; customer cues only -> customer.
    """
    parsed = [p for p in (label_of(l) for l in transcript.splitlines()) if p]
    cue_count = Counter(role for role, text in parsed
                        if AGENT_CUES.search(text) and not CUSTOMER_CUES.search(text))
    swapped = cue_count["customer"] > cue_count["agent"] + 1
    out = []
    for role, text in parsed:
        if swapped:
            role = "customer" if role == "agent" else "agent"
        a, c = bool(AGENT_CUES.search(text)), bool(CUSTOMER_CUES.search(text))
        if a and not c:
            role = "agent"
        elif c and not a:
            role = "customer"
        out.append((role, text))
    return out


# ------------------------------------------------------------- name lexicon
# Small list of common Indian first names. Transcript names are substituted, but
# substitution appears gender-consistent with the honorifics used.
MALE_NAMES = set("""
aarav abhay abhishek aditya ajay akash akhil alok aman amit amitabh anand anil ankit anuj arjun arun
ashish ashok atul ayush balaji bharat bhaskar chetan deepak dev dhruv dinesh gaurav girish gopal harish
hemant hitesh ishan jagdish jai jatin kabir kailash karan kartik kishore krishna kunal lalit mahesh manish
manoj mayank mehul mohan mohit mukesh nadeem naren naresh naveen neel nikhil nilesh nitin om pankaj
paresh pawan piyush pradeep prakash pranav prashant pratik praveen rahul raj rajan rajat rajesh rajiv
rakesh ram ramesh ravi rishi rohan rohit sachin sahil sameer sandeep sanjay santosh satish saurabh shankar
shivam shyam siddharth sudhir sumit sunil suresh tarun tushar uday umesh varun vedant vijay vikas vikram
vinay vinod vishal vivek yash yogesh adil arvind harsh imran irfan javed nakul omkar rahim salim zayan
""".split())
FEMALE_NAMES = set("""
aarti aditi aishwarya alina ananya anita anjali ankita anushka archana asha bhavna deepa deepika divya gauri
geeta heena isha jaya jyoti kajal kalpana kavita kavya komal lakshmi lata leena madhu mamta manisha
meena meera megha mira monika naina neelam neha nisha pallavi pooja poonam preeti priya priyanka
radha rani rashmi rekha renu riya ritu ruchi sadhana sakshi sangeeta sapna sarita saira seema shalini
shilpa shobha shreya shweta smita sneha sonal sonia sudha sujata sunita swati tanvi tara uma usha
vandana varsha vidya anu ruby fatima nazia zara ayesha farah rubina sana shabnam tina
""".split())
NOT_NAMES = {"aster", "sir", "madam", "doctor", "yes", "okay", "speaking", "calling", "from", "the", "your", "good"}

# ------------------------------------------------------------------- cue regexes
SIR = re.compile(r"\b(sir|bhaiya|brother|uncle|bhai ?sahab)\b", re.I)
# "the madam will tell you" talks ABOUT someone (e.g. the doctor); it is not an address.
MAAM = re.compile(r"(?<!the )\b(ma'?am|maam|madam|mam|didi|sister|aunty|auntie|behen ?ji)\b", re.I)
# Case-sensitive so that "MD/MS doctors" is not read as "Ms."
MR = re.compile(r"\b(?:Mr|MR)\.?\s+[A-Z][a-z]+")
MRS = re.compile(r"\b(?:Mrs|Ms|Miss|MRS)\.?\s+[A-Z][a-z]+")
CUST_FEMALE_REL = re.compile(r"\bmy (husband|hubby)\b"
                             r"|\b(?:i am|i'm|this is)\s+\w+'s\s+(wife|daughter|mother|sister|daughter-in-law)\b", re.I)
CUST_MALE_REL = re.compile(r"\bmy (wife|missus)\b"
                           r"|\b(?:i am|i'm|this is)\s+\w+'s\s+(husband|son|father|brother|son-in-law)\b", re.I)
HI_FEMALE = re.compile(r"\b(rahi|gayi|sakti|karti|jaati|aati|chahti|thi) (hoon|hun|hu)\b", re.I)
HI_MALE = re.compile(r"\b(raha|gaya|sakta|karta|jaata|aata|chahta|tha) (hoon|hun|hu)\b", re.I)
# "(?!'s)": in "I am Renu's husband" the name belongs to someone else, not the speaker
SELF_NAME = re.compile(r"\b(?:my name is|this is|i am|i'm)\s+([A-Z][a-z]+)\b(?!'s)", re.I)
AGENT_SELF_NAME = re.compile(r"\b(?:i am|i'm|this is|my name is)\s+([A-Z][a-z]+)", re.I)
ADDRESSED_NAME = re.compile(
    r"\b(?:speaking (?:with|to)|am i speaking (?:with|to)|hi|hello|is this)\s+(?:Mr\.?|Mrs\.?|Ms\.?)?\s*([A-Z][a-z]+)\b")
NAME_HONORIFIC = re.compile(r"\b([A-Z][a-z]+)\s+(sir|ma'?am|madam|ji)\b")

WEIGHTS = {"agent_sir": 1.0, "agent_maam": 1.0, "mr": 3.0, "mrs": 3.0, "rel_male": 4.0, "rel_female": 4.0,
           "hindi_male": 3.0, "hindi_female": 3.0, "name_male": 2.0, "name_female": 2.0}
MALE_KEYS = ("agent_sir", "mr", "rel_male", "hindi_male", "name_male")
FEMALE_KEYS = ("agent_maam", "mrs", "rel_female", "hindi_female", "name_female")


def name_gender(name: str) -> str | None:
    n = name.lower()
    if n in NOT_NAMES:
        return None
    if n in MALE_NAMES:
        return "male"
    if n in FEMALE_NAMES:
        return "female"
    return None


def classify(rec: dict) -> dict:
    turns = turn_roles(rec["transcript"])
    score: Counter = Counter()
    evidence: list[str] = []

    def add(key: str, text: str) -> None:
        score[key] += 1
        if len(evidence) < 6:
            evidence.append(f"{key}: {text.strip()[:90]}")

    agent_names = {m.group(1).lower() for role, text in turns if role == "agent"
                   for m in AGENT_SELF_NAME.finditer(text)}

    for role, text in turns:
        if role == "agent":  # how the clinic addresses the caller
            for _ in SIR.findall(text):
                add("agent_sir", text)
            for _ in MAAM.findall(text):
                add("agent_maam", text)
            if MR.search(text):
                add("mr", text)
            if MRS.search(text):
                add("mrs", text)
            for rx in (ADDRESSED_NAME, NAME_HONORIFIC):
                for m in rx.finditer(text):
                    if m.group(1).lower() in agent_names:
                        continue
                    g = name_gender(m.group(1))
                    if g:
                        add(f"name_{g}", text)
        else:  # what the caller reveals about themselves
            if CUST_FEMALE_REL.search(text):
                add("rel_female", text)
            if CUST_MALE_REL.search(text):
                add("rel_male", text)
            if HI_FEMALE.search(text):
                add("hindi_female", text)
            if HI_MALE.search(text):
                add("hindi_male", text)
            for m in SELF_NAME.finditer(text):
                g = name_gender(m.group(1))
                if g:
                    add(f"name_{g}", text)

    male = sum(score[k] * WEIGHTS[k] for k in MALE_KEYS)
    female = sum(score[k] * WEIGHTS[k] for k in FEMALE_KEYS)
    total = male + female
    if total == 0:
        gender, conf = "unknown", 0.0
    else:
        margin = abs(male - female) / total
        gender = "male" if male > female else "female" if female > male else "unknown"
        conf = round(margin * min(1.0, total / 6), 2)  # agreement x amount of evidence
        if margin < 0.34:  # signals genuinely conflict (e.g. two callers share the phone)
            gender = "unknown"
    return {"transcript_id": rec["transcript_id"], "gender": gender, "confidence": conf,
            "male_score": male, "female_score": female, "signals": dict(score), "evidence": evidence}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    with open(args.output, "w", encoding="utf-8") as f:
        for r in recs:
            res = classify(r)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"{res['transcript_id']}  {res['gender']:8s} conf={res['confidence']:.2f}  "
                  f"M={res['male_score']:.0f} F={res['female_score']:.0f}  {res['signals']}")


if __name__ == "__main__":
    main()
