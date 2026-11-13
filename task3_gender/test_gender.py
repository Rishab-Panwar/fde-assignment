"""Unit tests for gender.py: the cases that caused bugs during development, plus abstention.
Run: python -m unittest test_gender -v"""
import unittest

from gender import classify


def g(transcript):
    return classify({"transcript_id": "T", "transcript": transcript})


class TestGender(unittest.TestCase):
    def test_agent_address_male(self):
        self.assertEqual(g("Agent: Hi sir, I am Mehul speaking from Aster Vale Care.\n"
                           "Customer: Yes.\nAgent: What is the problem, sir?")["gender"], "male")

    def test_agent_address_female(self):
        self.assertEqual(g("Agent: Hello ma'am, I am calling from Aster Vale Care.\n"
                           "Customer: Yes.\nAgent: Since when, ma'am?")["gender"], "female")

    def test_customer_honorific_describes_agent_not_customer(self):
        # male caller says "ma'am" to a female agent; only the agent's "sir" is about the caller
        r = g("Agent: Hello, this is Alina from Aster Vale Pain Clinic.\n"
              "Customer: Are you speaking from Aster Vale Pain Clinic, Ma'am?\n"
              "Customer: Ma'am, I am calling from Juniper Heights.\n"
              "Agent: Okay sir, our doctors are available tomorrow.")
        self.assertEqual(r["gender"], "male")

    def test_swapped_speaker_labels(self):
        # labels swapped for the whole call: the "Customer" lines are clearly the clinic
        r = g("Customer: Hello sir, I am Tara speaking from Aster Vale Care.\n"
              "Agent: Yes madam.\n"
              "Customer: Sir, our doctors are available on Friday.\n"
              "Customer: Sir, the consultation fee is 1000 rupees.")
        self.assertEqual(r["gender"], "male")

    def test_md_ms_is_not_a_title(self):
        r = g("Agent: Hi sir. Our doctors have MD/MS degrees.\nAgent: MS doctors are specialists, sir.")
        self.assertEqual(r["signals"].get("mrs", 0), 0)
        self.assertEqual(r["gender"], "male")

    def test_the_madam_is_not_an_address(self):
        r = g("Agent: Can you hear me, sir?\nAgent: Exactly, the madam will tell you about the treatment.")
        self.assertEqual(r["signals"].get("agent_maam", 0), 0)

    def test_relationship_cue(self):
        self.assertEqual(g("Agent: Hello.\nCustomer: It is for my wife, she has knee pain.")["gender"], "male")
        self.assertEqual(g("Agent: Hello.\nCustomer: Let me ask my husband first.")["gender"], "female")

    def test_two_callers_abstain(self):
        r = g("Agent: Am I speaking with Renu ji?\nCustomer: I am Renu's husband speaking.\n"
              "Agent: Okay sir.\nCustomer: Wait, let me talk to my husband.\n"
              "Agent: Yes ma'am.\nAgent: Okay ma'am.")
        self.assertEqual(r["gender"], "unknown")

    def test_no_cues_unknown(self):
        r = g("Agent: Hello.\nCustomer: Hello.\nAgent: Hello?")
        self.assertEqual((r["gender"], r["confidence"]), ("unknown", 0.0))


if __name__ == "__main__":
    unittest.main()
