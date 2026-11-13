# Task 3: Customer gender from call transcripts

Works out whether the customer on each call is male or female, using simple rules and no AI model.

**Result:** matches my hand-checked labels on **49 of 49** calls with a single customer. On the one call where a husband and wife share the phone, it correctly answers "unknown". A standard machine-learning text classifier was tested for comparison and scored 79.6%.

## How to run

```bash
# Needs data/calls.jsonl from task 1 (see the main README). No packages to install.
python gender.py --input ../task1_labeler/data/calls.jsonl --output gender_predictions.jsonl
```

For each call it outputs: `male`, `female` or `unknown`; a confidence from 0 to 1; and the quotes it based the decision on. The same input always gives the same output.

Optional: `python evaluate.py --input ../task1_labeler/data/calls.jsonl --reference reference_labels.jsonl` reproduces the evaluation below (needs scikit-learn).

## How it works

In these Indian call-centre calls, the clearest clue is **how the agent addresses the customer**: "sir" or "ma'am", "Mr." or "Mrs.", "Vikram ji". Customers also give it away themselves: "my wife", "my husband", "my name is Sameer".

The hard part is knowing **who said it**. If the customer says "ma'am", they're talking to a female agent, which tells us nothing about the customer. Some transcripts also have the Agent and Customer labels swapped.

1. **Fix who is speaking.** Lines containing things only the clinic says ("speaking from Aster Vale", "our doctors", "the consultation fee is") are treated as the agent, whatever their label says.
2. **Score the clues, stronger clues counting more:**

   | Clue | Said by | Points |
   |---|---|---|
   | "sir" vs "ma'am" | agent | 1 each time |
   | "Mr. Sharma" vs "Mrs. Sharma" | agent | 3 |
   | "my wife" / "I am Renu's husband" vs "my husband" | customer | 4 |
   | Hindi verb endings: "kar raha hoon" (male) vs "kar rahi hoon" (female) | customer | 3 |
   | A first name from a list of 222 common Indian names | either | 2 |

3. **Decide.** The side with more points wins. If both sides are close, the answer is "unknown" rather than a guess.

## How well it works

The workbook has no gender labels, so I labelled all 50 calls by hand (`reference_labels.jsonl`, with the evidence for each).

| Method | Accuracy (49 single-customer calls) |
|---|---|
| Always guess "male" | 71.4% |
| Machine-learning classifier (TF-IDF + logistic regression) | 79.6% (missed 10 of 14 women) |
| The same classifier with "sir", "ma'am" etc. removed | 69.4% |
| **These rules** | **100%** |

**What this shows:** the only real gender clue in these transcripts is how people are addressed. With those words removed, the classifier does worse than always guessing "male". It also can't tell who said "ma'am" from only 49 examples, which the rules handle directly. So the simple approach is also the more accurate one here.

**Limitation:** my hand labels use the same kinds of clues, so 100% means "agrees with a careful human reader", not "always right". Five calls have only one or two clues (AVT-015, 019, 039, 042, 048); they get low confidence scores.

**Bugs fixed along the way:**
- A customer asking "Are you speaking from Aster Vale, ma'am?" was mistaken for the agent.
- "MD/MS doctors" was read as the title "Ms.".
- "The madam will tell you" (about the doctor) was counted as addressing the customer.
- In "I am Renu's husband speaking", "Renu" was taken as the speaker's own name.

## Other approaches considered

| Approach | Why not used here |
|---|---|
| Machine-learning text classifiers (TF-IDF, Naive Bayes, SVM, word embeddings) | Need labelled data, which doesn't exist. Tested anyway: less accurate (above). |
| Learning the clue weights from data | Worth it with many labelled calls; with 50, the hand-set weights already match. |
| A bigger name list (e.g. from census data) | A cheap improvement for production; not needed for these calls. |
| Voice pitch from the call audio | Probably the best clue in a live system, but this task gives only transcripts. |
| Using the name or title stored in the CRM | The most reliable source when available; the inference should defer to it. |

## In production

- **Fast and cheap:** about 225 calls per second on one CPU core, no model or GPU, no training data.
- **Safe default:** "unknown" plus a confidence score lets a voice agent fall back to a neutral greeting.
- **Easy to extend:** supporting other Indian languages means adding their forms of address (e.g. "amma" / "anna").
- **Use responsibly:** inferred gender should only decide how the customer is addressed, never anything clinical or financial. What the customer says about themselves always wins.

## Files

| File | What it is |
|---|---|
| `gender.py` | The classifier |
| `evaluate.py` | Accuracy against the hand labels, the ML comparison and the speed test |
| `reference_labels.jsonl` | My hand labels for all 50 calls, with evidence |
| `gender_predictions.jsonl` | Output for the 50 calls |
| `test_gender.py` | 9 tests for the tricky cases: `python -m unittest test_gender` |
