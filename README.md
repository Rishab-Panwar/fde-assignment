# FDE assignment: Anthropod

| Task | Folder | Result |
|---|---|---|
| 1. Label call transcripts with an AI model on Groq | [`task1_labeler/`](task1_labeler/) | 64.2 / 70 on the 50 calls (all four labels right on 78% of calls) |
| 2. Horses puzzle: 2nd and 3rd fastest | [`task2_horses/`](task2_horses/) (Excalidraw drawing; open it at [excalidraw.com](https://excalidraw.com)) | 7 races in total |
| 3. Customer gender without a transformer model | [`task3_gender/`](task3_gender/) | 49 / 49 single-customer calls correct against hand-checked labels |

Each folder has its own README.

## Setup

The assignment workbook and its call transcripts are **not included**, because they're the company's data. Put your copy of the workbook in this folder and export it once. This recreates the exact input and answer files used here (`task1_labeler/data/calls.jsonl` and `gold.jsonl`, verified byte-identical).

```bash
# Python 3.10+ (tested on 3.12.10)
pip install -r requirements.txt    # only needed for the export and task 3's comparison test

cd task1_labeler
python export_data.py "../FDE Hiring Assignment — Aster Vale Pain Clinic — Candidate.xlsx" data
```

## Run

```bash
# Task 1 (needs a Groq API key)
cd task1_labeler
export GROQ_API_KEY=gsk_...                   # PowerShell: $env:GROQ_API_KEY="gsk_..."
python run.py --input data/calls.jsonl --output out/predictions.jsonl

# Task 3
cd task3_gender
python gender.py --input ../task1_labeler/data/calls.jsonl --output gender_predictions.jsonl
```

## Tests

```bash
cd task1_labeler && python -m unittest test_combine test_timeresolve test_run    # 67 tests, no API calls
cd task3_gender  && python -m unittest test_gender                                # 9 tests
```

No API keys are included. Built with help from Claude Code (an AI coding assistant); task 1 itself uses only the Groq model.
