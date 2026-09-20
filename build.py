# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build/validate the VotingFacts dataset: check the questions, ground truth, and cached answers
line up, and print the dataset shape. Run this before scoring to confirm the data is consistent.

The dataset is two files under data/:
  data/questions.jsonl    one rendered voter question per row, joined to its reference fact
  data/ground-truth.csv   the verified facts (one row per fact), each carrying its own citation

and the cached system-under-test answers under results/:
  results/answers.jsonl   one answer per (question x model), already collected (web search on)

Run:  uv run build.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
QUESTIONS = HERE / "data" / "questions.jsonl"
GROUND_TRUTH = HERE / "data" / "ground-truth.csv"
ANSWERS = HERE / "results" / "answers.jsonl"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    questions = load_jsonl(QUESTIONS)
    answers = load_jsonl(ANSWERS)
    with GROUND_TRUTH.open(encoding="utf-8", newline="") as fh:
        gt = list(csv.DictReader(fh))

    qids = {q["qid"] for q in questions}
    models = sorted({a["model"] for a in answers})

    # every cached answer must point at a known question
    orphan = [a["qid"] for a in answers if a["qid"] not in qids]
    if orphan:
        raise SystemExit(f"answers reference unknown qids: {sorted(set(orphan))}")

    # every (question x model) pair should have exactly one cached answer
    have = Counter((a["qid"], a["model"]) for a in answers)
    missing = [(qid, m) for qid in qids for m in models if (qid, m) not in have]
    if missing:
        raise SystemExit(f"missing cached answers for: {missing}")

    print(f"questions     : {len(questions)}")
    print(f"ground-truth  : {len(gt)} fact rows")
    print(f"models        : {models}")
    print(f"cached answers: {len(answers)} ({len(qids)} questions x {len(models)} models)")
    print("by risk tier  :", dict(sorted(Counter(q['risk_tier'] for q in questions).items())))
    print("by election   :", dict(sorted(Counter(q['election_id'] for q in questions).items())))
    print("\ndataset OK -> run:  uv run score.py")


if __name__ == "__main__":
    main()
