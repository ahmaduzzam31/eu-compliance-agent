"""Ask one question from the terminal.

    .venv/bin/python scripts/ask.py --system baseline "What is the right to erasure?"
    .venv/bin/python scripts/ask.py --system agent "What is the right to erasure?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--system", choices=["baseline", "agent"], default="baseline")
    args = parser.parse_args()

    if args.system == "baseline":
        from src.baseline import answer
    else:
        from src.agent import answer  # added in step 4

    r = answer(args.question)
    print(f"\nQ: {r['question']}\n\nA: {r['answer']}\n")
    print("Cited sources:" if r["citations"] else "Cited sources: none")
    for c in r["citations"]:
        print(f"  [{c['n']}] {c['label']}")
    print("\nSteps:")
    for s in r["steps"]:
        print(f"  - {s['step']}: {s['detail']} ({s['seconds']} s)")
    print(f"\nTotal: {r['seconds']} s")


if __name__ == "__main__":
    main()
