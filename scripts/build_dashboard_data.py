"""Turn the saved evaluation results into the files the write-ups use.

Reads only files in results/ and data/, so no number is ever typed by hand:
  docs/data.json             data for the GitHub Pages dashboard (docs/index.html)
  results/results_table.md   the results table that is copied into the README

Run after scripts/evaluate.py:
    .venv/bin/python scripts/build_dashboard_data.py
"""

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

RESULTS = config.RESULTS_DIR
DOCS = config.ROOT / "docs"
SYSTEMS = ["baseline", "agent"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness"]
METRIC_LABELS = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer relevancy",
                 "context_precision": "Context precision", "context_recall": "Context recall",
                 "factual_correctness": "Factual correctness (F1)"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_ragas(system: str) -> dict:
    rows = {}
    with (RESULTS / f"ragas_{system}.csv").open() as f:
        for r in csv.DictReader(f):
            rows[r["id"]] = {m: (None if r[m] == "" or math.isnan(float(r[m])) else round(float(r[m]), 4))
                             for m in METRICS}
    return rows


def results_table(summary: dict) -> str:
    b, a = summary["systems"]["baseline"], summary["systems"]["agent"]
    lines = ["| Measure | Baseline RAG | Agentic RAG |", "|---|---|---|"]
    for m in METRICS:
        lines.append(f"| {METRIC_LABELS[m]} (mean of {b['ragas'][m]['scored']} / {a['ragas'][m]['scored']} scored) "
                     f"| {b['ragas'][m]['mean']} | {a['ragas'][m]['mean']} |")
    for key, label in [("unanswerable_correctly_refused", "Unanswerable questions correctly refused"),
                       ("answerable_wrongly_refused", "Answerable questions wrongly refused")]:
        total = "unanswerable_total" if key.startswith("unans") else "answerable_total"
        lines.append(f"| {label} | {b['refusals'][key]} of {b['refusals'][total]} "
                     f"| {a['refusals'][key]} of {a['refusals'][total]} |")
    lines.append(f"| Mean seconds per answer | {b['timing']['mean_seconds']} | {a['timing']['mean_seconds']} |")
    lines.append(f"| Median seconds per answer | {b['timing']['median_seconds']} | {a['timing']['median_seconds']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    index_stats = json.loads((RESULTS / "index_stats.json").read_text())
    with config.SOURCES_CSV.open() as f:
        sources = [{k: r[k] for k in ("doc_id", "title", "publisher", "version", "url", "pages")}
                   for r in csv.DictReader(f)]
    questions = json.loads((config.ROOT / "eval" / "test_questions.json").read_text())["questions"]

    answers = {s: {r["id"]: r for r in read_jsonl(RESULTS / f"answers_{s}.jsonl")} for s in SYSTEMS}
    ragas = {s: read_ragas(s) for s in SYSTEMS}

    rows = []
    for q in questions:
        row = {"id": q["id"], "category": q["category"], "answerable": q["answerable"],
               "question": q["question"], "reference_answer": q["reference_answer"]}
        for s in SYSTEMS:
            r = answers[s][q["id"]]
            row[s] = {"answer": r["answer"], "refused": r["refused"], "seconds": r["seconds"],
                      "citations": [c["label"] for c in r["citations"]],
                      "scores": ragas[s].get(q["id"])}
            if s == "agent":
                row[s]["searches"] = r.get("searches", 1)
                row[s]["steps"] = [{"step": st["step"], "detail": st["detail"]} for st in r["steps"]]
        rows.append(row)

    data = {"run": {k: summary[k] for k in ("run_started_utc", "run_finished_utc", "total_seconds", "stage_seconds")},
            "settings": summary["settings"], "systems": summary["systems"], "metric_labels": METRIC_LABELS,
            "index": index_stats, "sources": sources, "questions": rows}
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    (RESULTS / "results_table.md").write_text(results_table(summary))
    print(f"Wrote {DOCS / 'data.json'} ({len(rows)} questions) and {RESULTS / 'results_table.md'}")
    print(results_table(summary))


if __name__ == "__main__":
    main()
