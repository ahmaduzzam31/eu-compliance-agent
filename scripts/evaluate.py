"""Run both systems on the test questions, then score them.

Stage 1: answer every question with each system and save the raw answers straight away
         (the script resumes from these files if it is stopped).
Stage 2: refusal scores on all questions, RAGAS scores on the answerable ones,
         with the local chat model as judge. Everything is saved to the output folder.

Full run:   .venv/bin/python scripts/evaluate.py
Trial run:  .venv/bin/python scripts/evaluate.py --ids Q01 Q17 Q30 --out results/trial
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # no anonymous usage data sent by RAGAS
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

QUESTIONS_FILE = config.ROOT / "eval" / "test_questions.json"
SYSTEMS = ["baseline", "agent"]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def keep_mac_awake() -> None:
    """Start caffeinate for as long as this script runs (macOS only)."""
    if platform.system() == "Darwin":
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


def load_questions(ids: list[str] | None) -> list[dict]:
    questions = json.loads(QUESTIONS_FILE.read_text())["questions"]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


# ---------------- Stage 1: answers ----------------

def run_answers(system: str, questions: list[dict], out_dir: Path) -> list[dict]:
    if system == "baseline":
        from src.baseline import answer
    else:
        from src.agent import answer

    path = out_dir / f"answers_{system}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec

    with path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(questions, start=1):
            if q["id"] in done:
                continue
            t0 = time.perf_counter()
            error = None
            try:
                r = answer(q["question"])
            except Exception as e:  # keep going, but record the failure
                error = f"{type(e).__name__}: {e}"
                r = {"answer": "", "refused": False, "citations": [], "contexts": [],
                     "retrieved": [], "steps": [], "seconds": round(time.perf_counter() - t0, 2)}
            rec = {"id": q["id"], "category": q["category"], "answerable": q["answerable"],
                   "question": q["question"], "reference_answer": q["reference_answer"],
                   **{k: v for k, v in r.items() if k != "question"}, "error": error}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[q["id"]] = rec
            status = "ERROR" if error else ("refused" if rec["refused"] else "answered")
            print(f"  [{system}] {i}/{len(questions)} {q['id']} {status} in {rec['seconds']} s", flush=True)
    return [done[q["id"]] for q in questions]


# ---------------- Stage 2: scores ----------------

def refusal_scores(recs: list[dict]) -> dict:
    unans = [r for r in recs if not r["answerable"]]
    ans = [r for r in recs if r["answerable"]]
    return {
        "unanswerable_total": len(unans),
        "unanswerable_correctly_refused": sum(r["refused"] for r in unans),
        "refusal_accuracy": round(sum(r["refused"] for r in unans) / len(unans), 4) if unans else None,
        "answerable_total": len(ans),
        "answerable_wrongly_refused": sum(r["refused"] for r in ans),
        "false_refusal_rate": round(sum(r["refused"] for r in ans) / len(ans), 4) if ans else None,
    }


def ragas_scores(system: str, recs: list[dict], out_dir: Path) -> dict:
    from langchain_ollama import OllamaEmbeddings
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall,
                               ResponseRelevancy)

    from src.models import get_llm

    ans = [r for r in recs if r["answerable"] and not r["error"]]
    if not ans:
        return {}
    samples = [SingleTurnSample(user_input=r["question"], retrieved_contexts=r["contexts"] or [""],
                                response=r["answer"], reference=r["reference_answer"]) for r in ans]
    metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    # Judge embeddings without nomic's search labels: answer relevancy compares question with question.
    judge_embeddings = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=LangchainLLMWrapper(get_llm()),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings),
        run_config=RunConfig(timeout=900, max_retries=3, max_workers=2, seed=42),
        raise_exceptions=False,
        show_progress=True,
    )
    df = result.to_pandas()
    df = df.rename(columns={m.name: label for m, label in zip(metrics, METRIC_NAMES)})
    df.insert(0, "id", [r["id"] for r in ans])
    df.insert(1, "refused", [r["refused"] for r in ans])
    df[["id", "refused", *METRIC_NAMES]].to_csv(out_dir / f"ragas_{system}.csv", index=False)

    summary = {}
    for label in METRIC_NAMES:
        values = [float(v) for v in df[label].tolist()]
        valid = [v for v in values if not math.isnan(v)]
        summary[label] = {"mean": round(statistics.mean(valid), 4) if valid else None,
                          "scored": len(valid), "failed": len(values) - len(valid)}
    return summary


def timing(recs: list[dict]) -> dict:
    secs = [r["seconds"] for r in recs]
    return {"mean_seconds": round(statistics.mean(secs), 2), "median_seconds": round(statistics.median(secs), 2),
            "max_seconds": round(max(secs), 2), "total_seconds": round(sum(secs), 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", help="only these question ids (for a trial run)")
    parser.add_argument("--out", default=str(config.RESULTS_DIR))
    parser.add_argument("--systems", nargs="*", default=SYSTEMS, choices=SYSTEMS)
    args = parser.parse_args()

    keep_mac_awake()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(args.ids)
    started = now_utc()
    t_all = time.perf_counter()
    print(f"{len(questions)} questions, systems: {', '.join(args.systems)}, output: {out_dir}")

    stage_seconds, results = {}, {}
    for system in args.systems:
        t0 = time.perf_counter()
        recs = run_answers(system, questions, out_dir)
        stage_seconds[f"answers_{system}"] = round(time.perf_counter() - t0, 1)
        results[system] = {"questions": len(recs), "errors": sum(bool(r["error"]) for r in recs),
                           "refusals": refusal_scores(recs), "timing": timing(recs)}

    for system in args.systems:
        print(f"Scoring {system} with RAGAS...", flush=True)
        recs = [json.loads(l) for l in (out_dir / f"answers_{system}.jsonl").read_text().splitlines()]
        recs = [r for r in recs if r["id"] in {q["id"] for q in questions}]
        t0 = time.perf_counter()
        results[system]["ragas"] = ragas_scores(system, recs, out_dir)
        stage_seconds[f"ragas_{system}"] = round(time.perf_counter() - t0, 1)

    import ragas
    summary = {
        "run_started_utc": started,
        "run_finished_utc": now_utc(),
        "total_seconds": round(time.perf_counter() - t_all, 1),
        "stage_seconds": stage_seconds,
        "question_ids": [q["id"] for q in questions],
        "settings": {"chat_model": f"{config.LLM_PROVIDER}:{config.LLM_MODEL}",
                     "judge_model": f"{config.LLM_PROVIDER}:{config.LLM_MODEL}",
                     "embedding_model": config.EMBED_MODEL, "reranker": "BAAI/bge-reranker-base",
                     "temperature": config.LLM_TEMPERATURE, "top_k": config.TOP_K,
                     "chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
                     "ragas_version": ragas.__version__},
        "systems": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
