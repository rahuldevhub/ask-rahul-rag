"""Reproducible retrieval evaluation and reviewable live Groq test planning."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .answer import DEFAULT_MODEL, SCHEMA, make_messages
from .conversation import resolve_context
from .ingest import PROJECT_ROOT
from .retrieval import DEFAULT_INDEX, MiniLM, load_index, search

QUESTIONS = PROJECT_ROOT / "evaluation/questions.json"
OUTPUT = PROJECT_ROOT / "data/processed/evaluation"


def evidence_metrics(expected: list[str], hits: list[dict]) -> dict:
    if not expected:
        return {"coverage": None, "complete": None, "first_relevant_rank": None}
    ranks = [next((i for i, hit in enumerate(hits, 1)
                   if label.casefold() in hit["label"].casefold()), None) for label in expected]
    found = [rank for rank in ranks if rank is not None]
    return {"coverage": len(found) / len(expected), "complete": len(found) == len(expected),
            "first_relevant_rank": min(found) if found else None}


def summarize(rows: list[dict]) -> dict:
    scored = [row for row in rows if row["metrics"]["coverage"] is not None]
    return {"cases": len(rows), "retrieval_cases": len(scored),
            "complete_evidence_at_3": sum(row["metrics"]["complete"] for row in scored),
            "mean_evidence_coverage_at_3": sum(row["metrics"]["coverage"] for row in scored) / len(scored) if scored else 0,
            "first_relevant_at_1": sum(row["metrics"]["first_relevant_rank"] == 1 for row in scored),
            "mrr_at_3": sum(1 / row["metrics"]["first_relevant_rank"] if row["metrics"]["first_relevant_rank"] else 0 for row in scored) / len(scored) if scored else 0}


def run_retrieval(output_dir: Path, index_path=DEFAULT_INDEX, questions_path=QUESTIONS, semantic_only=False):
    questions = json.loads(questions_path.read_text())
    index, manifest = load_index(index_path)
    encoder = MiniLM(manifest["model_revision"])
    rows, requests = [], []
    for case in questions:
        if case["expected_status"] == "clarification":
            context, clarification = resolve_context(case["question"], [])
            hits = []
            route = "clarification" if clarification else "retrieval"
        else:
            context = case.get("context", "")
            query = context + "\n" + case["question"] if context else case["question"]
            hits = search(index, manifest, encoder, query, use_section_routing=not semantic_only)
            route = "retrieval"
        metrics = evidence_metrics(case["expected_labels"], hits)
        rows.append({**case, "route": route, "metrics": metrics, "hits": hits})
        if case.get("live"):
            requests.append({"case_id": case["id"], "expected_status": case["expected_status"],
                             "request": {
                "model": DEFAULT_MODEL, "messages": make_messages(case["question"], hits, context),
                "temperature": 0, "reasoning_effort": "low", "max_completion_tokens": 1800,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "resume_answer", "strict": True, "schema": SCHEMA}},
            }, "hits": hits})
        if metrics["complete"] is not None:
            print(f"{case['id']} {'PASS' if metrics['complete'] else 'FAIL'} | coverage {metrics['coverage']:.2f} | {case['question']}", flush=True)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "retrieval_mode": "semantic-only" if semantic_only else "section-aware",
              "index_bundle_sha256": hashlib.sha256(Path(index_path).read_bytes()).hexdigest(),
              "source_sha256": manifest["source_sha256"], "model_revision": manifest["model_revision"],
              "summary": summarize(rows), "rows": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (output_dir / "live-request-plan.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"Prepared {len(requests)} live requests for review. No Groq requests sent.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--semantic-only", action="store_true", help="Disable section routing for a baseline comparison")
    args = parser.parse_args()
    run_retrieval(args.output_dir, semantic_only=args.semantic_only)


if __name__ == "__main__":
    main()
