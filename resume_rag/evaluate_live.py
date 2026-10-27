"""Execute an explicitly approved request plan with checkpoints and no retries."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from .answer import AnswerError, GroqClient, load_settings, validate_answer
from .privacy import contains_phone_number, redact_phone_numbers


def run_plan(plan_path: Path, output_path: Path, client, interval: float = 35):
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if not isinstance(plan, list) or not 1 <= len(plan) <= 12:
        raise AnswerError("Evaluation plan must contain between 1 and 12 requests.")
    # Validate the entire plan before the first outbound request.
    ids = set()
    for case in plan:
        if contains_phone_number(json.dumps(case["request"], ensure_ascii=False)):
            raise AnswerError("A request contains a phone number. Regenerate a redacted plan before sending it.")
        if case["case_id"] in ids:
            raise AnswerError("Evaluation case IDs must be unique.")
        ids.add(case["case_id"])
        if client.request_payload(case["request"]["messages"]) != case["request"]:
            raise AnswerError("Model or request settings differ from the reviewed plan. Regenerate it before approval.")
    if output_path.exists():
        raise AnswerError("This output already exists. Review previous results before starting another live run.")
    report = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "results": [],
              "manual_review": "pending", "planned_requests": len(plan)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_start = None
    for case in plan:
        if previous_start is not None:
            time.sleep(max(0, interval - (time.monotonic() - previous_start)))
        previous_start = time.monotonic()
        question = json.loads(case["request"]["messages"][1]["content"])["question"]
        record = {"case_id": case["case_id"], "question": question,
                  "expected_status": case["expected_status"]}
        try:
            raw = client.complete(case["request"]["messages"])
            result = validate_answer(raw, case["hits"])
            record.update(result)
            if result["status"] == "unverified":
                # Keep private, redacted failure diagnostics for manual inspection.
                record["rejected_output"] = json.loads(redact_phone_numbers(json.dumps(raw, ensure_ascii=False)))
            record["status_match"] = result["status"] == case["expected_status"]
        except AnswerError as exc:
            record.update({"status": "provider_error", "error": str(exc), "status_match": False})
        report["results"].append(record)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"{record['case_id']} | {record['status']} | expected {record['expected_status']}", flush=True)
        if record["status"] == "provider_error":
            print("Stopped on provider error. No automatic retry or remaining request was sent.", flush=True)
            break
    report["attempted_requests"] = len(report["results"])
    report["status_matches"] = sum(row["status_match"] for row in report["results"])
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_plan(args.plan, args.output, GroqClient(load_settings()))
    except (AnswerError, OSError, ValueError) as exc:
        print(f"Evaluation stopped: {exc}")
        return 1
    print(f"Status matches: {report['status_matches']}/{report['attempted_requests']}; manual answer review still required.")
    return 0 if report["attempted_requests"] == report["planned_requests"] and report["status_matches"] == report["attempted_requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
