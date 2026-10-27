import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock

from resume_rag.answer import AnswerError, GroqClient, Settings, make_messages
from resume_rag.evaluate import evidence_metrics, summarize
from resume_rag.evaluate_live import run_plan
from resume_rag.retrieval import requested_sections


class EvaluationTests(unittest.TestCase):
    def test_multisource_coverage_counts_each_required_topic(self):
        hits = [{"label": "LedgerFlow"}, {"label": "LedgerFlow details"}, {"label": "Overview"}]
        metrics = evidence_metrics(["LedgerFlow", "Publishing Plan"], hits)
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertFalse(metrics["complete"])
        self.assertEqual(metrics["first_relevant_rank"], 1)

    def test_missing_information_is_not_scored_as_retrieval_failure(self):
        metrics = evidence_metrics([], [{"label": "Overview"}])
        self.assertIsNone(metrics["coverage"])
        self.assertIsNone(metrics["complete"])

    def test_aggregate_includes_failures_in_denominator(self):
        rows = [{"metrics": evidence_metrics(["A"], [{"label": "A"}])},
                {"metrics": evidence_metrics(["B"], [{"label": "C"}])},
                {"metrics": evidence_metrics([], [])}]
        result = summarize(rows)
        self.assertEqual(result["retrieval_cases"], 2)
        self.assertEqual(result["complete_evidence_at_3"], 1)
        self.assertEqual(result["mean_evidence_coverage_at_3"], 0.5)
        self.assertEqual(result["mrr_at_3"], 0.5)

    def test_explicit_lists_are_scoped_but_detail_questions_are_not(self):
        self.assertIn("selected projects", requested_sections("Name all three selected projects in Rahul's resume."))
        self.assertIn("experience", requested_sections("What are all three roles listed in Rahul's experience section?"))
        self.assertEqual(requested_sections("What did Rahul build in LedgerFlow?"), set())


class LivePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.plan_path = Path(self.temp.name) / "plan.json"
        self.output_path = Path(self.temp.name) / "result.json"
        self.client = GroqClient(Settings("fictional-key-never-used"))
        self.client.complete = Mock(return_value={"status": "insufficient_context", "claims": []})
        self.plan = [{"case_id": "one", "expected_status": "insufficient_context", "hits": [],
                      "request": self.client.request_payload(make_messages("Salary?", []))}]
        self.save_plan()

    def save_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def test_checkpoint_preserves_result_and_requires_manual_review(self):
        report = run_plan(self.plan_path, self.output_path, self.client, interval=0)
        self.assertEqual(report["status_matches"], 1)
        self.assertEqual(report["manual_review"], "pending")
        self.assertEqual(json.loads(self.output_path.read_text())["attempted_requests"], 1)

    def test_changed_payload_is_blocked_before_network(self):
        self.plan[0]["request"]["temperature"] = 1
        self.save_plan()
        with self.assertRaisesRegex(AnswerError, "differ"):
            run_plan(self.plan_path, self.output_path, self.client, interval=0)
        self.client.complete.assert_not_called()

    def test_provider_failure_stops_without_retry_or_next_request(self):
        self.plan.append({**self.plan[0], "case_id": "two"})
        self.save_plan()
        self.client.complete.side_effect = AnswerError("Groq's usage limit was reached.")
        report = run_plan(self.plan_path, self.output_path, self.client, interval=0)
        self.assertEqual(report["attempted_requests"], 1)
        self.assertEqual(self.client.complete.call_count, 1)

    def test_previous_output_is_not_silently_replayed(self):
        self.output_path.write_text("{}")
        with self.assertRaisesRegex(AnswerError, "already exists"):
            run_plan(self.plan_path, self.output_path, self.client, interval=0)
        self.client.complete.assert_not_called()

    def test_plan_with_phone_number_is_blocked_before_network(self):
        self.plan[0]["request"] = self.client.request_payload([
            {"role": "user", "content": "Please call +91 9876543210"}])
        self.save_plan()
        with self.assertRaisesRegex(AnswerError, "phone number"):
            run_plan(self.plan_path, self.output_path, self.client, interval=0)
        self.client.complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
