import json
import unittest
from unittest.mock import patch

from resume_rag.answer import AnswerError, GroqClient, Settings, make_messages, validate_answer
from resume_rag.privacy import contains_phone_number, redact_phone_numbers


class PrivacyTests(unittest.TestCase):
    def test_phone_is_removed_and_email_and_dates_survive(self):
        text = "Contact: +91 9876543210 | candidate@example.com\nJul 2024 - Jul 2026\n37 tests"
        cleaned = redact_phone_numbers(text)
        self.assertNotIn("9876543210", cleaned)
        self.assertIn("candidate@example.com", cleaned)
        self.assertIn("Jul 2024 - Jul 2026", cleaned)
        self.assertIn("37 tests", cleaned)

    def test_outbound_prompt_redacts_question_context_and_evidence(self):
        phone = "+91 9876543210"
        hit = {"id": "a", "label": "Overview", "text": phone + " candidate@example.com"}
        messages = make_messages("Can I call " + phone, [hit], phone)
        serialized = json.dumps(messages)
        self.assertNotIn("9876543210", serialized)
        self.assertIn("candidate@example.com", serialized)
        self.assertFalse(contains_phone_number(serialized))

    def test_generated_phone_is_rejected_even_with_existing_source_quote(self):
        hit = {"id": "a", "source": "resume.pdf", "page": 1, "label": "Profile", "text": "The candidate is a product engineer."}
        raw = {"status": "answered", "claims": [{"text": "Call +91 9876543210.", "evidence": [
            {"source_id": "a", "quote": "The candidate is a product engineer."}]}]}
        self.assertEqual(validate_answer(raw, [hit])["status"], "unverified")

    def test_formatted_phone_patterns(self):
        for phone in ["9876543210", "+1 (415) 555-0123", "+44 20 7946 0958"]:
            self.assertTrue(contains_phone_number(phone))

    def test_similarity_decimals_are_not_phone_numbers(self):
        self.assertFalse(contains_phone_number('"score": 0.40458083152771'))

    def test_transport_blocks_unredacted_direct_request(self):
        client = GroqClient(Settings("fictional-key-never-used"))
        with patch("resume_rag.answer.urlopen") as transport:
            with self.assertRaisesRegex(AnswerError, "withheld"):
                client.complete([{"role": "user", "content": "Call +91 9876543210"}])
            transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
