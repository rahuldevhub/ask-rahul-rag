import json
import threading
import unittest
from unittest.mock import Mock, patch

from resume_rag.answer import ResumeAssistant, SYSTEM_PROMPT
from resume_rag.conversation import resolve_context, submit_question


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.history = [{"role": "assistant", "status": "answered", "sources": [
            {"label": "SELECTED PROJECTS: LedgerFlow | Accounting & Finance Platform"}
        ]}]

    def test_followup_uses_cited_topic(self):
        context, clarification = resolve_context("What tools did he use in that project?", self.history)
        self.assertIn("LedgerFlow", context)
        self.assertFalse(clarification)

    def test_explicit_new_topic_does_not_inherit_context(self):
        self.assertEqual(resolve_context("What tools did he use in Publishing Plan?", self.history, ["Publishing Plan"]), ("", ""))
        self.assertEqual(resolve_context("Where did Rahul study?", self.history), ("", ""))

    def test_ambiguous_or_missing_context_asks_for_clarification(self):
        for history in [[], [{"role": "assistant", "status": "insufficient_context"}],
                        [{"role": "assistant", "status": "answered", "sources": [{"label": "A"}, {"label": "B"}]}]]:
            self.assertTrue(resolve_context("What tools did he use in that project?", history)[1])

    def test_clarification_does_not_call_groq(self):
        assistant = Mock()
        assistant.manifest = {"chunks": []}
        result = submit_question("Tell me more about that project", [], assistant)
        self.assertEqual(result["status"], "clarification")
        assistant.ask.assert_not_called()

    def test_context_is_used_for_search_and_is_not_evidence(self):
        assistant = ResumeAssistant.__new__(ResumeAssistant)
        assistant.index, assistant.manifest, assistant.encoder = object(), {}, object()
        assistant.client, assistant.lock = Mock(), threading.Lock()
        assistant.client.complete.return_value = {"status": "insufficient_context", "claims": []}
        hit = {"id": "a", "label": "Project", "text": "Actual source facts", "source": "resume.pdf", "page": 1}
        with patch("resume_rag.answer.search", return_value=[hit]) as retrieve:
            result = assistant.ask("What tools?", "LedgerFlow")
        self.assertEqual(retrieve.call_args.args[3], "LedgerFlow\nWhat tools?")
        payload = json.loads(assistant.client.complete.call_args.args[0][1]["content"])
        self.assertEqual(payload["question"], "What tools?")
        self.assertEqual(payload["conversation_context"], "LedgerFlow")
        self.assertEqual(payload["evidence"][0]["text"], "Actual source facts")
        self.assertIn("It is not evidence", SYSTEM_PROMPT)
        self.assertEqual(result["used_context"], "LedgerFlow")


if __name__ == "__main__":
    unittest.main()
