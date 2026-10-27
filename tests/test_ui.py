from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from resume_rag.answer import AnswerError
from resume_rag.ingest import PROJECT_ROOT
from resume_rag.ui import SUGGESTIONS

RESULT = {
    "status": "answered", "answer": "The candidate built an invoicing platform. [S1]",
    "sources": [{"citation": "S1", "id": "fictional", "source": "fictional.pdf", "page": 1,
                 "label": "SELECTED PROJECTS: LedgerFlow | Finance",
                 "text": "Built an invoicing platform with React.",
                 "quotes": ["Built an invoicing platform with React."]}],
}


class ChatUITests(unittest.TestCase):
    def setUp(self):
        self.assistant = Mock()
        self.assistant.manifest = {"chunks": [{"label": "SELECTED PROJECTS: LedgerFlow | Finance"}]}
        self.assistant.ask.return_value = deepcopy(RESULT)
        self.patch = patch("resume_rag.ui.ready_assistant", return_value=self.assistant)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=10).run()

    def test_initial_page_is_ready_without_model_request(self):
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.title[0].value, "Ask Rahul.")
        self.assertEqual(len(self.app.chat_input), 1)
        self.assertEqual(len(self.app.session_state["messages"]), 0)
        self.assistant.ask.assert_not_called()

    def test_suggested_question_runs_once_and_preserves_sources(self):
        self.app.button(key="suggestion_0").click().run()
        self.assertFalse(self.app.exception)
        self.assistant.ask.assert_called_once_with(SUGGESTIONS[0][1], conversation_context="")
        self.assertEqual(len(self.app.chat_message), 2)
        self.assertTrue(any(e.label == "View sources · 1" for e in self.app.expander))
        self.assertEqual(self.app.session_state["messages"][-1]["sources"][0]["page"], 1)
        self.app.run()
        self.assertEqual(self.assistant.ask.call_count, 1)

    def test_typed_followup_passes_previous_cited_topic(self):
        self.app.chat_input[0].set_value("Tell me about LedgerFlow.").run()
        self.app.chat_input[0].set_value("What tools did he use in that project?").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.assistant.ask.call_args.kwargs["conversation_context"], RESULT["sources"][0]["label"])
        self.assertEqual(len(self.app.session_state["messages"]), 4)

    def test_new_chat_clears_history_and_followup_context(self):
        self.app.chat_input[0].set_value("Tell me about LedgerFlow.").run()
        self.app.button(key="clear_chat").click().run()
        self.assertEqual(self.app.session_state["messages"], [])
        self.app.chat_input[0].set_value("What tools did he use in that project?").run()
        self.assertEqual(self.app.session_state["messages"][-1]["status"], "clarification")
        self.assertEqual(self.assistant.ask.call_count, 1)

    def test_missing_information_has_no_source_expander(self):
        self.assistant.ask.return_value = {"status": "insufficient_context", "answer": "No salary information is available.", "sources": []}
        self.app.chat_input[0].set_value("What is Rahul's expected salary?").run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.expander)
        self.assertEqual(self.app.session_state["messages"][-1]["status"], "insufficient_context")

    def test_evidence_only_fallback_shows_original_source_text(self):
        result = deepcopy(RESULT)
        result.update(status="evidence_only", answer="Read the closest resume passages instead.")
        result["sources"][0]["quotes"] = []
        self.assistant.ask.return_value = result
        self.app.chat_input[0].set_value("Tell me about LedgerFlow.").run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any(item.value == RESULT["sources"][0]["text"] for item in self.app.text))

    def test_rate_limit_and_unexpected_errors_are_displayed_safely(self):
        for error in [AnswerError("Groq's usage limit was reached. Wait before trying again."), RuntimeError("PRIVATE_INTERNAL_DETAIL")]:
            self.assistant.ask.side_effect = error
            self.app.chat_input[0].set_value("Tell me about LedgerFlow.").run()
            self.assertFalse(self.app.exception)
            self.assertTrue(self.app.error)
            self.assertNotIn("PRIVATE_INTERNAL_DETAIL", self.app.error[-1].value)
            self.assertEqual(self.app.session_state["messages"][-1]["status"], "error")

    def test_history_is_bounded_and_sessions_are_separate(self):
        self.app.session_state["messages"] = [
            entry for _ in range(10) for entry in [
                {"role": "user", "answer": "Previous question"},
                {"role": "assistant", **deepcopy(RESULT)},
            ]
        ]
        self.app.run()
        self.app.chat_input[0].set_value("Tell me about LedgerFlow.").run()
        self.assertEqual(len(self.app.session_state["messages"]), 20)
        other = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run()
        self.assertEqual(other.session_state["messages"], [])


if __name__ == "__main__":
    unittest.main()
