"""Streamlit chat interface. Session history helps navigation, not factual claims."""

import hashlib
import html

import streamlit as st

from .answer import AnswerError, GroqClient, ResumeAssistant, load_settings
from .conversation import MAX_MESSAGES, submit_question
from .retrieval import DEFAULT_INDEX, RetrievalError

SUGGESTIONS = [
    ("Finance & engineering", "What features did Rahul build in LedgerFlow?"),
    ("Publishing & products", "What did Rahul build for authors and publishers?"),
    ("Skills & tools", "What are Rahul's core technical skills?"),
    ("Education", "Where did Rahul study and what degree did he earn?"),
]


@st.cache_resource(show_spinner=False)
def get_assistant(index_signature: str, settings_signature: str):
    # Keys/model/index changes invalidate the shared resource; no history is shared.
    return ResumeAssistant(GroqClient(load_settings()))


def ready_assistant():
    settings = load_settings()
    signature = hashlib.sha256((settings.api_key + settings.model).encode()).hexdigest()
    fingerprint = hashlib.sha256(DEFAULT_INDEX.read_bytes()).hexdigest()
    return get_assistant(fingerprint, signature)


def plain_copy(text: str):
    # Escaping prevents generated Markdown links or images from loading remote assets.
    st.html("<div class='chat-copy'>" + html.escape(text).replace("\n", "<br>") + "</div>")


def show_answer(message: dict):
    if message.get("status") == "error":
        st.error(message["answer"])
    else:
        plain_copy(message["answer"])
    if message.get("used_context"):
        st.caption("Following up on: " + message["used_context"])
    sources = message.get("sources", [])
    if sources:
        with st.expander(f"View sources · {len(sources)}"):
            for source in sources:
                st.markdown(f"**[{source['citation']}] {source['label']}**")
                st.caption(f"{source['source']} · Page {source['page']}")
                for quote in source["quotes"]:
                    st.text(quote)
                if not source["quotes"]:
                    st.text(source["text"])
                else:
                    with st.expander("Full retrieved passage"):
                        st.text(source["text"])


def main():
    st.set_page_config(page_title="Ask Rahul · Resume Intelligence", page_icon="💬", layout="centered")
    st.html("""<style>
    :root { --ink: #193c38; }
    .stApp { background: #f7f8f4; }
    .stMainBlockContainer { max-width: 900px; padding-top: 3.5rem; padding-bottom: 5rem; }
    h1 { color: var(--ink); letter-spacing: -.045em; font-size: 3.5rem !important; }
    h3 { color: var(--ink); }
    .eyebrow { color: #537267; font-size: .74rem; letter-spacing: .18em; font-weight: 700; margin-bottom: .8rem; }
    .hero-sub { color: #587168; font-size: 1.12rem; max-width: 630px; line-height: 1.6; }
    .profile-line { color: #687d74; font-size: .85rem; margin: 1.3rem 0 2.4rem; }
    .chat-copy { font-family: inherit; color: #263d36; font-size: 1rem; line-height: 1.75; overflow-wrap: anywhere; }
    [data-testid='stSidebar'] { background: #eaf0e9; }
    [data-testid='stChatMessage'] { border: 1px solid #e1e8df; border-radius: 16px; padding: 1.2rem; background: #fff; }
    [data-testid='stChatInput'] { border-radius: 14px; }
    [data-testid='stBaseButton-secondary'] { border-color: #d8e2d5; border-radius: 10px; }
    @media(max-width: 600px) { .stMainBlockContainer { padding-top: 1.7rem; } h1 { font-size: 2.6rem !important; } }
    </style>""")
    history = st.session_state.setdefault("messages", [])
    with st.sidebar:
        st.markdown("### ◈ Resume Intelligence")
        st.caption("A conversation about the work behind the resume.")
        st.divider()
        st.markdown("**Rahul Rajendran**")
        st.caption("Product Engineer · Bengaluru")
        st.caption("Phone number withheld · Email available in the resume")
        st.write("Explore projects, technical skills, design experience, and education.")
        st.button("Start a new chat", key="clear_chat", on_click=lambda: st.session_state.update(messages=[]), width="stretch")
        st.divider()
        st.caption("Answers use resume evidence. Open the sources to read the supporting text.")
        st.caption("Your question, relevant resume passages, and any follow-up topic are sent to Groq. Chat history is kept in memory for your session and isn't saved to a database.")
    st.html("<div class='eyebrow'>MEET THE PERSON BEHIND THE RESUME</div>")
    st.title("Ask Rahul.")
    st.html("<div class='hero-sub'>A closer look at the projects, skills, and experience behind the work. Ask a question. Explore the evidence.</div><div class='profile-line'>PRODUCT ENGINEERING &nbsp; / &nbsp; FULL-STACK DEVELOPMENT &nbsp; / &nbsp; UI/UX</div>")
    submitted = None
    if not history:
        st.markdown("#### Start with a question")
        columns = st.columns(2)
        for i, (label, question) in enumerate(SUGGESTIONS):
            with columns[i % 2]:
                if st.button(label, key=f"suggestion_{i}", width="stretch", help=question):
                    submitted = question
        st.caption("Or ask something specific below. If the resume doesn't cover it, the chat will say so.")
    for message in history:
        with st.chat_message(message["role"], avatar=":material/auto_awesome:" if message["role"] == "assistant" else None):
            if message["role"] == "user":
                plain_copy(message["answer"])
            else:
                show_answer(message)
    question = st.chat_input("Ask about Rahul's projects, skills, or experience…", max_chars=700, key="question")
    prompt = question or submitted
    if prompt:
        previous = list(history)
        history.append({"role": "user", "answer": prompt})
        with st.chat_message("user"):
            plain_copy(prompt)
        with st.chat_message("assistant", avatar=":material/auto_awesome:"):
            with st.spinner("Finding the relevant resume passages…"):
                try:
                    result = submit_question(prompt, previous, ready_assistant())
                except (AnswerError, RetrievalError) as exc:
                    result = {"status": "error", "answer": str(exc), "sources": []}
                except Exception:
                    result = {"status": "error", "answer": "The chat is temporarily unavailable. Please try again later.", "sources": []}
            show_answer(result)
        history.append({"role": "assistant", **result})
        st.session_state.messages = history[-MAX_MESSAGES:]
        st.rerun()
