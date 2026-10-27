# Ask Rahul

A resume chatbot built with Python, Streamlit, MiniLM embeddings, FAISS, and a
Groq-hosted LLM. Visitors can explore Rahul Rajendran's projects, experience,
skills, and education, with supporting resume passages attached to answers.

**Deployment status:** prepared for Streamlit Community Cloud; live URL pending.

## How it works

```text
Resume PDF → phone redaction → section-aware chunks → MiniLM → FAISS bundle
Question → MiniLM → top-3 passages → Groq → evidence validation → cited answer
```

The 384-dimensional encoder runs on CPU. Text generation uses the hosted
`openai/gpt-oss-20b` model through Groq. Explicit project/role list questions use
section routing. Follow-ups use a single previously cited topic; ambiguous
references prompt clarification. Unsupported questions should abstain.

The validator checks output structure, source IDs, verbatim quotations, and phone
privacy. It cannot prove semantic entailment. When a summary fails validation,
the interface can show original retrieved passages instead. Provider errors are
reported without automatic retries.

## Run locally

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` privately:

```toml
GROQ_API_KEY = "your-key-here"
GROQ_MODEL = "openai/gpt-oss-20b"
```

```bash
streamlit run app.py
python -m unittest discover -s tests
python -m resume_rag.smoke
```

The first question downloads the pinned MiniLM encoder. The reviewed index is
included in `public_data/resume.rag.zip`; the original PDF is excluded. A local
`data/index/resume.rag.zip` overrides the public bundle. See
[deployment instructions](DEPLOYMENT.md) for Community Cloud and secret handling.

To use a different resume, run ingestion and indexing locally, review the text,
then replace the public bundle deliberately:

```bash
python -m resume_rag.ingest /path/to/resume.pdf
python -m resume_rag.retrieval build
```

## Evaluation

The development set contains 25 questions, including 20 answerable retrieval
cases. On the same sanitized index, section routing improved complete source
coverage at top-3 from 18/20 to 20/20 and MRR@3 from 0.7583 to 0.825. These are
small development-set results, not an independent benchmark or answer accuracy.

Live evaluation made 14 authorized attempts: 12 responses and two connection
failures. Latest-per-case status checks matched expectations for 10/12 cases,
mixing original tests and targeted rechecks. That is not a clean full rerun.
The tightened project-comparison prompt and one salary case remain unverified
in the latest live run because of connection failures. Citation validation does
not establish that every generated claim is correct.

Offline tests use fixtures and mocked provider calls. Retrieval checks with the
real encoder use the included bundle. Live evaluation commands send resume data
to Groq and should only be run after reviewing their request plans.

## Privacy and limits

The public bundle contains readable resume text, professional details, and email.
Phone numbers are withheld. The original PDF, API key, and private test responses
are excluded. Questions and retrieved passages are sent to Groq. Chat history
lives in each Streamlit session's server memory and is not stored in a database.

Free hosting may sleep and API quotas can reject requests. This is a portfolio
demo with a small resume corpus. It does not promise continuous availability,
unlimited use, or perfect factual accuracy.
