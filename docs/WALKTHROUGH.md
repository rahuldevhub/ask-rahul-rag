# Follow one question through the code

Use this guide with the repository open. Start with the first two sections, then
trace the functions in order. You can understand the pipeline without sending a
new paid or quota-consuming request.

## 1. Two paths: prepare once, retrieve per question

Preparation turns the PDF into reusable search data. A visitor's question does
not re-read and re-chunk the original PDF. The deployed app loads a reviewed
bundle and reuses its index and encoder.

The embedding model and generative model have different jobs. MiniLM maps text
to vectors for comparison. Groq hosts the model that writes the answer. Running
MiniLM on CPU does not mean this project runs a local generative LLM.

## 2. What actually lives in the index?

Open `retrieval.py`: `build_index` and `load_index`.

FAISS stores vectors. The manifest stores the corresponding passage text and
metadata. A vector result's position selects the correct manifest chunk; its
stable ID is later used in citations. The model does not decode a vector back
into the resume sentence.

The bundle is one ZIP containing `index.faiss` and `manifest.json`. Checksums,
dimensions, and counts catch corruption or mismatches. They are not signatures
that establish who created a bundle.

**Check your understanding:** If you delete the passage text but keep the
vectors, can the LLM still quote the original resume? No: the evidence text must
be stored or fetched separately.

## 3. Prepare the PDF

Open `ingest.py`: `extract_resume`, then `save_extraction`.

Extraction keeps page provenance and redacts common phone formats before saving
the working text. Read the text beside the PDF: column order and links can be
lost during extraction. This project does not perform OCR on scanned pages.

Open `retrieval.py`: `sections`, then `make_chunks`.

Recognized headings, jobs, and project boundaries keep related facts together.
Long sections use token windows: at most 192 tokens including heading context,
with 32 body tokens of overlap. Token counts come from the encoder's tokenizer,
not a fixed character count. A pinned encoder revision makes rebuilds comparable.

## 4. Trace a question

Example: **“What features did Rahul build in LedgerFlow?”**

| Step | Function | What to inspect |
| --- | --- | --- |
| Submit | `ui.main` | History is session-specific; the key is server-side |
| Reuse resources | `ui.ready_assistant` / `get_assistant` | Index and settings fingerprints invalidate cached resources |
| Resolve topic | `conversation.submit_question` | Explicit project names override previous topics |
| Retrieve | `ResumeAssistant.ask` | Question plus any resolved topic becomes the retrieval query |
| Embed/search | `MiniLM.encode` / `search` | Top three nearest passages and their metadata |
| Build evidence | `answer.make_messages` | JSON evidence, stable IDs, phone filtering, system rules |
| Generate | `GroqClient.complete` | Hosted HTTPS request, structured output, timeout, no automatic retry |
| Validate | `validate_answer` | IDs must exist; quotes must appear verbatim after whitespace normalization |
| Fall back | `evidence_fallback` | Original passages replace a summary that fails validation |
| Render | `ui.show_answer` | App-created citation markers, source label, page, quotation |

To observe retrieval without generation:

```bash
python -m resume_rag.retrieval search "What features did Rahul build in LedgerFlow?"
```

Normalized vectors let inner product behave like cosine similarity. A higher
score means a closer embedding match; it is not the probability an answer is
correct. FAISS returns nearest neighbors even for an unrelated question.

## 5. Why validation helps, and where it stops

The LLM returns a status and claims. Each claim must include a source ID and
quotation. The application assigns display markers such as `[S1]`; the LLM
cannot invent a valid citation to a passage it was not given.

An exact quotation can still be attached to a misleading claim. For example,
“Uses Python” does not support “Has ten years of Python experience.” Verbatim
checking proves the quoted text exists, not that the claim follows from it.
That relationship is called **semantic entailment**. This implementation does
not include an independent semantic entailment verifier.

An insufficient-context response produces an abstention. An invalid generated
summary can produce an evidence-only fallback. A provider error produces an
error message; it does not silently turn into a successful answer.

## 6. Follow-ups and memory

Open `conversation.py`: `resolve_context`.

“What tools did he use in that project?” uses one previously cited topic label,
not the previous generated answer as factual evidence. If the last answer cited
multiple topics, the system asks which project the visitor means. Heuristics
keep this small demo simple, but do not solve all conversational references.

There are at most 20 stored message entries in a session. Cached model resources
are shared across sessions; history is not. A lock protects embedding/search
work. It does not implement a global rate limit for hosted generation.

## 7. Read the evaluation correctly

Open `evaluate.py`: `evidence_metrics`, then `summarize`.

Complete expected evidence in top 3 asks whether every annotated expected label
was retrieved. One relevant hit can be insufficient for a question asking for
all projects. Section routing helped those list questions on this resume.

**MRR@3** averages reciprocal rank of the first relevant hit, truncated at three:
rank 1 contributes 1, rank 2 contributes 0.5, rank 3 contributes about 0.333,
and no relevant hit contributes 0. It measures ranking, not answer correctness.

The questions used to tune the system are a development set. A stronger next
evaluation would use unseen paraphrases, manually judge claim support, and
measure abstention errors and latency separately. More than three projects may
also require changing retrieval depth or adding a dedicated list route.

## 8. Deployment and a cold start

Community Cloud clones the public repository and installs dependencies. It
cannot read ignored files from your laptop. That is why `PUBLIC_INDEX` exists.
`LOCAL_INDEX` takes precedence for reviewed local development.

The API key belongs in Streamlit Secrets. Root-level secrets are made available
to the app environment. A resource fingerprint detects key/model changes, but a
restart is a clear way to reload deployment configuration after replacement.

A **cold start** is the work needed after a fresh process or sleeping host:
imports, initialization, and possibly the pinned encoder download. A cached
encoder makes later questions cheaper to initialize. Free hosting availability
and provider quotas still affect the visitor's experience.
