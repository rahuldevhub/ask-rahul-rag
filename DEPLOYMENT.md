# Deployment

Target: Streamlit Community Cloud, Python 3.12, `app.py`, branch `master`.
Live app: https://ask-rahul-rag.streamlit.app/
Public repository: https://github.com/rahuldevhub/ask-rahul-rag
Deployed October 27, 2026.
The normal public URL was verified rendering the homepage in Phase 7 after the
earlier post-reboot loading state cleared. No application patch was required.

## Public contents

The release contains application source, offline tests, evaluation questions,
documentation, and `public_data/resume.rag.zip`. That bundle contains readable
resume passages and their FAISS vectors. Email and professional resume details
are public; phone numbers are withheld. Embeddings do not make source text private.
The original PDF, API key, request logs, and private evaluation responses are excluded.

`data/index/resume.rag.zip` takes precedence in local development. Cloud checkouts
use the reviewed public bundle. Rebuilds write to the local directory so a new
resume is reviewed before replacing the public bundle.

## Streamlit setup

1. Create a public GitHub repository with the reviewed release files.
2. In Community Cloud choose the repository, `master`, and `app.py`.
3. In Advanced settings select Python 3.12 and add these private secrets:

   ```toml
   GROQ_API_KEY = "your-key-here"
   GROQ_MODEL = "openai/gpt-oss-20b"
   ```

4. Deploy. Root-level Streamlit secrets become environment variables, which the
   application reads. Never commit the real secrets file or paste the key into logs.
5. Confirm the homepage loads, submit an approved question, and open its sources.
   Verify an unsupported question abstains and that the phone is never shown.

The encoder downloads its pinned Hugging Face model on the first question after
a fresh start. Cold starts can take longer. Linux uses the official PyTorch CPU
wheel index to avoid installing CUDA packages.

## Cost and availability

Use Community Cloud and a Groq account on its free plan. The app does not purchase
credits or enable a paid tier. Provider limits may reject requests, and free
hosting may sleep when inactive. A free demo does not promise continuous uptime.
Check the provider dashboards before enabling billing or changing plans.

To rotate a key, update Streamlit Secrets and reboot the app. Revoke the old key
in Groq. To update the resume, ingest and review it locally, rebuild the index,
scan the public passages, and replace only the reviewed public bundle.

## Sources

- [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
- [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Official PyTorch CPU wheels](https://download.pytorch.org/whl/cpu/torch/)
