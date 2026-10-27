"""Section-aware chunking, MiniLM embeddings, and persistent FAISS retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile

# Small resume searches do not need native worker pools. A single thread also
# avoids OpenMP contention between PyTorch and FAISS on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np

from .ingest import PROJECT_ROOT

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
LOCAL_INDEX = PROJECT_ROOT / "data" / "index" / "resume.rag.zip"
PUBLIC_INDEX = PROJECT_ROOT / "public_data" / "resume.rag.zip"
# Development rebuilds take precedence; cloud checkouts use the reviewed bundle.
DEFAULT_INDEX = LOCAL_INDEX if LOCAL_INDEX.exists() else PUBLIC_INDEX
HEADINGS = {
    "profile", "summary", "professional summary", "experience", "work experience",
    "professional experience", "projects", "selected projects", "skills",
    "technical skills", "education", "certifications", "achievements", "languages",
}


class RetrievalError(ValueError):
    """A document or index cannot be used for retrieval."""


class MiniLM:
    """A small encoder, not a text-generating LLM. CPU works on cloud hosting."""

    def __init__(self, revision: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(1)
        faiss.omp_set_num_threads(1)
        self.revision = revision or MODEL_REVISION
        self.model = SentenceTransformer(MODEL, revision=self.revision, device="cpu")
        self.tokenizer = self.model.tokenizer
        self.max_tokens = self.model.max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        # Fail rather than silently truncate a long question or chunk.
        for text in texts:
            if len(self.tokenizer.encode(text)) > self.max_tokens:
                raise RetrievalError(f"Text exceeds the encoder's {self.max_tokens}-token limit. Shorten it.")
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        ).astype("float32")


def validate_document(document: dict) -> None:
    try:
        valid = (
            document["schema_version"] == 1
            and isinstance(document["source"], str) and bool(document["source"])
            and re.fullmatch(r"[a-f0-9]{64}", document["source_sha256"])
            and isinstance(document["pages"], list) and bool(document["pages"])
            and all(isinstance(p["text"], str) and type(p["page"]) is int and p["page"] > 0
                    for p in document["pages"])
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise RetrievalError("Invalid extraction JSON. Run Phase 1 ingestion again.")


def sections(document: dict):
    """Yield page offsets and labels. Unknown headings remain ordinary text."""
    section = "Overview"
    for page in document["pages"]:
        text = page["text"]
        lines = list(re.finditer(r"[^\n]+", text))
        start = 0
        label = section
        for i, line in enumerate(lines):
            value = line.group().strip()
            heading = value.rstrip(":").casefold()
            is_heading = heading in HEADINGS
            is_project = section.casefold() in {"projects", "selected projects"} and " | " in value and not value.startswith("-")
            next_line = lines[i + 1].group().strip() if i + 1 < len(lines) else ""
            is_job = "experience" in section.casefold() and bool(re.match(r"^[A-Z][a-z]{2} \d{4}\s*[-–]", next_line))
            if not (is_heading or is_project or is_job):
                continue
            if text[start:line.start()].strip():
                yield page["page"], section, label, start, line.start(), text
            if is_heading:
                section = value.rstrip(":")
                label = section
                start = line.end()
            else:
                label = f"{section}: {value}"
                start = line.start()
        if text[start:].strip():
            yield page["page"], section, label, start, len(text), text


def make_chunks(document: dict, encoder, max_tokens: int = 192, overlap: int = 32) -> list[dict]:
    validate_document(document)
    if not 0 <= overlap < max_tokens:
        raise RetrievalError("Require 0 <= overlap < max_tokens.")
    if max_tokens > encoder.max_tokens:
        raise RetrievalError("Chunk token limit exceeds the encoder limit.")
    chunks = []
    for page, section, label, begin, end, full_text in sections(document):
        prefix = label + "\n"
        budget = max_tokens - len(encoder.tokenizer.encode(prefix)) - 4
        if budget <= overlap:
            raise RetrievalError("Chunk budget is too small for heading and overlap.")
        body = full_text[begin:end]
        offsets = encoder.tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        position = 0
        while position < len(offsets):
            stop = min(position + budget, len(offsets))
            a = begin + offsets[position][0]
            b = begin + offsets[stop - 1][1]
            content = full_text[a:b]
            embedding_text = prefix + content
            token_count = len(encoder.tokenizer.encode(embedding_text))
            if token_count > max_tokens:
                raise RetrievalError("Chunk exceeds token budget; reduce max_tokens.")
            identity = f"{document['source_sha256']}:{page}:{a}:{b}:{label}:{content}"
            chunks.append({
                "id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                "source": document["source"], "page": page, "section": section,
                "label": label, "start_char": a, "end_char": b,
                "text": content, "embedding_text": embedding_text, "token_count": token_count,
            })
            if stop == len(offsets):
                break
            position = stop - overlap
    if not chunks:
        raise RetrievalError("No non-empty chunks. Check the extracted text.")
    return chunks


def build_index(document: dict, encoder, destination: Path, max_tokens=192, overlap=32, overwrite=False) -> dict:
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise RetrievalError("Index already exists. Use --overwrite to rebuild it.")
    chunks = make_chunks(document, encoder, max_tokens, overlap)
    vectors = np.ascontiguousarray(encoder.encode([c["embedding_text"] for c in chunks]), dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not np.isfinite(vectors).all():
        raise RetrievalError("Encoder returned invalid vectors.")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise RetrievalError("Encoder returned a zero vector.")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    index_bytes = faiss.serialize_index(index).tobytes()
    manifest = {
        "schema_version": 1, "model": MODEL, "model_revision": encoder.revision,
        "dimension": index.d, "metric": "cosine", "source_sha256": document["source_sha256"],
        "max_tokens": max_tokens, "overlap": overlap, "chunks": chunks,
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A single atomic bundle avoids mixing old metadata with a new index.
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            archive.writestr("index.faiss", index_bytes)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return manifest


def load_index(path: Path):
    try:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            data = archive.read("index.faiss")
        if manifest["schema_version"] != 1 or manifest["model"] != MODEL or manifest["metric"] != "cosine":
            raise RetrievalError("Unsupported index format or embedding model. Rebuild it.")
        if hashlib.sha256(data).hexdigest() != manifest["index_sha256"]:
            raise RetrievalError("Index checksum mismatch. Rebuild it.")
        index = faiss.deserialize_index(np.frombuffer(data, dtype="uint8"))
        if index.ntotal != len(manifest["chunks"]) or index.d != manifest["dimension"]:
            raise RetrievalError("Index and metadata do not match. Rebuild it.")
        return index, manifest
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        raise RetrievalError(f"Cannot load index: {exc}") from exc


def requested_sections(question: str) -> set[str]:
    """Explicit list requests need section coverage, not unrelated close matches."""
    if re.search(r"\b(?:all|list|name)\b.{0,70}\bprojects\b", question, re.IGNORECASE):
        return {"projects", "selected projects"}
    if re.search(r"\b(?:all|list|name|what are)\b.{0,70}\b(?:roles|work history|experience section)\b", question, re.IGNORECASE):
        return {"experience", "work experience", "professional experience"}
    return set()


def search(index, manifest: dict, encoder, question: str, top_k: int = 3, use_section_routing: bool = True) -> list[dict]:
    question = question.strip()
    if not question or len(question) > 2000:
        raise RetrievalError("Enter a non-empty question of at most 2,000 characters.")
    if not 1 <= top_k <= 20:
        raise RetrievalError("top_k must be between 1 and 20.")
    if encoder.revision != manifest["model_revision"]:
        raise RetrievalError("Embedding revision differs from the index. Reload the correct encoder.")
    vector = np.ascontiguousarray(encoder.encode([question]), dtype="float32")
    if vector.shape != (1, index.d) or not np.isfinite(vector).all() or not np.linalg.norm(vector):
        raise RetrievalError("Invalid query embedding.")
    faiss.normalize_L2(vector)
    scope = requested_sections(question) if use_section_routing else set()
    # Exact search over this small resume permits filtering without another service.
    scores, positions = index.search(vector, index.ntotal if scope else min(top_k, index.ntotal))
    hits = [{**manifest["chunks"][int(position)], "score": float(score)}
            for score, position in zip(scores[0], positions[0]) if position >= 0]
    scoped = [hit for hit in hits if hit["section"].casefold() in scope] if scope else hits
    return (scoped or hits)[:top_k]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--document", type=Path, default=PROJECT_ROOT / "data/processed/resume.json")
    build.add_argument("--index", type=Path, default=LOCAL_INDEX)
    build.add_argument("--max-tokens", type=int, default=192)
    build.add_argument("--overlap", type=int, default=32)
    build.add_argument("--overwrite", action="store_true")
    query = commands.add_parser("search")
    query.add_argument("question")
    query.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    query.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            document = json.loads(args.document.read_text(encoding="utf-8"))
            validate_document(document)
            manifest = build_index(document, MiniLM(), args.index, args.max_tokens, args.overlap, args.overwrite)
            print(f"Built {len(manifest['chunks'])} chunks; {manifest['dimension']} dimensions.")
            print(f"Index: {args.index}")
            for chunk in manifest["chunks"]:
                print(f"  {chunk['id']} | page {chunk['page']} | {chunk['label']} | {chunk['token_count']} tokens")
        else:
            index, manifest = load_index(args.index)
            results = search(index, manifest, MiniLM(manifest["model_revision"]), args.question, args.top_k)
            print("Retrieved evidence only. Similarity is not answer confidence.\n")
            for rank, chunk in enumerate(results, 1):
                print(f"{rank}. {chunk['score']:.3f} | {chunk['source']} | page {chunk['page']} | {chunk['label']}")
                print(chunk["text"] + "\n")
        return 0
    except (RetrievalError, OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Retrieval failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
