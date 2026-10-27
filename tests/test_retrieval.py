"""Offline behavior checks. Real MiniLM retrieval is tested separately."""

import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

import numpy as np

from resume_rag.retrieval import (
    RetrievalError, build_index, load_index, make_chunks, search,
)


class TestTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}

    def encode(self, text):
        return [0] + text.split() + [1]


class TestEncoder:
    tokenizer = TestTokenizer()
    max_tokens = 256
    revision = "offline-test-encoder"

    def encode(self, texts):
        return np.array([
            [1 + text.lower().count("python"), 1 + text.lower().count("college"), 1]
            for text in texts
        ], dtype="float32")


def document(text):
    return {"schema_version": 1, "source": "fictional.pdf",
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "pages": [{"page": 1, "text": text}]}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "resume.rag.zip"
        self.encoder = TestEncoder()
        self.doc = document("SKILLS\nPython Python Python APIs\nEDUCATION\nCollege College engineering degree")

    def test_section_boundaries_and_exact_source_offsets(self):
        chunks = make_chunks(self.doc, self.encoder)
        self.assertEqual([c["section"] for c in chunks], ["SKILLS", "EDUCATION"])
        for chunk in chunks:
            self.assertEqual(chunk["text"], self.doc["pages"][0]["text"][chunk["start_char"]:chunk["end_char"]])
        self.assertEqual(chunks, make_chunks(self.doc, self.encoder))

    def test_long_content_is_covered_and_overlaps(self):
        doc = document("PROJECTS\n" + " ".join(f"word{i}" for i in range(400)))
        chunks = make_chunks(doc, self.encoder, max_tokens=64, overlap=8)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["token_count"] <= 64 for c in chunks))
        all_words = set(" ".join(c["text"] for c in chunks).split())
        self.assertEqual(all_words, {f"word{i}" for i in range(400)})
        self.assertLess(chunks[1]["start_char"], chunks[0]["end_char"])

    def test_project_and_job_boundaries(self):
        doc = document("EXPERIENCE\nDeveloper\nJan 2024 - Jul 2025\nPython APIs\nDesigner\nAug 2025 - Present\nFigma\nSELECTED PROJECTS\nAlpha | Chat app\nReact\nBeta | Finance app\nSQL")
        labels = [c["label"] for c in make_chunks(doc, self.encoder)]
        self.assertEqual(len(labels), 4)
        self.assertIn("Developer", labels[0])
        self.assertIn("Designer", labels[1])
        self.assertIn("Alpha", labels[2])
        self.assertIn("Beta", labels[3])

    def test_persist_reload_and_rank_with_real_faiss(self):
        build_index(self.doc, self.encoder, self.path)
        index, manifest = load_index(self.path)
        hits = search(index, manifest, self.encoder, "Python", top_k=20)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["section"], "SKILLS")
        self.assertGreater(hits[0]["score"], hits[1]["score"])
        self.assertTrue(-1.001 <= hits[0]["score"] <= 1.001)

    def test_all_projects_request_returns_project_coverage(self):
        doc = document("Candidate Name\nPython profile\nSELECTED PROJECTS\nAlpha | Python app\nPython APIs\nBeta | College tool\nCollege schedules\nGamma | Finance app\nSQL reporting")
        build_index(doc, self.encoder, self.path)
        index, manifest = load_index(self.path)
        hits = search(index, manifest, self.encoder, "Name all three selected projects")
        self.assertEqual(len(hits), 3)
        self.assertTrue(all(hit["section"] == "SELECTED PROJECTS" for hit in hits))
        self.assertEqual({hit["label"].split(": ")[1].split(" | ")[0] for hit in hits}, {"Alpha", "Beta", "Gamma"})

    def test_overwrite_protection(self):
        build_index(self.doc, self.encoder, self.path)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(RetrievalError, "already exists"):
            build_index(self.doc, self.encoder, self.path)
        self.assertEqual(before, self.path.read_bytes())
        build_index(self.doc, self.encoder, self.path, overwrite=True)

    def test_query_validation_and_model_mismatch(self):
        build_index(self.doc, self.encoder, self.path)
        index, manifest = load_index(self.path)
        for question, k in [(" ", 3), ("Python", 0), ("x" * 2001, 3)]:
            with self.assertRaises(RetrievalError):
                search(index, manifest, self.encoder, question, k)
        self.encoder.revision = "different-model"
        with self.assertRaisesRegex(RetrievalError, "revision"):
            search(index, manifest, self.encoder, "Python")

    def test_bad_chunk_config_and_invalid_document(self):
        for maximum, overlap in [(32, 32), (192, -1), (300, 32), (4, 0)]:
            with self.assertRaises(RetrievalError):
                make_chunks(self.doc, self.encoder, maximum, overlap)
        for doc in [{}, document(""), {**self.doc, "source_sha256": "bad"}]:
            with self.assertRaises(RetrievalError):
                make_chunks(doc, self.encoder)

    def test_tampered_index_and_mismatched_metadata_are_rejected(self):
        build_index(self.doc, self.encoder, self.path)
        with zipfile.ZipFile(self.path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            binary = archive.read("index.faiss")
        for data, metadata in [(binary + b"corrupted", manifest),
                               (binary, {**manifest, "chunks": []})]:
            with zipfile.ZipFile(self.path, "w") as archive:
                archive.writestr("manifest.json", json.dumps(metadata))
                archive.writestr("index.faiss", data)
            with self.assertRaises(RetrievalError):
                load_index(self.path)


if __name__ == "__main__":
    unittest.main()
