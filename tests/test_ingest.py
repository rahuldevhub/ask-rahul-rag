"""Exercise extraction with synthetic PDF bytes; no real resume or API key."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from pypdf import PdfReader, PdfWriter

from resume_rag.ingest import IngestionError, extract_resume, main, save_extraction


def pdf_bytes(text: str) -> bytes:
    """Small, valid one-page test fixture held in memory."""
    content = f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset = len(result)
    result.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(result)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf = self.root / "sample.pdf"
        self.pdf.write_bytes(pdf_bytes("Fictional Candidate builds Python applications and retrieval systems."))

    def test_actual_pdf_extraction_and_page_provenance(self):
        document = extract_resume(self.pdf)
        self.assertIn("Fictional Candidate", document["pages"][0]["text"])
        self.assertEqual(document["pages"][0]["page"], 1)
        self.assertEqual(document["source"], "sample.pdf")
        self.assertEqual(len(document["source_sha256"]), 64)
        self.assertEqual(document["warnings"], [])

    def test_contact_privacy_preserves_email_and_original_pdf(self):
        original = pdf_bytes("Candidate +91 9876543210 candidate@example.com builds Python applications.")
        self.pdf.write_bytes(original)
        document = extract_resume(self.pdf)
        text = document["pages"][0]["text"]
        self.assertNotIn("9876543210", text)
        self.assertIn("[phone number withheld]", text)
        self.assertIn("candidate@example.com", text)
        self.assertEqual(self.pdf.read_bytes(), original)

    def test_layout_extraction(self):
        self.assertIn("retrieval systems", extract_resume(self.pdf, "layout")["pages"][0]["text"])

    def test_cli_writes_reviewable_outputs(self):
        output = self.root / "processed"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.pdf), "--output-dir", str(output)]), 0)
        document = json.loads((output / "resume.json").read_text())
        self.assertEqual(document["page_count"], 1)
        self.assertIn("--- Page 1 ---", (output / "resume.txt").read_text())

    def test_refuses_accidental_overwrite(self):
        output = self.root / "processed"
        document = extract_resume(self.pdf)
        save_extraction(document, output)
        with self.assertRaisesRegex(IngestionError, "already exist"):
            save_extraction(document, output)
        save_extraction(document, output, overwrite=True)

    def test_empty_pdf_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.write(self.pdf)
        with self.assertRaisesRegex(IngestionError, "No text"):
            extract_resume(self.pdf)

    def test_blank_page_warns_without_losing_page_numbers(self):
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_page(PdfReader(self.pdf).pages[0])
        writer.write(self.pdf)
        document = extract_resume(self.pdf)
        self.assertEqual(document["pages"][1]["page"], 2)
        self.assertIn("Page 1", document["warnings"][0])

    def test_encrypted_pdf_rejected(self):
        writer = PdfWriter()
        writer.add_page(PdfReader(self.pdf).pages[0])
        writer.encrypt("test-password")
        writer.write(self.pdf)
        with self.assertRaisesRegex(IngestionError, "encrypted"):
            extract_resume(self.pdf)

    def test_invalid_pdf_and_missing_file_fail_cleanly(self):
        self.pdf.write_bytes(b"%PDF-1.4\ninvalid document")
        with self.assertRaisesRegex(IngestionError, "Could not read"):
            extract_resume(self.pdf)
        with self.assertRaisesRegex(IngestionError, "not found"):
            extract_resume(self.root / "missing.pdf")

    def test_failure_does_not_create_output(self):
        output = self.root / "processed"
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(self.root / "missing.pdf"), "--output-dir", str(output)]), 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
