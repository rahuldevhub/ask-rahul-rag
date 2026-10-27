"""Extract resume text locally, keeping page provenance for later retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .privacy import redact_phone_numbers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PAGES = 20
LOW_TEXT_CHARACTERS = 40


class IngestionError(ValueError):
    """An input cannot be safely turned into useful resume text."""


def normalize_text(text: str) -> str:
    """Clean whitespace without discarding paragraph and line boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "").replace("\u00a0", " ")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_resume(path: Path, mode: str = "plain") -> dict:
    """Return JSON-ready text and provenance; make no external API requests."""
    path = Path(path)
    if mode not in {"plain", "layout"}:
        raise IngestionError("Extraction mode must be plain or layout.")
    if not path.is_file():
        raise IngestionError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise IngestionError("Please provide a PDF file.")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise IngestionError("PDF exceeds the 10 MB resume limit.")

    pages = []
    warnings = []
    try:
        # Keep the reader in memory, without leaving an open source-file handle.
        import io

        source_bytes = path.read_bytes()
        reader = PdfReader(io.BytesIO(source_bytes))
        if reader.is_encrypted:
            raise IngestionError("PDF is encrypted. Export an unencrypted copy first.")
        if len(reader.pages) > MAX_PAGES:
            raise IngestionError("PDF exceeds the 20-page resume limit.")
        for number, page in enumerate(reader.pages, start=1):
            text = redact_phone_numbers(normalize_text(page.extract_text(extraction_mode=mode) or ""))
            if len(text) < LOW_TEXT_CHARACTERS:
                warnings.append(
                    f"Page {number} has little or no extractable text. "
                    "Check it visually; scanned content may need OCR."
                )
            pages.append({"page": number, "text": text, "characters": len(text)})
    except IngestionError:
        raise
    except (PdfReadError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise IngestionError("Could not read this PDF. Try exporting a fresh PDF copy.") from exc

    if not any(page["text"] for page in pages):
        raise IngestionError(
            "No text could be extracted. If the PDF is scanned, it needs OCR first."
        )

    return {
        "schema_version": 1,
        "privacy": {"phone_numbers": "withheld", "email": "retained"},
        "source": path.name,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "extraction_mode": mode,
        "page_count": len(pages),
        "character_count": sum(page["characters"] for page in pages),
        "warnings": warnings,
        "pages": pages,
    }


def save_extraction(document: dict, output_dir: Path, overwrite: bool = False) -> tuple[Path, Path]:
    """Write structured data for Phase 2 and a readable copy for human review."""
    output_dir = Path(output_dir)
    json_path = output_dir / "resume.json"
    text_path = output_dir / "resume.txt"
    if not overwrite and (json_path.exists() or text_path.exists()):
        raise IngestionError(
            "Extraction files already exist. Review them first, then use --overwrite to replace them."
        )
    readable = "\n\n".join(
        f"--- Page {page['page']} ---\n{page['text']}" for page in document["pages"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(readable + "\n", encoding="utf-8")
    return json_path, text_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to your resume PDF")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("plain", "layout"), default="plain")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        document = extract_resume(args.pdf, mode=args.mode)
        json_path, text_path = save_extraction(document, args.output_dir, args.overwrite)
    except (IngestionError, OSError) as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1
    print(f"Extracted {document['page_count']} page(s), {document['character_count']} characters.")
    print(f"Structured data: {json_path}")
    print(f"Review copy: {text_path}")
    for warning in document["warnings"]:
        print(f"Warning: {warning}")
    print("Review the text against the PDF before proceeding to chunking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
