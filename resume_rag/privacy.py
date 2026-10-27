"""Contact-data policy: withhold phone numbers while preserving email addresses."""

import re

PHONE_CANDIDATE = re.compile(r"(?<![\w@])\+?\d[\d \t().-]{7,}\d(?![\w@])")


def redact_phone_numbers(text: str) -> str:
    def replace(match):
        candidate = match.group()
        # Similarity scores are decimals in [-1, 1], not contact numbers.
        if re.fullmatch(r"[01]\.\d+", candidate.strip()):
            return candidate
        count = sum(char.isdigit() for char in candidate)
        return "[phone number withheld]" if 10 <= count <= 15 else candidate
    return PHONE_CANDIDATE.sub(replace, text)


def contains_phone_number(text: str) -> bool:
    return redact_phone_numbers(text) != text
