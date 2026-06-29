from __future__ import annotations

import re


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def redact_basic_personal_info(text: str) -> str:
    """Redact direct contact and ID fields while keeping skill evidence intact."""
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    text = ID_CARD_RE.sub("[ID_CARD_REDACTED]", text)
    return text
