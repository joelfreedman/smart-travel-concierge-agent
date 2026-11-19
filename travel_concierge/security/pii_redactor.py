"""Active PII scrubbing pipeline with Google Cloud DLP integration and local fallback.

Used across three critical boundaries:
1. Ingress: before user messages reach any LLM prompt.
2. Observability: before any structured JSON log or trace attribute is emitted.
3. Persistence: before traveler preferences or compacted summaries are stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from travel_concierge.config.settings import AppSettings


@dataclass(frozen=True)
class RedactionResult:
    """Result of scanning and redacting text or structured payloads for PII."""

    redacted_text: str
    redacted_categories: tuple[str, ...] = field(default_factory=tuple)
    backend_used: str = "local_regex"

    @property
    def had_pii(self) -> bool:
        """Return True if at least one sensitive entity was scrubbed."""
        return len(self.redacted_categories) > 0


# Deterministic patterns ordered from highest specificity to broadest match.
_API_KEY_PATTERN = re.compile(r"\bAIzaSy[A-Za-z0-9_\-]{10,60}\b")
_SSN_PATTERN = re.compile(
    r"(?:\b\d{3}-\d{2}-\d{4}\b|\b(?:ssn|social security(?:\s+number)?)\s*[:#=-]?\s*\d{9,11}\b)",
    re.IGNORECASE,
)
_PASSPORT_PATTERN = re.compile(
    r"\bpassport(?:\s*(?:no\.?|num\.?|number|#))?\s*(?:(?:is\s+)|(?:[:#=-]\s*))?(?=[A-Z0-9]*\d)[A-Z0-9]{6,12}\b",
    re.IGNORECASE,
)
_CREDIT_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)
_DOB_PATTERN = re.compile(
    r"\b(?:dob|date of birth|born on)\s*[:#=-]?\s*(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b",
    re.IGNORECASE,
)


def _passes_luhn(digits: str) -> bool:
    """Verify a candidate payment card digit string using the Luhn checksum."""
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    reverse_digits = digits[::-1]
    for index, ch in enumerate(reverse_digits):
        num = int(ch)
        if index % 2 == 1:
            num *= 2
            if num > 9:
                num -= 9
        total += num
    return total % 10 == 0


class PIIRedactor:
    """Scrubs payment, identity, and contact PII using Cloud DLP or deterministic rules."""

    CLOUD_DLP_INFO_TYPES: tuple[str, ...] = (
        "CREDIT_CARD_NUMBER",
        "US_SOCIAL_SECURITY_NUMBER",
        "PASSPORT",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "DATE_OF_BIRTH",
    )

    def __init__(
        self,
        settings: AppSettings | None = None,
        dlp_client: Any | None = None,
    ) -> None:
        self._settings = settings or AppSettings.from_env()
        self._dlp_client = dlp_client

    def _can_use_cloud_dlp(self) -> bool:
        return bool(self._settings.enable_cloud_dlp and self._settings.gcp_project_id)

    def _get_dlp_client(self) -> Any:
        if self._dlp_client is None:
            from google.cloud import dlp_v2

            self._dlp_client = dlp_v2.DlpServiceClient()
        return self._dlp_client

    def _redact_with_cloud_dlp(self, text: str) -> RedactionResult | None:
        """Attempt de-identification via Google Cloud Sensitive Data Protection (DLP)."""
        if not self._can_use_cloud_dlp():
            return None
        try:
            client = self._get_dlp_client()
            parent = f"projects/{self._settings.gcp_project_id}/locations/global"
            inspect_config = {
                "info_types": [{"name": info_type} for info_type in self.CLOUD_DLP_INFO_TYPES]
            }
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            }
            response = client.deidentify_content(
                request={
                    "parent": parent,
                    "deidentify_config": deidentify_config,
                    "inspect_config": inspect_config,
                    "item": {"value": text},
                }
            )
            redacted = response.item.value
            categories: list[str] = []
            for info_type in self.CLOUD_DLP_INFO_TYPES:
                if f"[{info_type}]" in redacted and f"[{info_type}]" not in text:
                    categories.append(info_type)
            return RedactionResult(
                redacted_text=redacted,
                redacted_categories=tuple(sorted(set(categories))),
                backend_used="google_cloud_dlp",
            )
        except Exception:
            # Fail safe by falling through to deterministic local redaction.
            return None

    def redact_text(self, text: str) -> RedactionResult:
        """Scrub PII and credential patterns from arbitrary text."""
        if not text:
            return RedactionResult(redacted_text=text)

        categories: set[str] = set()
        backend = "local_regex"
        working = text

        dlp_result = self._redact_with_cloud_dlp(working)
        if dlp_result is not None:
            working = dlp_result.redacted_text
            categories.update(dlp_result.redacted_categories)
            backend = dlp_result.backend_used

        # Always run deterministic local pass as defense-in-depth (and primary offline mode).
        if _API_KEY_PATTERN.search(working):
            working = _API_KEY_PATTERN.sub("[REDACTED_API_KEY]", working)
            categories.add("API_KEY")

        if _SSN_PATTERN.search(working):
            working = _SSN_PATTERN.sub("[REDACTED_SSN]", working)
            categories.add("US_SOCIAL_SECURITY_NUMBER")

        if _PASSPORT_PATTERN.search(working):
            working = _PASSPORT_PATTERN.sub("[REDACTED_PASSPORT]", working)
            categories.add("PASSPORT")

        def _replace_card(match: re.Match[str]) -> str:
            raw = match.group(0)
            digits = re.sub(r"\D", "", raw)
            if _passes_luhn(digits):
                categories.add("CREDIT_CARD_NUMBER")
                return "[REDACTED_CREDIT_CARD]"
            return raw

        working = _CREDIT_CARD_CANDIDATE.sub(_replace_card, working)

        if _EMAIL_PATTERN.search(working):
            working = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", working)
            categories.add("EMAIL_ADDRESS")

        if _PHONE_PATTERN.search(working):
            working = _PHONE_PATTERN.sub("[REDACTED_PHONE]", working)
            categories.add("PHONE_NUMBER")

        if _DOB_PATTERN.search(working):
            working = _DOB_PATTERN.sub("[REDACTED_DOB]", working)
            categories.add("DATE_OF_BIRTH")

        return RedactionResult(
            redacted_text=working,
            redacted_categories=tuple(sorted(categories)),
            backend_used=backend,
        )

    def redact_mapping(self, data: Any) -> tuple[Any, tuple[str, ...]]:
        """Recursively scrub all string values inside dicts, lists, and tuples."""
        found: set[str] = set()

        def _walk(obj: Any) -> Any:
            if isinstance(obj, str):
                result = self.redact_text(obj)
                found.update(result.redacted_categories)
                return result.redacted_text
            if isinstance(obj, dict):
                redacted_dict: dict[str, Any] = {}
                for key, val in obj.items():
                    lower_key = str(key).lower()
                    if any(
                        token in lower_key
                        for token in ("api_key", "secret", "password", "card_number", "cvv", "ssn")
                    ):
                        found.add("SENSITIVE_FIELD")
                        redacted_dict[key] = "[REDACTED_SECRET]"
                    else:
                        redacted_dict[key] = _walk(val)
                return redacted_dict
            if isinstance(obj, list):
                return [_walk(item) for item in obj]
            if isinstance(obj, tuple):
                return tuple(_walk(item) for item in obj)
            return obj

        scrubbed = _walk(data)
        return scrubbed, tuple(sorted(found))
