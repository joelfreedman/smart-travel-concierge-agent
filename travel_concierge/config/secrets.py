"""Secure secret resolution backed by Google Cloud Secret Manager and env vars.

Security invariants:
- Never hardcode API keys, tokens, or fallback literal secret strings.
- Never log raw secret material.
- Fail closed (raise SecretResolutionError) when a required secret is unavailable.
"""

from __future__ import annotations

import os
import re
from typing import Any, Protocol

from travel_concierge.config.settings import AppSettings

_SECRET_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,255}$")
_PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")


class SecretResolutionError(RuntimeError):
    """Raised when a required secret cannot be securely resolved."""


class SecretManagerClientProtocol(Protocol):
    """Minimal protocol matching google.cloud.secretmanager.SecretManagerServiceClient."""

    def access_secret_version(self, request: dict[str, str]) -> Any: ...


class SecretProvider:
    """Resolves application secrets from Google Cloud Secret Manager or environment."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        client: SecretManagerClientProtocol | None = None,
    ) -> None:
        self._settings = settings or AppSettings.from_env()
        self._client = client
        self._cache: dict[str, str] = {}

    def _get_client(self) -> SecretManagerClientProtocol:
        """Lazily initialize the Google Cloud Secret Manager client."""
        if self._client is None:
            from google.cloud import secretmanager

            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def get_secret(
        self,
        secret_id: str,
        *,
        env_var: str | None = None,
        version: str = "latest",
    ) -> str:
        """Retrieve a secret securely without ever falling back to hardcoded literals.

        Args:
            secret_id: Secret Manager secret identifier (alphanumeric, hyphen, underscore).
            env_var: Optional environment variable name for local development or
                Cloud Run Secret Manager env-var mounting.
            version: Secret Manager version label or number (defaults to ``latest``).

        Returns:
            The resolved secret string.

        Raises:
            SecretResolutionError: If validation fails or the secret is missing.
        """
        if not _SECRET_ID_PATTERN.match(secret_id):
            raise SecretResolutionError("Invalid Secret Manager secret identifier format.")

        cache_key = f"{secret_id}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Production / GCP Secret Manager direct fetch when explicitly enabled.
        if self._settings.use_secret_manager:
            project_id = self._settings.gcp_project_id
            if not project_id or not _PROJECT_ID_PATTERN.match(project_id):
                raise SecretResolutionError(
                    "GOOGLE_CLOUD_PROJECT must be set to a valid GCP project ID "
                    "when USE_SECRET_MANAGER=true."
                )
            resource_name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
            try:
                response = self._get_client().access_secret_version(
                    request={"name": resource_name}
                )
                secret_value = response.payload.data.decode("utf-8").strip()
            except Exception as exc:
                raise SecretResolutionError(
                    f"Failed to access secret '{secret_id}' from Google Cloud Secret Manager."
                ) from exc

            if not secret_value:
                raise SecretResolutionError(
                    f"Secret '{secret_id}' in Google Cloud Secret Manager is empty."
                )
            self._cache[cache_key] = secret_value
            return secret_value

        # 2. Environment variable injection (local dev or Cloud Run secret mount).
        if env_var:
            env_val = os.getenv(env_var, "").strip()
            if env_val:
                self._cache[cache_key] = env_val
                return env_val

        raise SecretResolutionError(
            f"Required secret '{secret_id}' is not configured. Set environment variable "
            f"'{env_var or secret_id}' for local dev or enable USE_SECRET_MANAGER=true."
        )

    def resolve_gemini_api_key(self) -> str | None:
        """Resolve the Gemini API key unless ADC/Vertex AI authentication is active.

        When ``GOOGLE_GENAI_USE_VERTEXAI=true``, Application Default Credentials (ADC)
        are used and no API key is required.
        """
        if self._settings.use_vertex_ai:
            return None

        # Respect standard Google GenAI / ADK env var names.
        for candidate_env in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if not self._settings.use_secret_manager and os.getenv(candidate_env, "").strip():
                return self.get_secret(
                    self._settings.gemini_api_key_secret_name,
                    env_var=candidate_env,
                )

        return self.get_secret(
            self._settings.gemini_api_key_secret_name,
            env_var="GEMINI_API_KEY",
        )
