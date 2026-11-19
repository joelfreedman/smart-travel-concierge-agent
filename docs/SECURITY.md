# Security Architecture, STRIDE Threat Model & Verification Plan

## 1. Security Boundaries

1. **User Ingress (`AgentOpsGuardrailPlugin.on_user_message_callback`)**
   - Every user message is scanned for payment, identity, and contact PII via `PIIRedactor` (Google Cloud DLP API with deterministic local regex + Luhn fallback) **before** reaching any Gemini prompt, log sink, or database table.
   - Prompt-injection patterns attempting to override system rules, bypass budget ceilings, or auto-approve bookings short-circuit in `before_model_callback` before any LLM call is made.
2. **Tool Argument & Output Validation (`travel_concierge/schemas.py`)**
   - Every tool validates incoming LLM parameters against strict Pydantic v2 models (`ConfigDict(extra="forbid")`), enforcing IATA format, positive finite budgets, bounded traveler counts, and chronological trip dates.
   - All error conditions return structured `StructuredToolError` payloads without exposing raw stack traces or database internals.
3. **Fail-Closed Human-in-the-Loop (HITL) Booking Gate**
   - `execute_confirmed_booking` is wrapped in ADK's native `FunctionTool(..., require_confirmation=True)` and calls `tool_context.request_confirmation(...)` when confirmation is absent.
   - Every booking hold is persisted in SQLite with a SHA-256 `quote_fingerprint`, expiration timestamp, owner `traveler_id`, and `status='pending'`.
   - Execution requires an atomic SQL transition (`WHERE approval_id = ? AND status = 'approved'`) verifying exact traveler match, exact USD amount match, non-expiration, and single-use consumption (preventing replay or race-condition double-booking).
4. **Persistence & Secret Management**
   - All SQLite queries in `PersistentTravelStore` are 100% parameterized (`?` placeholders) to prevent SQL injection (CWE-89).
   - `SecretProvider` loads production secrets from Google Cloud Secret Manager (`google-cloud-secret-manager`) or explicit environment variables in local dev, failing closed (`SecretResolutionError`) rather than falling back to hardcoded literals.
   - Local HTTP server (`travel_concierge/app_server.py`) binds strictly to `127.0.0.1` by default.

## 2. STRIDE Threat Model

| STRIDE Pillar | Threat Scenario | Concrete Mitigation | Verification Test |
|---|---|---|---|
| **Spoofing** | Traveler A attempts to consume Traveler B's `approval_id`. | `consume_approved_booking` enforces `record["traveler_id"] == traveler_id`. | `tests/test_security_and_hitl.py` |
| **Tampering** | LLM or caller alters booking charge after human approved a lower amount. | `quote_fingerprint` + exact `expected_total_cost_usd` equality check (`AMOUNT_MISMATCH`). | `tests/test_security_and_hitl.py` |
| **Repudiation** | Disputed booking or unlogged policy override. | Structured JSON `agent.intent` (pre-execution) + `agent.outcome` (post-execution) logs correlated with OpenTelemetry `trace_id`/`span_id` and `approved_by`/`approved_at` DB columns. | `tests/test_observability_and_routing.py` |
| **Information Disclosure** | User pastes credit card, passport, SSN, or email into chat; leaked to LLM, logs, or SQLite. | Multi-boundary `PIIRedactor` (ingress, pre-model, post-model, JSON log formatter, SQLite writes, compaction summaries). | `tests/test_security_and_hitl.py`, `evals/test_golden_evals.py` |
| **Denial of Service** | Context window bloat across long sessions or excessive single-trip budget abuse. | ADK `EventsCompactionConfig` (`SlidingWindowEventSummarizer`) + `MAX_SINGLE_TRIP_BUDGET_USD` guardrail + bounded async concurrency semaphore (`max_concurrent_tasks=8`). | `tests/test_memory_and_compaction.py`, `tests/test_security_and_hitl.py` |
| **Elevation of Privilege** | Prompt injection attempts to auto-approve a booking or claim confirmation without tool execution. | Pre-LLM injection short-circuit + dual-layer HITL confirmation gate + post-model self-eval blocking unverified confirmation claims. | `tests/test_security_and_hitl.py`, `evals/test_golden_evals.py` |

## 3. Verification Plan (Automated Security Checks)

- **Unit & Security Tests**: `uv run pytest tests/` (verifies PII scrubbing, Cloud DLP integration, prompt-injection containment, budget cap enforcement, self-eval correction, dual-layer HITL fail-closed checks, single-use replay prevention, and Secret Manager fail-closed resolution).
- **Golden Eval Regression Suite**: `uv run pytest evals/test_golden_evals.py` (verifies ADK `EvalSet`/`EvalConfig` compliance and all 6 security/functional trajectories).
- **Static Analysis & Secret Scanning**:
  - `uv run ruff check travel_concierge tests evals` (includes `flake8-bandit` `S` security rules).
  - Custom Semgrep rules in `.semgrep/rules.yaml` catching hardcoded Google API keys, raw `print()` calls in runtime code, and interpolated SQL statements.
- **Infrastructure Least-Privilege**: Terraform IAM bindings in `infra/terraform/main.tf` grant only scoped runtime roles (`aiplatform.user`, `secretmanager.secretAccessor`, `dlp.user`, `cloudtrace.agent`, `logging.logWriter`) to a dedicated service account with `public_access_prevention = "enforced"` on GCS staging storage.
