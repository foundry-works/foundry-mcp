# Plan: Alpha Branch Review Remediation

## Context

Senior engineer review of the `alpha` branch (38 commits, 188 files, +69k/-8.9k lines) identified issues across configuration parsing, concurrency safety, security, CI, and repo hygiene. This plan tracks remediation before merge to `main`.

---

## P0: Critical — Silent Config Data Loss

### Claim Verification Config Fields Not Parsed

**File**: `src/foundry_mcp/config/research.py:611-825`

**Problem**: All 10 `deep_research_claim_verification_*` fields are declared as dataclass attributes (lines 330-339) but are never passed into the `cls(...)` constructor call in `from_toml_dict()`. TOML configuration for claim verification is silently dropped. The unknown-key warning system does not catch this because the fields exist in `dataclasses.fields(cls)`.

**Root cause**: The `from_toml_dict` method is a ~430-line manual field mapping function. Any new field added to the class but missed in this method is silently ignored. This is exactly what happened with the 10 claim verification fields.

**Fix**:
1. Add all 10 missing fields to the `cls(...)` constructor call in `from_toml_dict()`:
   - `deep_research_claim_verification_enabled`
   - `deep_research_claim_verification_sample_rate`
   - `deep_research_claim_verification_provider`
   - `deep_research_claim_verification_model`
   - `deep_research_claim_verification_timeout`
   - `deep_research_claim_verification_max_claims`
   - `deep_research_claim_verification_max_concurrent`
   - `deep_research_claim_verification_max_corrections`
   - `deep_research_claim_verification_annotate_unsupported`
   - `deep_research_claim_verification_max_input_tokens`
2. Add `_validate_claim_verification_config()` to `__post_init__` validation chain. Validate:
   - `sample_rate` in `[0.0, 1.0]`
   - `timeout` > 0
   - `max_claims` >= 1
   - `max_concurrent` >= 1
   - `max_corrections` >= 0
   - `max_input_tokens` > 0
3. Add the claim verification fields to `DeepResearchSettings` sub-config in `research_sub_configs.py`.
4. Add a regression test that asserts every `dataclasses.fields(ResearchConfig)` field name appears in the `from_toml_dict` constructor call. This prevents recurrence of this class of bug.

**Scope**: ~50 lines config, ~30 lines validation, ~20 lines test

---

## P1: High — Concurrency & Error Handling

### 1.1 `asyncio.gather` Missing `return_exceptions=True`

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py:496`

**Problem**: `_verify_claims_batch` uses `asyncio.gather(*tasks, return_exceptions=False)`. A single unexpected exception (network error, cancellation) aborts ALL remaining verification tasks. Other gather calls in the codebase (compression.py:997, supervision) correctly use `return_exceptions=True`.

**Fix**: Change to `return_exceptions=True` and handle exception instances in the result processing loop. Pattern matches compression.py:997.

**Scope**: ~5 lines

### 1.2 Dict Iteration Without Snapshot in Concurrent Cleanup

**File**: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py:662`

**Problem**: `_active_sessions` is iterated during cleanup while background threads may mutate it. RuntimeError possible on dict size change during iteration.

**Fix**: Iterate over `list(self._active_sessions.items())` to snapshot before cleanup.

**Scope**: 1 line

### 1.3 Background Task State Mutations Without Synchronization

**File**: `src/foundry_mcp/core/research/workflows/deep_research/background_tasks.py:96-223`

**Problem**: `DeepResearchState` is shared between daemon threads and the main thread. While CPython's GIL protects individual attribute writes, composite operations (read-modify-write on lists/dicts in state) can produce inconsistent views.

**Assessment**: This is a known limitation of the daemon thread architecture. A full fix (lock-based or message-passing) is out of scope for this remediation. Document the constraint and add a comment at the thread entry point.

**Scope**: Comment-level documentation

---

## P2: Medium — Correctness & Security

### 2.1 OpenAlex Filter Injection via Unvalidated Reference IDs

**File**: `src/foundry_mcp/core/research/providers/openalex.py:372-377, 406-411`

**Problem**: IDs from OpenAlex API responses (`referenced_works`, `related_works`) are pipe-joined into a filter string without validation. A compromised API response could inject filter clauses via `,` (the OpenAlex filter separator).

**Fix**: Validate each ID from `referenced_works`/`related_works` against `_OPENALEX_WORK_ID_RE` before joining. Discard IDs that don't match.

**Scope**: ~10 lines

### 2.2 SIGTERM Handler Bypasses `cancel()` API

**File**: `src/foundry_mcp/core/research/workflows/deep_research/infrastructure.py:192`

**Problem**: Signal handler directly mutates state instead of going through the public `cancel()` method, which handles audit events and proper cleanup.

**Fix**: Have the signal handler call `self.cancel()` instead of direct mutation.

**Scope**: ~3 lines

### 2.3 `PARTIALLY_SUPPORTED` Claims Uncounted in Aggregation

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py:808-815`

**Problem**: `PARTIALLY_SUPPORTED` verdicts don't increment any counter, so `claims_verified != claims_supported + claims_contradicted + claims_unsupported`. This creates a silent accounting gap.

**Fix**: Add `claims_partially_supported: int = 0` to `ClaimVerificationResult` and count it in the aggregation loop.

**Scope**: ~5 lines

### 2.4 Inline Compression Skips Structured Data Validation

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py:1543-1547`

**Problem**: `_apply_inline_compression_async` does not pass `structured_blocks` to `_compression_output_is_valid`, so tables/numeric data loss goes undetected in the inline path. The batch compression path (compression.py:1026-1030) correctly passes it.

**Fix**: Detect structured blocks before inline compression and pass to validation, matching the batch path.

**Scope**: ~8 lines

### 2.5 Token Budget Returns Full Text When Exhausted

**File**: `src/foundry_mcp/core/research/workflows/deep_research/_token_budget.py:77-82`

**Problem**: When remaining budget is 0 or negative, `budget_truncate_text` returns the full text rather than an empty string. This is counter-intuitive and defeats the purpose of budget tracking.

**Fix**: Return empty string when budget <= 0.

**Scope**: ~3 lines

### 2.6 Inconsistent Timeout Types

**File**: `src/foundry_mcp/config/research.py:159, 334`

**Problem**: `deep_research_summarization_timeout` and `deep_research_claim_verification_timeout` are typed as `int` while all other timeout fields are `float`. The timeout preset multiplier produces floats that get truncated.

**Fix**: Change both to `float` for consistency.

**Scope**: 2 lines

---

## P3: Low — Hygiene & Hardening

### 3.1 Remove Committed Research Artifact

**File**: `conversation_assessment_research.md`

**Problem**: This is a 293-line generated research output (academic report on "Conversation-Based Assessment in Education"), not source code. Should not be in the repo.

**Fix**: Delete the file and add a gitignore pattern for generated research reports.

### 3.2 Remove Runtime State File

**File**: `dev_docs/plans/specs/.autonomy/context/context.json`

**Problem**: Runtime autonomy session state. Should be gitignored.

**Fix**: Delete the file and add `.autonomy/` to `.gitignore`.

### 3.3 Pin `softprops/action-gh-release` in CI

**File**: `.github/workflows/publish-alpha.yml`

**Problem**: `softprops/action-gh-release@v1` is a third-party action pinned only to major version. Supply chain risk. Also, v2 has been available for a long time.

**Fix**: Update to `@v2` or pin to specific commit SHA.

### 3.4 Narrow CI Tag Pattern

**File**: `.github/workflows/publish-alpha.yml`

**Problem**: Tag pattern `v*a*` is overly broad — matches any tag containing the letter "a" anywhere. Mitigated by the PEP 440 regex check but unnecessarily noisy.

**Fix**: Narrow to `v*a[0-9]*` or `v*.*.a*`.

### 3.5 `parse_researcher_response` Exception Scope Too Broad

**File**: `src/foundry_mcp/core/research/models/deep_research.py:702-706`

**Problem**: Catches `TypeError` alongside `json.JSONDecodeError` and `ValidationError`, which could mask genuine programming errors.

**Fix**: Remove `TypeError` from the catch clause.

### 3.6 Redundant Import

**File**: `src/foundry_mcp/core/research/workflows/deep_research/background_tasks.py:281`

**Problem**: Duplicate `import time`.

**Fix**: Remove the redundant import.

### 3.7 f-string in Logger Calls

**File**: `src/foundry_mcp/core/research/docx_extractor.py:421, 471-474, 519`

**Problem**: Using f-strings in log calls bypasses lazy evaluation. Should use `%s` formatting.

**Fix**: Change to `logger.warning("Failed to read DOCX: %s", e)` pattern.

### 3.8 Evaluator Score Truncation

**File**: `src/foundry_mcp/core/research/evaluation/evaluator.py:206`

**Problem**: `int(3.7)` truncates to 3 rather than rounding. LLM may return fractional scores.

**Fix**: Use `round(raw_score)` instead of `int(raw_score)`.

---

## Implementation Order

1. **P0** — Config parsing fix (critical, silent data loss affecting all users who configure claim verification)
2. **P1.1** — gather return_exceptions (high, could abort entire verification batch)
3. **P1.2** — Dict iteration snapshot (high, potential RuntimeError)
4. **P2.1–P2.6** — Medium fixes (can be batched into one commit)
5. **P3** — Low/hygiene fixes (can be batched into one commit)

## Testing

Run the existing test suite after each priority tier to ensure no regressions:
```bash
python -m pytest tests/ -x --timeout=60
```

For the P0 config fix specifically, verify with:
```bash
python -m pytest tests/unit/test_config_*.py -v
```
