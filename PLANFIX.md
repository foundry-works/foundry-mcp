# Post-Review Fix Plan v3 — `deep-academic` Branch

> **Branch**: `deep-academic`
>
> **Context**: Senior engineering review (v3) of 36 commits (~68K LOC added) implementing PLAN-0 through PLAN-4, including FIX-0 through FIX-5 from review v2. All 7,597 tests pass. This plan addresses remaining and newly discovered issues, organized by priority and dependency.
>
> **Total estimated scope**: ~200-320 LOC fixes + ~120-200 LOC test additions
>
> **Source**: Code review of `deep-academic` branch against `main`, conducted by 6 parallel review agents covering models/config, providers, phases, handlers/orchestration, tests, and new modules.
>
> **Relationship to prior reviews**: v1 (FIX-0) and v2 (FIX-1 through FIX-5) were implemented in commits `a9f0db8` through `243ac76`. This plan covers issues found after those fixes.

---

## Execution Order

```
FIX-0  Security hardening ──────────────────────────────┐ (no deps, do first)
FIX-1  Correctness fixes ───────────────────────────────┤ (parallel with FIX-0)
FIX-2  Provider robustness ─────────────────────────────┤ (after FIX-0)
FIX-3  Test quality ────────────────────────────────────┤ (parallel, independent)
FIX-4  Cleanup & consistency ───────────────────────────┘ (after FIX-1)
```

---

## FIX-0: Security Hardening

> **Scope**: ~40-60 LOC | **Risk**: Low (targeted fixes) | **Priority**: CRITICAL/HIGH

### Item 0.1: Fail Closed on DNS Resolution Failure in `pdf_extractor.py`

**Problem**: `pdf_extractor.py:289-291` catches `socket.gaierror` during SSRF validation and **allows the request to proceed**. The sibling implementation in `_injection_protection.py:131-132` correctly returns `False` (fail closed). An attacker could use a hostname that fails DNS for `getaddrinfo` but succeeds for httpx's resolver.

**File**: `src/foundry_mcp/core/research/pdf_extractor.py`

**Fix**: Replace the `except socket.gaierror` handler to raise `SSRFError` instead of logging debug and continuing. Match the `_injection_protection.py` pattern.

### Item 0.2: URL-Encode `paper_id` in Semantic Scholar URL Paths

**Problem**: `semantic_scholar.py:403,432` interpolates `paper_id` directly into URL paths (`f"{PAPER_ENDPOINT}/{paper_id}"`) with no URL encoding. Compare with OpenAlex which correctly uses `_url_quote(work_id, safe='')`. A malicious `paper_id` containing `../` or `?injected=param` could redirect the request.

**File**: `src/foundry_mcp/core/research/providers/semantic_scholar.py`

**Fix**: Apply `urllib.parse.quote(paper_id, safe='')` to all `paper_id` values used in URL path construction for `get_paper()`, `get_citations()`, and `get_recommendations()`.

### Item 0.3: Validate `work_id` in OpenAlex Filter String Construction

**Problem**: `openalex.py:312` passes caller-supplied `work_id` directly into the filter string via `f"cites:{work_id}"`. A value like `W123,publication_year:2024` injects additional filter conditions. The `_build_filter_string` helper (which sanitizes values) is not used for these internal filter constructions.

**Files**: `src/foundry_mcp/core/research/providers/openalex.py`

**Affected lines**: `get_citations` (line 312: `f"cites:{work_id}"`), `search_by_topic` (line 438: `f"topics.id:{topic_id}"`)

**Fix**: Validate `work_id` matches `^(W\d+|https://openalex\.org/W\d+)$` pattern and `topic_id` matches `^T\d+$` before interpolation. Raise `ValueError` on mismatch.

### Item 0.4: Remove `application/octet-stream` from Valid PDF Content Types

**Problem**: `pdf_extractor.py:167` accepts `application/octet-stream` as a valid PDF content type. This generic binary MIME type could be used to serve any binary payload, weakening the content-type gate. The downstream magic byte check mitigates partially, but the gate should be tight.

**File**: `src/foundry_mcp/core/research/pdf_extractor.py`

**Fix**: Remove `"application/octet-stream"` from `VALID_PDF_CONTENT_TYPES`. Rely on the `%PDF` magic byte check for ambiguous content types. Log a warning when `application/octet-stream` is encountered so operators can track servers that need the fallback.

---

## FIX-1: Correctness Fixes

> **Scope**: ~60-80 LOC | **Risk**: Low | **Priority**: HIGH/MEDIUM

### Item 1.1: Fix Evaluator Metadata Overwrite

**Problem**: `evaluator.py:337` uses `eval_result.metadata = {...}` which replaces the metadata dict built by `_parse_evaluation_response` (line 215), discarding the `imputed_count` and `warnings` keys.

**File**: `src/foundry_mcp/core/research/evaluation/evaluator.py`

**Fix**: Change `eval_result.metadata = {...}` to `eval_result.metadata.update({...})` at line 337.

### Item 1.2: Change `content_basis` to `Literal` Type

**Problem**: `MethodologyAssessment.content_basis` is typed as `str` but the model's safety invariant (force `confidence="low"` for abstract-only) depends on matching `"abstract"`. A misspelled value like `"Abstract"` silently bypasses the confidence downgrade.

**File**: `src/foundry_mcp/core/research/models/sources.py`

**Fix**: Change `content_basis: str` to `content_basis: Literal["abstract", "full_text"]` at line 577. Add `Literal` to the typing imports.

### Item 1.3: Add Missing Model Exports to `__init__.py`

**Problem**: New models (`ProvenanceLog`, `ProvenanceEntry`, `CitationNetwork`, `CitationNode`, `CitationEdge`, `ResearchLandscape`, `StudyComparison`, `ResearchThread`, `MethodologyAssessment`, `StudyDesign`) are not exported from `models/__init__.py`, violating the stated re-export contract.

**File**: `src/foundry_mcp/core/research/models/__init__.py`

**Fix**: Add imports and `__all__` entries for all new public models.

### Item 1.4: Add `MAX_ABSTRACT_POSITIONS` Cap in OpenAlex Provider

**Problem**: `openalex.py:130` (`max_pos = max(position_map.keys())`) allocates a list of `max_pos + 1` entries for abstract reconstruction. A malicious inverted index with position `999999999` causes OOM.

**File**: `src/foundry_mcp/core/research/providers/openalex.py`

**Fix**: Add `_MAX_ABSTRACT_POSITIONS = 100_000` constant. Return `None` if `max_pos > _MAX_ABSTRACT_POSITIONS`.

### Item 1.5: Add Upper Bound Validation on Citation Network Parameters

**Problem**: `handlers_deep_research.py` accepts `max_references_per_paper` and `max_citations_per_paper` from user input with no upper bound. A value of `999999` causes excessive API calls.

**File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

**Fix**: Clamp both values to `[1, 100]` range before passing to `CitationNetworkBuilder`.

### Item 1.6: Sanitize `research_id` Before Filesystem Path Use

**Problem**: `infrastructure.py:150-152` uses `research_id` directly in a file path for crash dumps. If `research_id` contains `../` or other traversal characters, the crash file could be written outside the expected directory.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/infrastructure.py`

**Fix**: Validate `research_id` matches `^[a-zA-Z0-9_-]+$` or use `Path(research_id).name` to strip directory components.

---

## FIX-2: Provider Robustness

> **Scope**: ~50-70 LOC | **Risk**: Low | **Priority**: MEDIUM

### Item 2.1: Replace `"404" in str(e)` with Structured Status Code Check

**Problem**: All three new providers (`openalex.py:295`, `crossref.py:218`, `semantic_scholar.py:410`) detect 404 by `"404" in str(e)`. A message like "Found 404 results matching..." false-positives.

**Files**: `src/foundry_mcp/core/research/providers/openalex.py`, `crossref.py`, `semantic_scholar.py`

**Fix**: Add `status_code: Optional[int] = None` attribute to `SearchProviderError`. Populate it in `_execute_request` when an HTTP error occurs. Check `e.status_code == 404` instead of string matching.

### Item 2.2: Extract Duplicated `classify_error` Logic from CrossrefProvider

**Problem**: `crossref.py:309-327` copy-pastes the `classify_error` logic from `SearchProvider.classify_error` in `base.py:270-283`. These will drift.

**Files**: `src/foundry_mcp/core/research/providers/shared.py`, `crossref.py`, `base.py`

**Fix**: Extract a `classify_with_registry(error, classifiers, provider_name)` function into `shared.py`. Have both `SearchProvider.classify_error` and `CrossrefProvider.classify_error` delegate to it.

### Item 2.3: Fix Stale Docstrings Referencing Removed Cost-Tier Defaults

**Problem**: `ResearchConfig` docstring (line 29-31) and `ModelRoleConfig` docstring (`research_sub_configs.py:79-83`) reference `gemini-2.5-flash` cost-tier defaults that were removed.

**Files**: `src/foundry_mcp/config/research.py`, `src/foundry_mcp/config/research_sub_configs.py`

**Fix**: Update docstrings to reflect current behavior — cost-tier defaults require explicit tier configuration.

### Item 2.4: Add Warning to Response on Failed Report Save

**Problem**: `handlers_deep_research.py:271` catches all exceptions when saving report to `output_path` and only logs a warning. The user receives no indication their save failed.

**File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

**Fix**: Add `"save_warning": str(exc)` to the response data dict when the save fails.

### Item 2.5: Add `deep-research-provenance` to MCP Tool Docstring

**Problem**: The `research()` MCP tool description (`__init__.py:267-273`) omits `deep-research-provenance` from the action list, making it undiscoverable by LLM agents.

**File**: `src/foundry_mcp/tools/unified/research_handlers/__init__.py`

**Fix**: Add `deep-research-provenance` to the action list in the tool description.

---

## FIX-3: Test Quality

> **Scope**: ~30-50 LOC fixes + ~80-120 LOC new tests | **Risk**: Very Low | **Priority**: HIGH (tautological tests), LOW (others)

### Item 3.1: Remove Tautological Tests in `test_sanitize_external_content.py`

**Problem**: 7 tests re-implement production logic inline and test themselves:
- `TestRawNotesCapping` (lines 712-820): 3 tests manually pop from `state.raw_notes` and assert the pop worked. They never call `_trim_raw_notes()`.
- `TestSupervisionWallClockTimeout` (lines 865-968): 4 tests manually set metadata then assert it was set. They never call `_execute_supervision_async()`.

Real coverage for both behaviors exists in `test_supervision.py`.

**File**: `tests/core/research/workflows/deep_research/test_sanitize_external_content.py`

**Fix**: Remove `TestRawNotesCapping` and `TestSupervisionWallClockTimeout` classes entirely. They provide false confidence and are covered by `test_supervision.py`.

### Item 3.2: Add `DigestPolicy.PROACTIVE` Test Coverage

**Problem**: `test_proactive_digest.py` was deleted (385 lines) with no replacement. `DigestPolicy.PROACTIVE` in `config.py:41` has zero test coverage.

**Files**: `tests/core/research/` (new test file or added to existing)

**Fix**: Add tests for proactive digest eligibility and execution flow — at minimum: eligibility check, execution after gathering phase, and skip-already-digested behavior.

### Item 3.3: Move Misplaced Test Classes

**Problem**: `TestSupervisionWallClockTimeout` and `TestRawNotesCapping` in `test_sanitize_external_content.py` have nothing to do with content sanitization. (Subsumed by 3.1 — if classes are removed rather than moved, this is a no-op.)

---

## FIX-4: Cleanup & Consistency

> **Scope**: ~20-40 LOC | **Risk**: Very Low | **Priority**: LOW

### Item 4.1: Fix Dead Code in `ris.py`

**Problem**: `ris.py:36-38` has identical return statements in both branches of `if venue` — the inner `if` is dead code.

**File**: `src/foundry_mcp/core/research/export/ris.py`

**Fix**: Remove the dead `if venue` branch. Both paths return `"JOUR"`.

### Item 4.2: Add `$` and `~` to BibTeX Special Character Escaping

**Problem**: `bibtex.py:17-24` escapes `& % # _ { }` but omits `$` (enters math mode) and `~` (non-breaking space).

**File**: `src/foundry_mcp/core/research/export/bibtex.py`

**Fix**: Add `ord("$"): r"\$"` and `ord("~"): r"\textasciitilde{}"` to `_BIBTEX_SPECIAL`.

### Item 4.3: Replace `assert` with Proper Type Check in `evaluator.py`

**Problem**: `evaluator.py:316` uses `assert isinstance(call_result, LLMCallResult)` for runtime flow control. Assertions are stripped with `python -O`.

**File**: `src/foundry_mcp/core/research/evaluation/evaluator.py`

**Fix**: Replace with `if not isinstance(call_result, LLMCallResult): raise TypeError(...)`.

### Item 4.4: Remove Empty `TYPE_CHECKING` Block in `evaluator.py`

**Problem**: `evaluator.py:31-32` has `if TYPE_CHECKING: pass` — does nothing.

**File**: `src/foundry_mcp/core/research/evaluation/evaluator.py`

**Fix**: Remove the block.

### Item 4.5: Fix `asyncio.get_event_loop()` Deprecation in `pdf_extractor.py`

**Problem**: `pdf_extractor.py:485` uses `asyncio.get_event_loop()`, deprecated in Python 3.10+.

**File**: `src/foundry_mcp/core/research/pdf_extractor.py`

**Fix**: Replace with `asyncio.get_running_loop()`.

### Item 4.6: Consistent `extra="forbid"` on New Output Models

**Problem**: `ResearchLandscape`, `StudyComparison`, and `StructuredResearchOutput` do not set `extra = "forbid"`, while all other new models do. Typos in field names will be silently accepted.

**File**: `src/foundry_mcp/core/research/models/deep_research.py`

**Fix**: Add `model_config = {"extra": "forbid"}` to `ResearchLandscape`, `StudyComparison`, and `StructuredResearchOutput`.

---

## Deferred / Tracked as Tech Debt

These items are real but lower priority and higher effort. Track for future work.

| ID | Issue | Rationale for deferral |
|----|-------|----------------------|
| TD-1 | SSRF DNS rebinding TOCTOU in `pdf_extractor.py` | Complex fix (custom transport), low probability in practice since PDF fetch is server-side |
| TD-2 | HTTP session-per-request in all providers | Inherited pattern, needs design for session lifecycle and cleanup |
| TD-3 | Sequential methodology assessment | Performance only, no correctness impact |
| TD-4 | `topic_research.py` at 2,980 lines | Should split but risky mid-branch |
| TD-5 | `test_supervision.py` at 5,311 lines | Should split into 4-6 files |
| TD-6 | Config explosion (90+ flat fields on `ResearchConfig`) | Needs sub-config migration, large refactor |
| TD-7 | `ResearchLandscape` mixed typed/untyped list fields | Cosmetic inconsistency |
| TD-8 | Source eviction doesn't consider quality | Enhancement, not a bug |
| TD-9 | `DigestPolicy.PROACTIVE` code path untested beyond FIX-3.2 | Broader integration testing needed |

---

## Estimated Scope

| Phase | Fix LOC | Test LOC | Focus |
|-------|---------|----------|-------|
| 0. Security | ~40-60 | ~20-30 | SSRF, URL encoding, filter injection, content types |
| 1. Correctness | ~60-80 | ~30-50 | Metadata overwrite, type safety, exports, bounds |
| 2. Provider robustness | ~50-70 | ~20-30 | 404 detection, DRY, docstrings, UX |
| 3. Test quality | ~30-50 | ~80-120 | Remove tautological tests, add digest coverage |
| 4. Cleanup | ~20-40 | — | Dead code, deprecations, consistency |
| **Total** | **~200-300** | **~150-230** | |
