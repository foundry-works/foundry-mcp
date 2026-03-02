# Post-Review Fix Plan v3 — Checklist

> Track implementation progress for [PLANFIX.md](PLANFIX.md).
> Mark items `[x]` as completed.

---

## FIX-0: Security Hardening

### Item 0.1: Fail Closed on DNS Resolution Failure in `pdf_extractor.py`
> **File**: `src/foundry_mcp/core/research/pdf_extractor.py`

- [x] Change `except socket.gaierror` handler (line ~289) to raise `SSRFError` instead of logging and continuing
- [x] Match the pattern from `_injection_protection.py:131-132` which correctly returns `False`
- [x] Log at WARNING level before raising (for operator visibility)

#### Item 0.1 Validation

- [x] DNS resolution failure for a hostname raises `SSRFError`
- [x] Valid hostnames still resolve and proceed normally
- [x] Add unit test: `validate_url_for_ssrf` with unresolvable hostname raises `SSRFError`
- [x] Existing PDF extraction tests pass unchanged

---

### Item 0.2: URL-Encode `paper_id` in Semantic Scholar URL Paths
> **File**: `src/foundry_mcp/core/research/providers/semantic_scholar.py`

- [x] Import `urllib.parse.quote` at module level
- [x] Apply `quote(paper_id, safe='')` in `get_paper()` URL construction (line ~403)
- [x] Apply `quote(paper_id, safe='')` in `get_citations()` URL construction (line ~432)
- [x] Apply `quote(paper_id, safe='')` in `get_recommendations()` URL construction — N/A: uses POST body, not URL path
- [x] Add comment explaining why URL encoding is needed (unlike the prefix formats like `DOI:` which httpx handles)

#### Item 0.2 Validation

- [x] `paper_id` with path traversal chars `"../admin"` is safely encoded in URL
- [x] Standard paper IDs still resolve correctly
- [x] Add unit test: paper_id with special characters is URL-encoded in request
- [x] Existing Semantic Scholar tests pass unchanged

---

### Item 0.3: Validate `work_id` in OpenAlex Filter String Construction
> **File**: `src/foundry_mcp/core/research/providers/openalex.py`

- [x] Add `_OPENALEX_WORK_ID_RE = re.compile(r"^(W\d+|https://openalex\.org/W\d+)$")` constant
- [x] Add `_OPENALEX_TOPIC_ID_RE = re.compile(r"^T\d+$")` constant
- [x] Validate `work_id` in `get_citations()` (line ~312) before filter interpolation
- [x] Validate `topic_id` in `search_by_topic()` (line ~438) before filter interpolation
- [x] Raise `ValueError` with descriptive message on mismatch

#### Item 0.3 Validation

- [x] Valid `work_id` `"W2741809807"` passes validation
- [x] Valid `work_id` `"https://openalex.org/W2741809807"` passes validation
- [x] Malicious `work_id` `"W123,publication_year:2024"` raises `ValueError`
- [x] Valid `topic_id` `"T12345"` passes validation
- [x] Malicious `topic_id` `"T123,type:dataset"` raises `ValueError`
- [x] Existing OpenAlex tests pass unchanged

---

### Item 0.4: Remove `application/octet-stream` from Valid PDF Content Types
> **File**: `src/foundry_mcp/core/research/pdf_extractor.py`

- [x] Remove `"application/octet-stream"` from `VALID_PDF_CONTENT_TYPES` frozenset (line ~167)
- [x] Add WARNING log when `application/octet-stream` content type is encountered (before magic byte check)
- [x] Ensure magic byte check (`%PDF`) still handles ambiguous content types downstream

#### Item 0.4 Validation

- [x] PDF served with `application/pdf` content type is accepted
- [x] PDF served with `application/octet-stream` is rejected at content-type gate
- [x] Magic byte check still catches real PDFs served with unexpected content types (if applicable)
- [x] Existing PDF extraction tests pass unchanged

---

## FIX-1: Correctness Fixes

### Item 1.1: Fix Evaluator Metadata Overwrite
> **File**: `src/foundry_mcp/core/research/evaluation/evaluator.py`

- [ ] Change `eval_result.metadata = {` (line ~337) to `eval_result.metadata.update({`
- [ ] Change the closing `}` to `})`
- [ ] Verify `imputed_count` and `warnings` keys from `_parse_evaluation_response` are preserved

#### Item 1.1 Validation

- [ ] `eval_result.metadata` contains both LLM call metadata (`provider_id`, `model_used`, `duration_ms`) and parse metadata (`imputed_count`, `warnings`)
- [ ] Add unit test: evaluation result metadata contains both parse-time and call-time keys
- [ ] Existing evaluator tests pass unchanged

---

### Item 1.2: Change `content_basis` to `Literal` Type
> **File**: `src/foundry_mcp/core/research/models/sources.py`

- [ ] Add `Literal` to `typing` imports (if not already present)
- [ ] Change `content_basis: str = Field(default="abstract", ...)` to `content_basis: Literal["abstract", "full_text"] = Field(default="abstract", ...)`
- [ ] Verify existing `model_validator` at line ~585 still works with Literal type

#### Item 1.2 Validation

- [ ] `MethodologyAssessment(content_basis="abstract")` succeeds
- [ ] `MethodologyAssessment(content_basis="full_text")` succeeds
- [ ] `MethodologyAssessment(content_basis="Abstract")` raises `ValidationError`
- [ ] `MethodologyAssessment(content_basis="typo")` raises `ValidationError`
- [ ] Confidence still forced to `"low"` for `content_basis="abstract"`
- [ ] Existing methodology assessment tests pass unchanged

---

### Item 1.3: Add Missing Model Exports to `__init__.py`
> **File**: `src/foundry_mcp/core/research/models/__init__.py`

- [ ] Add import for `ProvenanceLog` from `deep_research`
- [ ] Add import for `ProvenanceEntry` from `deep_research`
- [ ] Add import for `CitationNetwork` from `deep_research`
- [ ] Add import for `CitationNode` from `deep_research`
- [ ] Add import for `CitationEdge` from `deep_research`
- [ ] Add import for `ResearchLandscape` from `deep_research`
- [ ] Add import for `StudyComparison` from `deep_research`
- [ ] Add import for `ResearchThread` from `deep_research`
- [ ] Add import for `MethodologyAssessment` from `sources`
- [ ] Add import for `StudyDesign` from `sources`
- [ ] Add all to `__all__`

#### Item 1.3 Validation

- [ ] `from foundry_mcp.core.research.models import ProvenanceLog` works
- [ ] `from foundry_mcp.core.research.models import CitationNetwork` works
- [ ] `from foundry_mcp.core.research.models import MethodologyAssessment` works
- [ ] `from foundry_mcp.core.research.models import StudyDesign` works
- [ ] Existing import tests pass unchanged

---

### Item 1.4: Add `MAX_ABSTRACT_POSITIONS` Cap in OpenAlex Provider
> **File**: `src/foundry_mcp/core/research/providers/openalex.py`

- [ ] Add `_MAX_ABSTRACT_POSITIONS = 100_000` module-level constant
- [ ] In `_reconstruct_abstract()` (line ~130), after computing `max_pos`, check `if max_pos > _MAX_ABSTRACT_POSITIONS: return None`
- [ ] Log at WARNING level when cap is hit

#### Item 1.4 Validation

- [ ] Normal inverted index (positions 0-500) reconstructs correctly
- [ ] Malicious inverted index with position `999999999` returns `None`
- [ ] Add unit test: abstract with oversized position returns None
- [ ] Existing OpenAlex tests pass unchanged

---

### Item 1.5: Add Upper Bound Validation on Citation Network Parameters
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [ ] Clamp `max_references_per_paper` to `max(1, min(value, 100))` before passing to builder
- [ ] Clamp `max_citations_per_paper` to `max(1, min(value, 100))` before passing to builder
- [ ] Log at DEBUG level if clamping was applied

#### Item 1.5 Validation

- [ ] `max_references_per_paper=999999` clamped to 100
- [ ] `max_references_per_paper=0` clamped to 1
- [ ] `max_references_per_paper=20` passes through unchanged
- [ ] Existing citation network tests pass unchanged

---

### Item 1.6: Sanitize `research_id` Before Filesystem Path Use
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/infrastructure.py`

- [ ] In `_crash_handler()` (line ~150), sanitize `research_id` before path construction
- [ ] Use `Path(research_id).name` to strip directory components, or validate against `^[a-zA-Z0-9_-]+$`
- [ ] Apply same sanitization in any other filesystem path construction using `research_id`

#### Item 1.6 Validation

- [ ] Normal `research_id` `"deepres-abc123"` produces expected crash path
- [ ] Traversal `research_id` `"../../etc/passwd"` is sanitized to safe filename
- [ ] Add unit test: path traversal in research_id does not escape target directory

---

## FIX-2: Provider Robustness

### Item 2.1: Replace `"404" in str(e)` with Structured Status Code Check
> **Files**: `src/foundry_mcp/core/research/providers/openalex.py`, `crossref.py`, `semantic_scholar.py`, `shared.py`

- [ ] Add `status_code: Optional[int] = None` field to `SearchProviderError` in `core/errors/provider.py`
- [ ] In `_execute_request()` (shared.py), populate `status_code` from the HTTP response when raising `SearchProviderError`
- [ ] Replace `if "404" in str(e):` with `if getattr(e, "status_code", None) == 404:` in `openalex.py` (lines ~295, ~337, ~372)
- [ ] Replace `if "404" in str(e):` with `if getattr(e, "status_code", None) == 404:` in `crossref.py` (line ~218)
- [ ] Replace `if "404" in str(e):` with `if getattr(e, "status_code", None) == 404:` in `semantic_scholar.py` (line ~410)

#### Item 2.1 Validation

- [ ] 404 HTTP response produces `SearchProviderError` with `status_code=404`
- [ ] `get_work()` with 404 returns `None` (existing behavior preserved)
- [ ] Error message containing "404" for non-404 status does NOT trigger the 404 path
- [ ] Add unit test: error with "404" in message but status_code=500 is NOT treated as not-found
- [ ] Existing provider tests pass unchanged

---

### Item 2.2: Extract Duplicated `classify_error` Logic
> **Files**: `src/foundry_mcp/core/research/providers/shared.py`, `crossref.py`, `base.py`

- [ ] Add `classify_with_registry(error, classifiers, provider_name) -> ErrorClassification` function to `shared.py`
- [ ] Refactor `SearchProvider.classify_error` in `base.py` to delegate to `classify_with_registry`
- [ ] Refactor `CrossrefProvider.classify_error` in `crossref.py` to delegate to `classify_with_registry`
- [ ] Remove the duplicated logic from `crossref.py`

#### Item 2.2 Validation

- [ ] `CrossrefProvider.classify_error` produces same results as before
- [ ] `SearchProvider.classify_error` produces same results as before
- [ ] Existing provider error classification tests pass unchanged

---

### Item 2.3: Fix Stale Docstrings Referencing Removed Cost-Tier Defaults
> **Files**: `src/foundry_mcp/config/research.py`, `src/foundry_mcp/config/research_sub_configs.py`

- [ ] Update `ResearchConfig` class docstring (lines ~29-31) to remove `gemini-2.5-flash` reference
- [ ] Update `ModelRoleConfig` docstring (`research_sub_configs.py:79-83`) to remove cost-tier default reference
- [ ] Replace with accurate description of current behavior (explicit tier configuration required)

#### Item 2.3 Validation

- [ ] No references to `gemini-2.5-flash` remain in config docstrings
- [ ] Docstrings accurately describe current model routing behavior

---

### Item 2.4: Add Warning to Response on Failed Report Save
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [ ] In `_handle_deep_research_report` (line ~271), capture exception message on save failure
- [ ] Add `"save_warning": str(exc)` to response data dict when save fails
- [ ] Ensure the response still returns successfully (non-blocking save)

#### Item 2.4 Validation

- [ ] Successful save: no `save_warning` key in response
- [ ] Failed save: `save_warning` key present with error description
- [ ] Report data still returned regardless of save success/failure

---

### Item 2.5: Add `deep-research-provenance` to MCP Tool Docstring
> **File**: `src/foundry_mcp/tools/unified/research_handlers/__init__.py`

- [ ] Add `deep-research-provenance` to the action list in the `research()` tool description (lines ~267-273)

#### Item 2.5 Validation

- [ ] `deep-research-provenance` appears in MCP tool description
- [ ] LLM agent can discover the provenance action from tool metadata

---

## FIX-3: Test Quality

### Item 3.1: Remove Tautological Tests
> **File**: `tests/core/research/workflows/deep_research/test_sanitize_external_content.py`

- [ ] Remove `TestRawNotesCapping` class (lines ~712-820, 3 tests)
- [ ] Remove `TestSupervisionWallClockTimeout` class (lines ~865-968, 4 tests)
- [ ] Remove `_StubSupervision` helper class (lines ~844-864) if no other tests use it
- [ ] Verify real coverage exists in `test_supervision.py` for both behaviors

#### Item 3.1 Validation

- [ ] 7 tautological tests removed
- [ ] `test_supervision.py` `TestSupervisionWallClockTimeout` still covers wall-clock timeout behavior
- [ ] `test_supervision.py` still covers raw_notes trimming behavior
- [ ] All remaining tests pass
- [ ] Net test count decrease is exactly 7 (no collateral removal)

---

### Item 3.2: Add `DigestPolicy.PROACTIVE` Test Coverage
> **File**: New file `tests/core/research/test_proactive_digest.py` or added to existing digest test file

- [ ] Add test: `DigestPolicy.PROACTIVE` eligibility check (sources with content above threshold)
- [ ] Add test: proactive digest execution produces digested content
- [ ] Add test: already-digested sources are skipped
- [ ] Add test: `DigestPolicy.PROACTIVE` with zero eligible sources is a no-op

#### Item 3.2 Validation

- [ ] At least 4 new tests covering `PROACTIVE` digest policy
- [ ] All new tests pass
- [ ] Existing digest tests pass unchanged

---

## FIX-4: Cleanup & Consistency

### Item 4.1: Fix Dead Code in `ris.py`
> **File**: `src/foundry_mcp/core/research/export/ris.py`

- [ ] Remove the redundant inner `if venue:` check at lines ~36-38
- [ ] Both branches return `"JOUR"` — collapse to single `return "JOUR"`

#### Item 4.1 Validation

- [ ] Academic sources with venue produce `TY  - JOUR`
- [ ] Academic sources without venue produce `TY  - JOUR`
- [ ] Existing RIS tests pass unchanged

---

### Item 4.2: Add `$` and `~` to BibTeX Special Character Escaping
> **File**: `src/foundry_mcp/core/research/export/bibtex.py`

- [ ] Add `ord("$"): r"\$"` to `_BIBTEX_SPECIAL` translation table
- [ ] Add `ord("~"): r"\textasciitilde{}"` to `_BIBTEX_SPECIAL` translation table

#### Item 4.2 Validation

- [ ] Title containing `$` is escaped to `\$` in BibTeX output
- [ ] Title containing `~` is escaped to `\textasciitilde{}` in BibTeX output
- [ ] Existing BibTeX tests pass unchanged

---

### Item 4.3: Replace `assert` with Type Check in `evaluator.py`
> **File**: `src/foundry_mcp/core/research/evaluation/evaluator.py`

- [ ] Replace `assert isinstance(call_result, LLMCallResult)` (line ~316) with `if not isinstance(call_result, LLMCallResult): raise TypeError(...)`

#### Item 4.3 Validation

- [ ] Invalid call_result type raises `TypeError` (not `AssertionError`)
- [ ] Valid `LLMCallResult` passes through normally
- [ ] Works with `python -O` (optimized mode)

---

### Item 4.4: Remove Empty `TYPE_CHECKING` Block in `evaluator.py`
> **File**: `src/foundry_mcp/core/research/evaluation/evaluator.py`

- [ ] Remove `if TYPE_CHECKING: pass` block (lines ~31-32)

#### Item 4.4 Validation

- [ ] No import errors after removal
- [ ] Existing evaluator tests pass unchanged

---

### Item 4.5: Fix `asyncio.get_event_loop()` Deprecation
> **File**: `src/foundry_mcp/core/research/pdf_extractor.py`

- [ ] Replace `asyncio.get_event_loop()` (line ~485) with `asyncio.get_running_loop()`

#### Item 4.5 Validation

- [ ] No `DeprecationWarning` on Python 3.12+
- [ ] Existing tests pass unchanged

---

### Item 4.6: Add `extra="forbid"` to Remaining New Models
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [ ] Add `model_config = {"extra": "forbid"}` to `ResearchLandscape`
- [ ] Add `model_config = {"extra": "forbid"}` to `StudyComparison`
- [ ] Add `model_config = {"extra": "forbid"}` to `StructuredResearchOutput`

#### Item 4.6 Validation

- [ ] `ResearchLandscape(typo_field="value")` raises `ValidationError`
- [ ] `StudyComparison(typo_field="value")` raises `ValidationError`
- [ ] `StructuredResearchOutput(typo_field="value")` raises `ValidationError`
- [ ] Existing tests pass unchanged (no tests rely on extra fields being silently accepted)

---

## Final Validation

- [ ] All 7,597+ tests pass
- [ ] No new `DeprecationWarning` on Python 3.12+
- [ ] DNS resolution failure in `pdf_extractor.py` fails closed
- [ ] All provider URL paths use URL-encoded identifiers
- [ ] OpenAlex filter values validated against expected patterns
- [ ] `application/octet-stream` no longer accepted as valid PDF content type
- [ ] Evaluator metadata contains both parse-time and call-time keys
- [ ] `content_basis` rejects invalid string values
- [ ] All new models exported from `models/__init__.py`
- [ ] 404 detection uses structured status codes, not string matching
- [ ] 7 tautological tests removed
- [ ] `DigestPolicy.PROACTIVE` has test coverage
- [ ] No references to `gemini-2.5-flash` in config docstrings
- [ ] `deep-research-provenance` discoverable in MCP tool description

---

## Estimated Scope

| Phase | Fix LOC | Test LOC | Focus |
|-------|---------|----------|-------|
| 0. Security | ~40-60 | ~20-30 | SSRF, URL encoding, filter injection, content types |
| 1. Correctness | ~60-80 | ~30-50 | Metadata, types, exports, bounds, path safety |
| 2. Provider robustness | ~50-70 | ~20-30 | 404 detection, DRY, docstrings, UX |
| 3. Test quality | ~30-50 | ~80-120 | Remove tautological tests, add digest coverage |
| 4. Cleanup | ~20-40 | — | Dead code, deprecations, consistency |
| **Total** | **~200-300** | **~150-230** | |
