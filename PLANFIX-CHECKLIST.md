# Post-Review Fix Plan v2 — Checklist

> Track implementation progress for [PLANFIX.md](PLANFIX.md).
> Mark items `[x]` as completed.

---

## FIX-0: Security Fixes

### Item 0.1: Move OpenAlex API Key to Header *(carryover)*
> **File**: `src/foundry_mcp/core/research/providers/openalex.py`

- [x] Replace `params["api_key"] = self._api_key` with `headers["x-api-key"] = self._api_key`
- [x] Remove `api_key` from `params` dict entirely
- [x] Verify `redact_headers()` in `shared.py` covers the `x-api-key` header name
- [x] Update any tests that assert on query params containing `api_key`

#### Item 0.1 Validation

- [x] API key not present in any request URL or query string
- [x] API key present in request headers
- [x] `redact_headers()` redacts the key in debug logs
- [x] All existing OpenAlex tests pass

---

### Item 0.2: Sanitize Assistant Messages in ReAct Prompt *(carryover)*
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] Apply `sanitize_external_content()` to `content` in the `role == "assistant"` branch of `_build_react_user_prompt` (line ~504)
- [x] Verify `sanitize_external_content` is already imported at module level (it is — line 40)

#### Item 0.2 Validation

- [x] Assistant messages containing `<system>` tags are sanitized in the prompt
- [x] Existing topic research tests pass unchanged
- [x] Add unit test: assistant content with injection payload is sanitized

---

### Item 0.3: Sanitize Content in Methodology Assessment Prompts *(carryover)*
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py`

- [x] Import `sanitize_external_content` from `_injection_protection`
- [x] Wrap `source_title` with `sanitize_external_content()` in `_build_extraction_user_prompt` (line ~76)
- [x] Wrap `content` with `sanitize_external_content()` in `_build_extraction_user_prompt` (line ~78)
- [x] Wrap `assessment.effect_size` with `sanitize_external_content()` in `format_methodology_context`
- [x] Wrap `assessment.sample_description` with `sanitize_external_content()` in `format_methodology_context`
- [x] Wrap `assessment.limitations_noted` list items with `sanitize_external_content()` in `format_methodology_context`
- [x] Wrap `assessment.potential_biases` list items with `sanitize_external_content()` in `format_methodology_context`

#### Item 0.3 Validation

- [x] Source title with `<system>` tags is sanitized in extraction prompt
- [x] Source content with injection payload is sanitized in extraction prompt
- [x] Assessment fields with injection payloads are sanitized in synthesis context
- [x] Add unit test: injection payload in source content is stripped

---

### Item 0.4: Sanitize PDF Content Preview at Point of Creation
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] Apply `sanitize_external_content()` to `content_preview` at line ~2225
- [x] Verify `sanitize_external_content` is already imported at module level (it is — line 40)

#### Item 0.4 Validation

- [x] PDF content preview with injection payload is sanitized before storage in message_history
- [x] Existing topic research tests pass unchanged
- [x] Add unit test: PDF content with `<system>` tags is sanitized in tool result

---

## FIX-1: Broken Features

### Item 1.1: Fix Methodology Assessment Handler — All Results UNKNOWN
> **Files**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`, `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py`

- [x] Refactor `_assess_single` to accept an `llm_call_fn` callable parameter instead of requiring full `workflow` + `state`
- [x] Update `assess_sources()` to accept and pass through `llm_call_fn`
- [x] In the handler (`_handle_deep_research_assess`), construct an appropriate `llm_call_fn` using `resolve_provider` + `ProviderRequest` from provider infrastructure
- [x] Pass the `llm_call_fn` through to `assess_sources()`
- [x] Verify the LLM call path is actually exercised (not falling through to UNKNOWN)

#### Item 1.1 Validation

- [x] `deep-research-assess` action on a session with academic sources produces real assessments (not all UNKNOWN)
- [x] Assessments include correct `study_design` values from LLM extraction
- [x] Fallback to UNKNOWN still works when LLM call fails
- [x] `confidence` forced to `"low"` for abstract-only content
- [ ] Add integration test: assess action with mocked LLM returns valid assessments

---

### Item 1.2: Wire Export Parameters Through MCP Tool Signature
> **File**: `src/foundry_mcp/tools/unified/research_handlers/__init__.py`, `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [x] Add `export_format: Optional[str] = None` parameter to `research()` function signature
- [x] Add `academic_only: Optional[bool] = None` parameter to `research()` function signature
- [x] Rename `format` parameter to `export_format` in `_handle_deep_research_export` (fixes Python built-in shadow)
- [x] Update dispatch to pass `export_format` and `academic_only` through to handler
- [x] Add validation: reject unknown `export_format` values (only `"bibtex"` and `"ris"` accepted)

#### Item 1.2 Validation

- [x] LLM client can see `export_format` and `academic_only` in the tool schema
- [x] `export_format="ris"` produces RIS output
- [x] `export_format="bibtex"` produces BibTeX output
- [x] `export_format="csv"` returns a validation error (not silent bibtex fallback)
- [x] Default behavior unchanged when parameters not provided
- [x] Existing export tests pass (updated parameter names)

---

## FIX-2: Correctness Fixes

### Item 2.1: Fix Context Window Retry — Same Truncation Produces Same Failure
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] Add `budget_fraction: float = 1.0` parameter to `_truncate_researcher_history`
- [x] Apply `budget_fraction` multiplier to `effective_budget` calculation inside the function
- [x] On `ContextWindowError` retry (line ~927), call with `budget_fraction=0.5`
- [x] On generic context-window-sniff retry (line ~990), call with `budget_fraction=0.5`
- [x] Log the aggressive truncation at DEBUG level

#### Item 2.1 Validation

- [x] Retry with `budget_fraction=0.5` produces a shorter prompt than initial attempt
- [x] Normal (non-retry) truncation behavior unchanged (`budget_fraction=1.0` default)
- [x] Existing context window tests pass unchanged (103 passed)
- [ ] Add unit test: retry truncation is strictly shorter than initial truncation

---

### Item 2.2: Add Cancellation Check in Synthesis Retry Loop
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] Add `self._check_cancellation(state)` at the top of the `for outer_attempt in range(...)` loop body (line ~628)

#### Item 2.2 Validation

- [x] Cancelled research exits synthesis retry loop promptly
- [x] Non-cancelled research continues retry loop normally
- [x] Existing synthesis tests pass unchanged (64 passed)

---

### Item 2.3: Fix Premature State Save — Provenance Event Lost on Crash
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] Move `synthesis_completed` provenance append (lines ~920-935) to before the state save (line ~895)
- [x] Reorder: build landscape → build structured output → append provenance → save state → finalize_phase
- [x] Verify `finalize_phase` still runs after the save

#### Item 2.3 Validation

- [x] `synthesis_completed` provenance event is present in saved state
- [x] Landscape and structured output are present in saved state
- [x] Existing synthesis tests pass unchanged (64 passed)

---

### Item 2.4: Wrap Profile Resolution in try/except for ValidationError
> **File**: `src/foundry_mcp/config/research.py`

- [x] Import `ValidationError` from pydantic (lazy import inside method)
- [x] Wrap `ResearchProfile(**self.deep_research_profiles[profile_name])` in try/except
- [x] Catch `(TypeError, ValidationError)` and raise `ValueError` with descriptive message
- [x] Wrap `profile.model_copy(update=profile_overrides)` in try/except
- [x] Catch `(TypeError, ValidationError)` and raise `ValueError` with descriptive message

#### Item 2.4 Validation

- [x] Malformed profile config (wrong types, bad field names) produces clean `ValueError`
- [x] Invalid `profile_overrides` produces clean `ValueError`
- [x] Valid profile construction still works unchanged
- [x] Handler's `except ValueError` catches both cases
- [ ] Add unit test: invalid profile config produces ValueError with descriptive message

---

### Item 2.5: Validate Academic Coverage Weight Keys
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `_VALID_ACADEMIC_WEIGHT_KEYS = {"source_adequacy", "domain_diversity", "query_completion_rate", "source_influence"}` constant
- [x] Add validation in `__post_init__` for `deep_research_academic_coverage_weights` when not None
- [x] Strip unknown keys (with warning log) matching the pattern for general weights
- [x] Validate values are numeric and >= 0

#### Item 2.5 Validation

- [x] Academic weights with valid keys pass validation
- [x] Academic weights with unknown key `"typo"` strips the key and logs warning
- [x] Academic weights with `source_influence` key passes validation (unlike general weights)
- [x] General weights with `source_influence` key still stripped (no regression)
- [x] Existing config tests pass unchanged (57 passed)

---

### Item 2.6: Fix `_save_report_markdown` Using `Path.cwd()`
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] Accept `output_dir` parameter (default None) in `_save_report_markdown`
- [x] When None, derive path from `self.memory` workspace directory if available
- [x] Fall back to `Path.cwd()` only as last resort, with debug log
- [x] Wrap entire save in try/except to ensure failures never crash synthesis (already existed)

#### Item 2.6 Validation

- [x] Report saved to workspace directory when available
- [x] Falls back to cwd when no workspace configured
- [x] Failed save logs warning but does not crash synthesis
- [x] Existing tests pass unchanged (7,587 passed)

---

## FIX-3: Input Validation & Defense-in-Depth

### Item 3.1: Validate `paper_id` from LLM Tool Calls
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] Add `_PAPER_ID_RE = re.compile(r"^[a-zA-Z0-9._/:\-]{1,256}$")` constant
- [x] Add `_validate_paper_id(paper_id: str) -> str | None` helper (returns error message or None)
- [x] Call `_validate_paper_id` in `_handle_citation_search_tool` before provider call (line ~2280)
- [x] Call `_validate_paper_id` in `_handle_related_papers_tool` before provider call (line ~2362)
- [x] Return validation error message to researcher LLM on failure

#### Item 3.1 Validation

- [x] Valid DOI `"10.1038/nature12373"` passes validation
- [x] Valid S2 ID `"649def34f8be52c8b66281af98ae884c09aef38b"` passes validation
- [x] Valid ArXiv ID `"2301.07041"` passes validation
- [x] Empty string is rejected
- [x] String > 256 chars is rejected
- [x] String with shell metacharacters is rejected
- [x] Existing citation search tests pass unchanged

---

### Item 3.2: URL-Encode DOI Values in Provider API Paths
> **Files**: `src/foundry_mcp/core/research/providers/openalex.py`, `src/foundry_mcp/core/research/providers/crossref.py`

- [x] Import `urllib.parse.quote` in both files
- [x] In `openalex.py:258`: apply `quote(work_id, safe="")` before f-string interpolation
- [x] In `crossref.py:214`: apply `quote(doi, safe="")` before f-string interpolation
- [x] Also applied to `get_references` (line 300) and `get_related` (line 335) path interpolations in openalex.py

#### Item 3.2 Validation

- [x] DOI with special chars `"10.1000/foo_bar#baz"` is URL-encoded in request path
- [x] Standard DOI `"10.1038/nature12373"` still resolves correctly (encoding is transparent)
- [x] Existing provider tests pass (updated assertions to expect encoded form)
- [ ] Add unit test: DOI with path traversal chars is safely encoded

---

### Item 3.3: Sanitize OpenAlex Filter Values
> **File**: `src/foundry_mcp/core/research/providers/openalex.py`

- [x] Add `_ALLOWED_FILTER_KEYS` frozenset with known OpenAlex filter field names
- [x] Add `_build_filter_string(filters: dict) -> str` helper
- [x] Validate filter keys against allowlist (warn and skip unknown keys)
- [x] Strip commas, pipes, and colons from filter values (operator characters)
- [x] Replace inline filter construction (lines ~227-232) with helper call

#### Item 3.3 Validation

- [x] Known filter key passes through: `{"publication_year": "2024"}` → `"publication_year:2024"`
- [x] Unknown filter key is skipped with warning: `{"evil_key": "value"}` → dropped
- [x] Malicious filter value stripped: `{"publication_year": "2024,type:dataset"}` → `"publication_year:2024typedataset"`
- [x] Boolean filter values still handled correctly: `{"is_oa": True}` → `"is_oa:true"`
- [x] Existing provider tests pass unchanged

---

## FIX-4: Robustness & Quality

### Item 4.1: Fix Duplicate `_classify_query_type` / Duplicate Provenance
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [ ] Extract pure `_classify_query_type` function (no provenance side effect)
- [ ] Call it once early in `_execute_synthesis_async` and store result
- [ ] Pass stored `query_type` to `_build_synthesis_system_prompt`, `_build_synthesis_tail`, and `_finalize_synthesis_report`
- [ ] Log `synthesis_query_type` provenance event once explicitly after classification
- [ ] Remove provenance logging from `_build_synthesis_system_prompt`

#### Item 4.1 Validation

- [ ] Only one `synthesis_query_type` provenance event per synthesis run
- [ ] Query type classification result consistent across all consumers
- [ ] Existing synthesis tests pass (update provenance count assertions if needed)

---

### Item 4.2: Use `Literal` Type for `MethodologyAssessment.confidence`
> **File**: `src/foundry_mcp/core/research/models/sources.py`

- [ ] Change `confidence: str = "low"` to `confidence: Literal["high", "medium", "low"] = "low"`
- [ ] Add `from typing import Literal` import (if not already present)

#### Item 4.2 Validation

- [ ] `MethodologyAssessment(confidence="invalid")` raises `ValidationError`
- [ ] `MethodologyAssessment(confidence="high")` succeeds
- [ ] Existing tests pass unchanged

---

### Item 4.3: Request `confidence` Field in Methodology LLM Prompt
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py`

- [ ] Add `"confidence"` field to the JSON schema in `METHODOLOGY_EXTRACTION_SYSTEM_PROMPT` (line ~43-58)
- [ ] Add description: `"Your confidence in the assessment: 'high' (full text with clear methods section), 'medium' (substantial content), 'low' (abstract only or limited content)"`

#### Item 4.3 Validation

- [ ] LLM prompt includes `confidence` in the requested JSON schema
- [ ] `_parse_llm_response` correctly reads LLM-provided confidence
- [ ] Content-basis override still forces `"low"` for abstract-only (FIX from v1 review)
- [ ] Existing tests pass unchanged

---

### Item 4.4: Add Timeout to Citation Network `asyncio.gather`
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/citation_network.py`

- [ ] Wrap `asyncio.gather(*tasks, return_exceptions=True)` at line ~165 with `asyncio.wait_for(..., timeout=timeout or 90.0)`
- [ ] Catch `asyncio.TimeoutError` and log warning with partial results

#### Item 4.4 Validation

- [ ] Citation network build respects timeout
- [ ] Partial results returned on timeout (nodes/edges collected before timeout)
- [ ] Existing citation network tests pass unchanged

---

## FIX-5: Test Improvements

### Item 5.1: Fix `study_design="rct"` Test Bug *(carryover)*
> **File**: `tests/core/research/test_methodology_assessment.py`

- [ ] Change `_make_llm_json_response(study_design="rct")` to `_make_llm_json_response(study_design="randomized_controlled_trial")` (line ~805)
- [ ] Verify the test now properly validates the success path for the first source

#### Item 5.1 Validation

- [ ] First source produces a valid `randomized_controlled_trial` assessment (not UNKNOWN fallback)
- [ ] Second source still correctly triggers the LLM failure fallback path

---

### Item 5.2: Add PDF Extraction HTTP Tests *(carryover)*
> **File**: `tests/core/research/test_pdf_analysis.py`

- [ ] Add test: `test_extract_from_url_success` with mocked HTTP response returning PDF bytes
- [ ] Add test: `test_extract_from_url_timeout` with mocked timeout
- [ ] Add test: `test_extract_from_url_malformed_pdf` with invalid PDF bytes
- [ ] Add test: `test_extract_from_url_ssrf_blocked` with private IP URL

#### Item 5.2 Validation

- [ ] All new tests pass
- [ ] Existing tests pass unchanged

---

### Item 5.3: Fix RIS Page Range Spec Compliance *(carryover)*
> **File**: `src/foundry_mcp/core/research/export/ris.py`

- [ ] Split page ranges into `SP` and `EP` tags
- [ ] Update existing RIS tests to expect split tags

#### Item 5.3 Validation

- [ ] Page range `"123-456"` produces `SP  - 123` and `EP  - 456`
- [ ] Single page `"42"` produces `SP  - 42` only
- [ ] No page value produces no SP/EP tags
- [ ] Existing tests pass (with updated assertions)

---

### Item 5.4: Add Methodology Assessment Integration Test
> **File**: `tests/core/research/test_methodology_assessment.py` or `tests/integration/`

- [ ] Add test that mocks LLM at `execute_llm_call` level (not at workflow level)
- [ ] Verify handler path produces real assessments (not UNKNOWN)
- [ ] Verify `confidence` override for abstract-only content

#### Item 5.4 Validation

- [ ] Test covers the handler → assessor → LLM call → parse → return path
- [ ] At least one assessment has `study_design != "unknown"`
- [ ] Abstract-only source has `confidence == "low"`

---

## Final Validation

- [ ] All 7,582+ tests pass
- [ ] No new warnings on Python 3.12+
- [ ] No API keys in URL query strings
- [ ] All external content sanitized before LLM prompt interpolation
- [ ] All MCP tool parameters visible in tool schema
- [ ] Methodology assessment produces real results (not all UNKNOWN)
- [ ] Context window retry uses more aggressive truncation
- [ ] Synthesis retry loop checks cancellation
- [ ] Profile resolution errors produce clean error messages
- [ ] Academic coverage weights validated
- [ ] paper_id validated before API calls
- [ ] DOI values URL-encoded in provider paths
- [ ] OpenAlex filter values sanitized
- [ ] RIS export produces spec-compliant page ranges
- [ ] Legacy sessions (pre-PLAN-1) can continue without errors
