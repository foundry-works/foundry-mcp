# Post-Review Fix Plan — Checklist

> Track implementation progress for [PLANFIX.md](PLANFIX.md).
> Mark items `[x]` as completed.

---

## FIX-0: Security Fixes

### Item 0.1: Move OpenAlex API Key to Header
> **File**: `src/foundry_mcp/core/research/providers/openalex.py`

- [ ] Replace `params["api_key"] = self._api_key` with `headers["x-api-key"] = self._api_key`
- [ ] Remove `api_key` from `params` dict entirely
- [ ] Verify `redact_headers()` in `shared.py` covers the `x-api-key` header name
- [ ] Update any tests that assert on query params containing `api_key`

#### Item 0.1 Validation

- [ ] API key not present in any request URL or query string
- [ ] API key present in request headers
- [ ] `redact_headers()` redacts the key in debug logs
- [ ] All existing OpenAlex tests pass

---

### Item 0.2: Sanitize Assistant Messages in ReAct Prompt
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [ ] Apply `sanitize_external_content()` to `content` in the `role == "assistant"` branch of `_build_react_user_prompt` (line ~504)
- [ ] Import `sanitize_external_content` if not already imported at module level

#### Item 0.2 Validation

- [ ] Assistant messages containing `<system>` tags are sanitized in the prompt
- [ ] Existing topic research tests pass unchanged
- [ ] Add unit test: assistant content with injection payload is sanitized

---

### Item 0.3: Sanitize Content in Methodology Assessment Prompts
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py`

- [ ] Import `sanitize_external_content` from `_injection_protection`
- [ ] Wrap `source_title` with `sanitize_external_content()` in `_build_extraction_user_prompt`
- [ ] Wrap `content` with `sanitize_external_content()` in `_build_extraction_user_prompt`
- [ ] Wrap `assessment.effect_size` with `sanitize_external_content()` in `format_methodology_context`
- [ ] Wrap `assessment.sample_description` with `sanitize_external_content()` in `format_methodology_context`
- [ ] Wrap `assessment.limitations_noted` list items with `sanitize_external_content()` in `format_methodology_context`
- [ ] Wrap `assessment.potential_biases` list items with `sanitize_external_content()` in `format_methodology_context`

#### Item 0.3 Validation

- [ ] Source title with `<system>` tags is sanitized in extraction prompt
- [ ] Source content with injection payload is sanitized in extraction prompt
- [ ] Assessment fields with injection payloads are sanitized in synthesis context
- [ ] Add unit test: injection payload in source content is stripped

---

## FIX-1: Integration Blockers

### Item 1.1: Wire Methodology Assessment as User-Triggered Action
> **Files**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`, `src/foundry_mcp/tools/unified/research_handlers/__init__.py`

- [x] Add `_handle_deep_research_assess()` handler function
  - [x] Load completed research state by `research_id`
  - [x] Validate session is completed (or has sources)
  - [x] Filter to academic sources with content > min_content_length
  - [x] Skip if fewer than 2 eligible sources
  - [x] Instantiate `MethodologyAssessor` and call `assess_sources()`
  - [x] Save assessments to `state.extensions.methodology_assessments`
  - [x] Persist updated state
  - [x] Return assessments in success response
- [x] Register `"deep-research-assess"` action in `ACTION_REGISTRY` in `__init__.py`
- [x] Add `ActionDefinition` with appropriate summary and validation schema

#### Item 1.1 Validation

- [x] `deep-research-assess` action callable on completed research session
- [x] Returns methodology assessments for eligible academic sources
- [x] Handles missing/incomplete session gracefully
- [x] Assessments persisted and visible in subsequent `deep-research-report` calls
- [ ] Add integration test: assess action on session with academic sources

---

### Item 1.2: Add Missing Config Fields to `from_toml_dict()`
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `deep_research_pdf_max_pages=data.get("deep_research_pdf_max_pages", 50)` to `cls()` call
- [x] Add `deep_research_pdf_priority_sections=data.get("deep_research_pdf_priority_sections", ["methods", "results", "discussion"])` to `cls()` call
- [x] Add `deep_research_citation_network_max_refs_per_paper=data.get("deep_research_citation_network_max_refs_per_paper", 20)` to `cls()` call
- [x] Add `deep_research_citation_network_max_cites_per_paper=data.get("deep_research_citation_network_max_cites_per_paper", 20)` to `cls()` call
- [x] Add `deep_research_methodology_assessment_provider=data.get("deep_research_methodology_assessment_provider", None)` to `cls()` call
- [x] Add `deep_research_methodology_assessment_timeout=data.get("deep_research_methodology_assessment_timeout", 60.0)` to `cls()` call
- [x] Add `deep_research_methodology_assessment_min_content_length=data.get("deep_research_methodology_assessment_min_content_length", 200)` to `cls()` call
- [x] Add `deep_research_academic_coverage_weights=data.get("deep_research_academic_coverage_weights", ...)` to `cls()` call
- [x] Add `deep_research_influence_high_citation_threshold=data.get("deep_research_influence_high_citation_threshold", 100)` to `cls()` call
- [x] Add `deep_research_influence_medium_citation_threshold=data.get("deep_research_influence_medium_citation_threshold", 20)` to `cls()` call
- [x] Add `deep_research_influence_low_citation_threshold=data.get("deep_research_influence_low_citation_threshold", 5)` to `cls()` call

#### Item 1.2 Validation

- [ ] Add unit test: `from_toml_dict` with each new field set to non-default value
- [x] Verify defaults match class field declarations
- [x] Existing config tests pass unchanged

---

### Item 1.3: Guard Against Legacy Session `AttributeError`
> **Files**: Multiple phase files

- [x] `synthesis.py:1140` — Guard `state.research_profile.name` with None check
- [x] Audit `synthesis.py` for all other `state.research_profile.X` accesses without guards
- [x] Audit `brief.py` for unguarded `state.research_profile` accesses
- [x] Audit `supervision_prompts.py` for unguarded `state.research_profile` accesses
- [x] Audit `topic_research.py` for unguarded `state.research_profile` accesses
- [x] Audit `_citation_postprocess.py` for unguarded `state.research_profile` accesses

#### Item 1.3 Validation

- [ ] Add unit test: synthesis with `state.research_profile = None` does not raise
- [ ] Add unit test: brief with `state.research_profile = None` does not raise
- [x] Existing tests pass unchanged

---

### Item 1.4: Fix `pubmed` Provider Hint Always Dropped
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`

- [x] Decision: replace `pubmed` with `semantic_scholar` in `_DISCIPLINE_PROVIDER_MAP` (no PubMed MCP integration exists)
- [x] Implement chosen fix
- [x] Document rationale in code comment

#### Item 1.4 Validation

- [x] Biomedical query brief produces a working provider hint (not silently dropped)
- [x] Existing brief tests pass (updated assertions for semantic_scholar)

---

## FIX-2: Correctness Fixes

### Item 2.1: Fix Citation Network Foundational Paper Threshold
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/citation_network.py`

- [x] Remove dead `threshold = max(3, ...)` calculation
- [x] Fix `effective_threshold` to implement intended logic (document which logic is intended)
- [x] Update docstring to match actual behavior

#### Item 2.1 Validation

- [x] Threshold calculation produces correct values for 5, 10, 20 discovered papers
- [x] Existing citation network tests pass (update threshold assertions if needed)

---

### Item 2.2: Add `MethodologyAssessment` Content-Basis Validator
> **File**: `src/foundry_mcp/core/research/models/sources.py`

- [x] Add `@model_validator(mode="after")` to `MethodologyAssessment`
- [x] Validator forces `confidence = "low"` when `content_basis == "abstract"`
- [x] Add warning log when confidence is downgraded

#### Item 2.2 Validation

- [x] `MethodologyAssessment(confidence="high", content_basis="abstract")` -> confidence is `"low"`
- [x] `MethodologyAssessment(confidence="high", content_basis="full_text")` -> confidence stays `"high"`
- [x] Existing tests pass unchanged

---

### Item 2.3: Fix `mark_interrupted()` Missing Provenance Timestamp
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add provenance timestamp to `mark_interrupted()`:
  ```python
  if self.extensions.provenance:
      self.extensions.provenance.completed_at = datetime.now(timezone.utc)
  ```

#### Item 2.3 Validation

- [x] Interrupted session has `provenance.completed_at` set
- [x] Existing tests pass unchanged

---

### Item 2.4: Fix Duplicate Synthesis Provenance Event
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] In `_inject_supplementary_raw_notes`, replace `_build_synthesis_system_prompt()` call with a length-only calculation (pass system prompt length as parameter, or cache and reuse)
- [x] Verify only one `synthesis_query_type` provenance event is appended per synthesis

#### Item 2.4 Validation

- [x] Run synthesis and verify exactly one `synthesis_query_type` provenance entry
- [x] Existing tests pass unchanged

---

### Item 2.5: Tighten `extract_status_code` Regex
> **File**: `src/foundry_mcp/core/research/providers/shared.py`

- [x] Replace `r"\b([1-5]\d{2})\b"` with a more anchored pattern (e.g., `r"(?:HTTP|API|status)\s*(?:error\s*)?(\d{3})\b|^(\d{3})\s"`)
- [x] Or extract status code from exception's response attribute when available, falling back to regex

#### Item 2.5 Validation

- [x] Error message "Found 200 results" does NOT extract 200 as status code
- [x] Error message "API error 429: rate limited" correctly extracts 429
- [x] Existing resilience tests pass unchanged

---

### Item 2.6: Fix `min_citation_count=0` Falsy Check
> **File**: `src/foundry_mcp/core/research/providers/semantic_scholar.py`

- [x] Change `if min_citation_count:` to `if min_citation_count is not None:`

#### Item 2.6 Validation

- [x] `min_citation_count=0` is applied as a filter (not skipped)
- [x] `min_citation_count=None` is correctly skipped
- [x] Existing tests pass unchanged

---

## FIX-3: Code Quality & Cleanup

### Item 3.1: Remove `if api_key or True:` Debug Artifact
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [ ] Remove the `if api_key or True:` conditional
- [ ] Unconditionally create `SemanticScholarProvider`
- [ ] Keep the comment explaining S2 works without a key at lower rate

#### Item 3.1 Validation

- [ ] Provider creation is unconditional
- [ ] Existing tests pass unchanged

---

### Item 3.2: Replace Deprecated `asyncio.get_event_loop()`
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [ ] Replace `asyncio.get_event_loop()` with try/except `asyncio.get_running_loop()` pattern
- [ ] Match the pattern used in `action_handlers.py:663-673`

#### Item 3.2 Validation

- [ ] No `DeprecationWarning` on Python 3.12+
- [ ] Existing tests pass unchanged

---

### Item 3.3: Consolidate Report Handler State Loading
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [ ] Load state once early in the success branch of `_handle_deep_research_report`
- [ ] Remove duplicate `memory.load_deep_research()` calls (lines ~264, 274, 288, 291)
- [ ] Reuse single `state` variable throughout

#### Item 3.3 Validation

- [ ] State loaded exactly once per report request
- [ ] Report response still includes provenance when `include_provenance=True`
- [ ] Existing tests pass unchanged

---

### Item 3.4: Fix Magic-Number Default Comparison for Network Config
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [ ] Change `max_refs` and `max_cites` parameter types to `Optional[int] = None`
- [ ] Replace `if effective_max_refs == 20:` with `if effective_max_refs is None:`
- [ ] Replace `if effective_max_cites == 20:` with `if effective_max_cites is None:`

#### Item 3.4 Validation

- [ ] User passing `max_refs=20` explicitly is not overridden by config
- [ ] User passing no value correctly falls back to config
- [ ] Existing tests pass unchanged

---

## FIX-4: Robustness Hardening

### Item 4.1: Use `find` Instead of `rfind` for Supervisor Brief Split
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`

- [ ] Change `content.rfind(_SUPERVISOR_BRIEF_MARKER)` to `content.find(_SUPERVISOR_BRIEF_MARKER)`

#### Item 4.1 Validation

- [ ] First occurrence of marker is used for split (not last)
- [ ] Existing compression tests pass unchanged

---

### Item 4.2: Use Word Boundaries for Provider Hint Keywords
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`

- [ ] Replace `if keyword in text_lower:` with `if re.search(rf"\b{re.escape(keyword)}\b", text_lower):`
- [ ] Import `re` if not already imported

#### Item 4.2 Validation

- [ ] `"health"` does NOT match "healthy eating habits"
- [ ] `"health"` DOES match "public health research"
- [ ] `"learning"` does NOT match "learning disabilities"
- [ ] `"machine learning"` DOES match "advances in machine learning"
- [ ] Existing brief tests pass (update keyword test assertions if needed)

---

### Item 4.3: Make `methodology_assessments` Optional for Exclude-None Consistency
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [ ] Change `methodology_assessments: list[MethodologyAssessment] = Field(default_factory=list)` to `methodology_assessments: Optional[list[MethodologyAssessment]] = None`
- [ ] Update convenience accessor on `DeepResearchState` to return `self.extensions.methodology_assessments or []`

#### Item 4.3 Validation

- [ ] Unused extensions serialize without `methodology_assessments` key
- [ ] Accessor returns empty list when None
- [ ] Existing tests pass unchanged

---

### Item 4.4: Use `Literal` Type for `MethodologyAssessment.confidence`
> **File**: `src/foundry_mcp/core/research/models/sources.py`

- [ ] Change `confidence: str = "low"` to `confidence: Literal["high", "medium", "low"] = "low"`
- [ ] Add `from typing import Literal` import

#### Item 4.4 Validation

- [ ] `MethodologyAssessment(confidence="invalid")` raises `ValidationError`
- [ ] `MethodologyAssessment(confidence="high")` succeeds
- [ ] Existing tests pass unchanged

---

### Item 4.5: Extract `_truncate_abstract` to Shared Utility
> **Files**: `providers/shared.py`, `providers/openalex.py`, `providers/crossref.py`, `providers/semantic_scholar.py`

- [ ] Move `_truncate_abstract` to `shared.py` as a module-level function `truncate_abstract()`
- [ ] Update imports in `openalex.py`
- [ ] Update imports in `crossref.py`
- [ ] Update imports in `semantic_scholar.py`
- [ ] Remove duplicate method definitions from all three providers

#### Item 4.5 Validation

- [ ] All three providers use the shared function
- [ ] Existing provider tests pass unchanged

---

## FIX-5: Test Improvements

### Item 5.1: Fix `study_design="rct"` Test Bug
> **File**: `tests/core/research/test_methodology_assessment.py`

- [ ] Change `_make_llm_json_response(study_design="rct")` to `_make_llm_json_response(study_design="randomized_controlled_trial")` (line ~805)
- [ ] Verify the test now properly validates the success path for the first source

#### Item 5.1 Validation

- [ ] First source in the test produces a valid `randomized_controlled_trial` assessment (not UNKNOWN fallback)
- [ ] Second source still correctly triggers the LLM failure fallback path

---

### Item 5.2: Add PDF Extraction HTTP Tests
> **File**: `tests/core/research/test_pdf_analysis.py`

- [ ] Add test: `test_extract_from_url_success` with mocked HTTP response returning PDF bytes
- [ ] Add test: `test_extract_from_url_timeout` with mocked timeout
- [ ] Add test: `test_extract_from_url_malformed_pdf` with invalid PDF bytes
- [ ] Add test: `test_extract_from_url_ssrf_blocked` with private IP URL

#### Item 5.2 Validation

- [ ] All new tests pass
- [ ] Existing tests pass unchanged

---

### Item 5.3: Fix RIS Page Range Spec Compliance
> **File**: `src/foundry_mcp/core/research/export/ris.py`

- [ ] Split page ranges into `SP` and `EP` tags:
  ```python
  if pages and "-" in str(pages):
      parts = str(pages).split("-", 1)
      lines.append(f"SP  - {parts[0].strip()}")
      lines.append(f"EP  - {parts[1].strip()}")
  else:
      lines.append(f"SP  - {pages}")
  ```
- [ ] Update existing RIS tests to expect split tags

#### Item 5.3 Validation

- [ ] Page range `"123-456"` produces `SP  - 123` and `EP  - 456`
- [ ] Single page `"42"` produces `SP  - 42` only
- [ ] No page value produces no SP/EP tags
- [ ] Existing tests pass (with updated assertions)

---

## Final Validation

- [ ] All 7,582+ tests pass
- [ ] No new warnings on Python 3.12+
- [ ] No API keys in URL query strings
- [ ] All external content sanitized before LLM prompt interpolation
- [ ] All config fields parsed from TOML
- [ ] Legacy sessions (pre-PLAN-1) can continue without errors
- [ ] RIS export produces spec-compliant output
- [ ] Methodology assessment callable as user-triggered action
