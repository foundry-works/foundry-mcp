# Post-Review Fix Plan — `deep-academic` Branch

> **Branch**: `deep-academic`
>
> **Context**: Senior engineering review of 38 commits (~67K LOC added) implementing PLAN-0 through PLAN-4. All 7,582 tests pass. This plan addresses issues found during review, organized by priority and dependency.
>
> **Total estimated scope**: ~250-400 LOC fixes + ~80-150 LOC test additions
>
> **Source**: Code review of `deep-academic` branch against `main`

---

## Execution Order

```
FIX-0  Security fixes ──────────────────────────────────┐ (no deps, do first)
FIX-1  Integration blockers ─────────────────────────────┤ (parallel with FIX-0)
FIX-2  Correctness fixes ───────────────────────────────┤ (parallel with FIX-0/1)
FIX-3  Code quality & cleanup ──────────────────────────┤ (after FIX-0)
FIX-4  Robustness hardening ────────────────────────────┤ (after FIX-2)
FIX-5  Test improvements ──────────────────────────────┘ (after FIX-1/2)
```

All six phases are largely independent and can be parallelized.

---

## FIX-0: Security Fixes

> **Scope**: ~30-50 LOC | **Risk**: Low (targeted fixes) | **Priority**: HIGH

Address credential exposure and prompt injection gaps found during review.

### Item 0.1: Move OpenAlex API Key to Header

**Problem**: `openalex.py:429` sends the API key as a URL query parameter (`params["api_key"] = self._api_key`), exposing it in server logs, proxy logs, and httpx debug output. Semantic Scholar correctly uses header-based auth.

**File**: `src/foundry_mcp/core/research/providers/openalex.py`

**Fix**: Replace `params["api_key"] = self._api_key` with `headers["x-api-key"] = self._api_key`. The `headers` dict is already scaffolded (line 426) but never populated.

### Item 0.2: Sanitize Assistant Messages in ReAct Prompt

**Problem**: In `_build_react_user_prompt` (topic_research.py:504), tool results are sanitized via `sanitize_external_content()`, but assistant messages are included verbatim. Poisoned web pages can cause the LLM to echo injection payloads in its response, which are then re-injected into subsequent turns without sanitization — a second-order prompt injection vector.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

**Fix**: Apply `sanitize_external_content()` to assistant message content in the `role == "assistant"` branch (line 504), matching the treatment of tool results on line 509.

### Item 0.3: Sanitize Content in Methodology Assessment Prompts

**Problem**: `_build_extraction_user_prompt` (methodology_assessment.py:75-78) interpolates `source_title` and `content` directly into the LLM prompt without calling `sanitize_external_content()`. Every other phase file consistently sanitizes external content. Additionally, `format_methodology_context` (line 383) injects LLM-generated assessment fields into the synthesis prompt without sanitization, propagating any injection payloads.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py`

**Fix**:
- Wrap `source_title` and `content` with `sanitize_external_content()` in `_build_extraction_user_prompt`
- Wrap LLM-generated string fields (`effect_size`, `sample_description`, `limitations_noted`, `potential_biases`) in `format_methodology_context`

---

## FIX-1: Integration Blockers

> **Scope**: ~80-120 LOC | **Risk**: Low-Medium | **Priority**: HIGH

Fix dead code paths and broken integration points.

### Item 1.1: Wire Methodology Assessment into Pipeline

**Problem**: `MethodologyAssessor.assess_sources()` exists and `synthesis.py` consumes `state.methodology_assessments`, but nothing in the orchestration pipeline ever calls the assessor. The only callsites are in tests. `state.methodology_assessments` will always be an empty list.

**Decision**: Wire as a user-triggered post-hoc action (consistent with citation network), since automatic assessment of 20+ sources with 60s timeout each would be too expensive for the default pipeline.

**Files**:
- `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py` — Add `_handle_deep_research_assess()` handler
- `src/foundry_mcp/tools/unified/research_handlers/__init__.py` — Register `deep-research-assess` action

### Item 1.2: Add Missing Config Fields to `from_toml_dict()`

**Problem**: Nine PLAN-3/4 fields are declared as class fields but never parsed in `from_toml_dict()`. Users configuring these in TOML get silent defaults.

**File**: `src/foundry_mcp/config/research.py`

**Missing fields**:
- `deep_research_pdf_max_pages`
- `deep_research_pdf_priority_sections`
- `deep_research_citation_network_max_refs_per_paper`
- `deep_research_citation_network_max_cites_per_paper`
- `deep_research_methodology_assessment_provider`
- `deep_research_methodology_assessment_timeout`
- `deep_research_methodology_assessment_min_content_length`
- `deep_research_academic_coverage_weights`
- `deep_research_influence_high_citation_threshold`
- `deep_research_influence_medium_citation_threshold`
- `deep_research_influence_low_citation_threshold`

**Fix**: Add each field to the `cls(...)` constructor call in `from_toml_dict()`, using the same `data.get("field", default)` pattern as existing fields.

### Item 1.3: Guard Against Legacy Session `AttributeError`

**Problem**: `synthesis.py:1140` calls `state.research_profile.name` without a None guard. Sessions started before PLAN-1 have `research_profile = None`, causing `AttributeError` when they reach synthesis via `continue`.

**Files**:
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` — Guard all `state.research_profile.X` accesses
- Audit other phase files for the same pattern

**Fix**: Replace bare `state.research_profile.name` with `state.research_profile.name if state.research_profile else "general"` (and equivalent guards for other attribute accesses).

### Item 1.4: Fix `pubmed` Provider Hint Always Dropped

**Problem**: `_DISCIPLINE_PROVIDER_MAP` in `brief.py:401-404` emits `"pubmed"` for biomedical queries, but `known_providers` (line 486) doesn't include `"pubmed"`. The hint is extracted then silently discarded.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`

**Fix**: Either add `"pubmed"` to the `known_providers` set (if PubMed MCP integration is expected), or remove `"pubmed"` from `_DISCIPLINE_PROVIDER_MAP` and replace with `"semantic_scholar"` (which is the actual biomedical-capable provider available in the pipeline). Document the decision.

---

## FIX-2: Correctness Fixes

> **Scope**: ~60-90 LOC | **Risk**: Low | **Priority**: MEDIUM

Fix logic errors, missing validators, and consistency gaps.

### Item 2.1: Fix Citation Network Foundational Paper Threshold

**Problem**: `citation_network.py:262-264` has `threshold = max(3, ...)` followed by `effective_threshold = min(3, threshold)`, which always evaluates to exactly 3. The 30% calculation has no effect.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/citation_network.py`

**Fix**: Clarify intent and fix. If the goal is "cited by at least 3 OR at least 30% of discovered papers, whichever is lower": `effective_threshold = min(3, max(1, int(len(discovered_ids) * 0.3)))`. Remove the dead `threshold` variable.

### Item 2.2: Add `MethodologyAssessment` Content-Basis Validator

**Problem**: Docstring states confidence is forced to `"low"` when `content_basis == "abstract"`, but no `model_validator` enforces this. Callers can construct invalid state.

**File**: `src/foundry_mcp/core/research/models/sources.py`

**Fix**: Add a Pydantic `model_validator(mode="after")` that forces `confidence = "low"` when `content_basis == "abstract"`.

### Item 2.3: Fix `mark_interrupted()` Missing Provenance Timestamp

**Problem**: `mark_failed`, `mark_cancelled`, and `mark_completed` all stamp `provenance.completed_at`. `mark_interrupted` does not, making interrupted sessions indistinguishable from in-progress via provenance.

**File**: `src/foundry_mcp/core/research/models/deep_research.py`

**Fix**: Add `if self.extensions.provenance: self.extensions.provenance.completed_at = datetime.now(timezone.utc)` to `mark_interrupted()`.

### Item 2.4: Fix Duplicate Synthesis Provenance Event

**Problem**: `_inject_supplementary_raw_notes` (synthesis.py:1654) calls `_build_synthesis_system_prompt()` again, which appends a second `synthesis_query_type` provenance event.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

**Fix**: Pass the system prompt length as a parameter instead of rebuilding the prompt. Or extract query type classification into a separate method that doesn't log provenance, and call the provenance-logging version only once.

### Item 2.5: Tighten `extract_status_code` Regex

**Problem**: `shared.py:296-301` regex `r"\b([1-5]\d{2})\b"` matches any 3-digit number (100-599) anywhere in the error message, causing false positives in error classification.

**File**: `src/foundry_mcp/core/research/providers/shared.py`

**Fix**: Anchor the pattern to known error message formats: `r"(?:HTTP|API|status)\s*(?:error\s*)?(\d{3})\b|^(\d{3})\s"`. Or extract the status code from the exception's response attribute when available rather than parsing strings.

### Item 2.6: Fix `min_citation_count=0` Treated as Falsy

**Problem**: `semantic_scholar.py:311` uses `if min_citation_count:` which skips the filter when the value is `0` (a valid filter meaning "include all").

**File**: `src/foundry_mcp/core/research/providers/semantic_scholar.py`

**Fix**: Change to `if min_citation_count is not None:`.

---

## FIX-3: Code Quality & Cleanup

> **Scope**: ~40-60 LOC | **Risk**: Very Low | **Priority**: MEDIUM

Remove debug artifacts, fix deprecations, and improve clarity.

### Item 3.1: Remove `if api_key or True:` Debug Artifact

**Problem**: `handlers_deep_research.py:649` has `if api_key or True:` which is always True. Debugging leftover.

**File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

**Fix**: Remove the conditional. Just unconditionally create the provider.

### Item 3.2: Replace Deprecated `asyncio.get_event_loop()`

**Problem**: `handlers_deep_research.py:677` uses `asyncio.get_event_loop()`, deprecated since Python 3.10. The rest of the codebase uses the try/except `asyncio.get_running_loop()` pattern.

**File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

**Fix**: Use the same pattern as `action_handlers.py:663-673`.

### Item 3.3: Consolidate Report Handler State Loading

**Problem**: `_handle_deep_research_report` (handlers_deep_research.py:264-291) loads state up to 3 times with convoluted conditional logic.

**File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

**Fix**: Load state once early in the success branch and reuse throughout.

### Item 3.4: Fix Magic-Number Default Comparison for Network Config

**Problem**: `handlers_deep_research.py:625-628` compares `effective_max_refs == 20` to detect whether the user passed a parameter. If the user explicitly passes `20`, the config value overrides their explicit choice.

**File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

**Fix**: Use `Optional[int] = None` parameter type and check for `None`.

---

## FIX-4: Robustness Hardening

> **Scope**: ~30-50 LOC | **Risk**: Very Low | **Priority**: LOW

Tighten safety margins and fix minor functional issues.

### Item 4.1: Use `rfind` → `find` for Supervisor Brief Split

**Problem**: `compression.py:418` uses `rfind` to locate `## SUPERVISOR BRIEF` marker. If the LLM produces the marker twice (or it appears in preserved source content), only the content after the last occurrence is captured.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`

**Fix**: Change `rfind` to `find` (first occurrence).

### Item 4.2: Use Word Boundaries for Provider Hint Keywords

**Problem**: `brief.py:438-448` matches keywords as raw substrings. `"health"` matches "healthy eating habits", `"learning"` matches "learning disabilities". False positives activate incorrect providers.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`

**Fix**: Use `re.search(rf"\b{re.escape(keyword)}\b", text_lower)` instead of `keyword in text_lower`.

### Item 4.3: Add `methodology_assessments` Exclude-None Consistency

**Problem**: `ResearchExtensions.methodology_assessments` is `list[MethodologyAssessment]` with `default_factory=list`. An empty list won't be stripped by `exclude_none=True`, adding noise to serialized state for sessions that never use methodology assessment.

**File**: `src/foundry_mcp/core/research/models/deep_research.py`

**Fix**: Change to `Optional[list[MethodologyAssessment]] = None` for consistency with the exclude-none pattern. Update accessors to return `[]` on None.

### Item 4.4: Use `Literal` Type for `MethodologyAssessment.confidence`

**Problem**: `confidence` field is typed as `str` but should only accept `"high"`, `"medium"`, or `"low"`.

**File**: `src/foundry_mcp/core/research/models/sources.py`

**Fix**: Change type to `Literal["high", "medium", "low"]` with default `"low"`.

### Item 4.5: Extract `_truncate_abstract` to Shared Utility

**Problem**: Identical `_truncate_abstract` method duplicated across `openalex.py`, `crossref.py`, and `semantic_scholar.py`.

**Files**: `providers/openalex.py`, `providers/crossref.py`, `providers/semantic_scholar.py`, `providers/shared.py`

**Fix**: Move to `shared.py` as a module-level function. Update imports in all three providers.

---

## FIX-5: Test Improvements

> **Scope**: ~50-80 LOC | **Risk**: Very Low | **Priority**: LOW

Fix known test bugs and address coverage gaps.

### Item 5.1: Fix `study_design="rct"` Test Bug

**Problem**: `test_methodology_assessment.py:805` uses `_make_llm_json_response(study_design="rct")` but `"rct"` is not a valid `StudyDesign` enum value. The valid value is `"randomized_controlled_trial"`. The first source falls back to UNKNOWN for the wrong reason, masking whether the success path actually works.

**File**: `tests/core/research/test_methodology_assessment.py`

**Fix**: Change `"rct"` to `"randomized_controlled_trial"`.

### Item 5.2: Add PDF Extraction HTTP Tests

**Problem**: `test_pdf_analysis.py` has a 0.42:1 test-to-impl ratio. No tests for `extract_from_url()`, multi-page PDFs, or warning handling.

**File**: `tests/core/research/test_pdf_analysis.py`

**Fix**: Add tests for `extract_from_url()` with mocked HTTP responses covering success, timeout, and malformed PDF cases.

### Item 5.3: Fix RIS Page Range Spec Compliance

**Problem**: `export/ris.py:101` emits `SP  - 123-456` for page ranges. The RIS spec requires separate `SP` and `EP` tags.

**File**: `src/foundry_mcp/core/research/export/ris.py`

**Fix**:
```python
if pages and "-" in str(pages):
    parts = str(pages).split("-", 1)
    lines.append(f"SP  - {parts[0].strip()}")
    lines.append(f"EP  - {parts[1].strip()}")
else:
    lines.append(f"SP  - {pages}")
```

---

## Estimated Scope

| Phase | Fix LOC | Test LOC | Focus |
|-------|---------|----------|-------|
| 0. Security | ~30-50 | ~20-30 | API key, sanitization |
| 1. Integration | ~80-120 | ~30-50 | Dead paths, config, compat |
| 2. Correctness | ~60-90 | ~20-30 | Logic bugs, validators |
| 3. Quality | ~40-60 | — | Debug artifacts, deprecations |
| 4. Hardening | ~30-50 | — | Safety margins, types |
| 5. Tests | ~10-20 | ~50-80 | Test bugs, coverage |
| **Total** | **~250-390** | **~120-190** | |
