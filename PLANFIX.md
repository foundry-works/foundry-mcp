# Post-Review Fix Plan v4 — `deep-academic` Branch

> **Branch**: `deep-academic`
>
> **Context**: Senior engineering review (v4) of 46 commits (~69K LOC added) implementing PLAN-0 through PLAN-4, including FIX-0 through FIX-4 from review v3. All 7,607 tests pass. This plan addresses remaining and newly discovered issues, organized by priority and dependency.
>
> **Total estimated scope**: ~80-140 LOC fixes + ~150-250 LOC test additions
>
> **Source**: Code review of `deep-academic` branch against `main`, conducted by 8 parallel review agents covering models, config, providers (OpenAlex, Crossref), supervision, topic_research, synthesis, handlers, and utility/export modules.
>
> **Relationship to prior reviews**: v1–v3 (FIX-0 through FIX-4) were implemented in commits `a9f0db8` through `3371782`. This plan covers issues found after those fixes.

---

## Execution Order

```
FIX-0  Correctness & data integrity ───────────────────┐ (no deps, do first)
FIX-1  Config validation ──────────────────────────────┤ (parallel with FIX-0)
FIX-2  Test coverage gaps ─────────────────────────────┤ (after FIX-0)
FIX-3  Cleanup & consistency ──────────────────────────┘ (after FIX-0, FIX-1)
```

---

## FIX-0: Correctness & Data Integrity

> **Scope**: ~30-50 LOC | **Risk**: Low (targeted fixes) | **Priority**: MEDIUM

### Item 0.1: Remove Duplicate Metadata Key in OpenAlex Provider

**Problem**: `openalex.py` stores `cited_by_count` under two keys — `citation_count` (line ~616) and `cited_by_count` (line ~626) — in the same metadata dict. This is a DRY violation and creates ambiguity for downstream consumers about which key to use.

**File**: `src/foundry_mcp/core/research/providers/openalex.py`

**Fix**: Remove the `citation_count` key. Keep only `cited_by_count` (which matches the OpenAlex API field name). Grep for any consumers of `metadata["citation_count"]` and update to `metadata["cited_by_count"]` or `metadata.get("citation_count", metadata.get("cited_by_count"))` for safety.

### Item 0.2: Fix Provider Fallback Naming in Topic Research

**Problem**: `topic_research.py:~2318` — when both Semantic Scholar and OpenAlex fail for `citation_search`, `provider_name` is set to `"unknown"`. The response message will say "unknown provider returned X results", which is uninformative for debugging. Same issue in `related_papers` handler.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

**Fix**: Track which providers were attempted and report the last one tried. For example: `provider_name = last_attempted_provider or "no_provider_available"`.

### Item 0.3: Simplify Redundant Boolean Check in Coverage Scoring

**Problem**: `supervision_coverage.py:354` has `if influential_count and influential_count > 0:` — the second condition subsumes the first since `0` is falsy in Python. Not a bug, but misleading to readers.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_coverage.py`

**Fix**: Simplify to `if influential_count > 0:`.

---

## FIX-1: Config Validation

> **Scope**: ~15-25 LOC | **Risk**: Low | **Priority**: MEDIUM

### Item 1.1: Validate Citation Influence Threshold Ordering

**Problem**: `config/research.py:802-809` defines three citation influence thresholds (`low=5`, `medium=20`, `high=100`) but never validates that `low < medium < high`. A misconfiguration like `low=50, medium=20, high=10` would silently produce nonsensical influence scoring where every source gets the same weight tier.

**File**: `src/foundry_mcp/config/research.py`

**Fix**: Add validation in `__post_init__()` (alongside existing `_validate_deep_research_bounds()`):
```python
if not (self.deep_research_influence_low_citation_threshold
        < self.deep_research_influence_medium_citation_threshold
        < self.deep_research_influence_high_citation_threshold):
    raise ValueError(
        "Citation influence thresholds must satisfy: low < medium < high "
        f"(got {self.deep_research_influence_low_citation_threshold} < "
        f"{self.deep_research_influence_medium_citation_threshold} < "
        f"{self.deep_research_influence_high_citation_threshold})"
    )
```

### Item 1.2: Use Provider Registry Instead of Hardcoded Provider List in Brief

**Problem**: `phases/brief.py:~494` defines `_KNOWN_PROVIDERS` as a hardcoded frozenset of 6 provider names. If a new provider is registered in the provider system but not added to this list, adaptive provider selection hints for that provider will be silently dropped.

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`

**Fix**: Either (a) import the provider registry and query dynamically, or (b) add a comment documenting that this list must be kept in sync with the provider registry and move the constant to a shared location. Option (b) is simpler and sufficient since new providers are rare.

---

## FIX-2: Test Coverage Gaps

> **Scope**: ~150-250 LOC test additions | **Risk**: None (test-only) | **Priority**: MEDIUM

### Item 2.1: Add Citation Tool Gating Tests

**Problem**: No tests verify that `citation_search` and `related_papers` tools are rejected when `enable_citation_tools=False` on the research profile.

**File**: `tests/core/research/workflows/test_topic_research.py` (or `deep_research/` subfolder)

**Tests to add**:
- `test_citation_search_rejected_when_gated()` — profile with `enable_citation_tools=False`, researcher attempts `citation_search` tool call, verify rejection response
- `test_related_papers_rejected_when_gated()` — same for `related_papers`

### Item 2.2: Add Provider Fallback Chain Tests

**Problem**: No tests verify the Semantic Scholar → OpenAlex fallback behavior in `citation_search` and `related_papers` handlers.

**File**: `tests/core/research/workflows/test_topic_research.py`

**Tests to add**:
- `test_citation_search_falls_back_to_openalex()` — mock S2 to raise, verify OpenAlex is tried
- `test_related_papers_falls_back_to_openalex()` — same pattern
- `test_citation_search_both_providers_fail()` — verify empty result, no crash

### Item 2.3: Add OpenAlex `get_related()` and OOM Cap Tests

**Problem**: `get_related()` method exists but has no dedicated test. Also, the abstract reconstruction OOM cap at 100K positions is untested.

**File**: `tests/core/research/providers/test_openalex.py`

**Tests to add**:
- `test_get_related()` — mock response, verify correct results
- `test_get_related_no_related_works()` — verify empty result
- `test_abstract_reconstruction_oom_cap()` — inverted index with position > 100K returns None

### Item 2.4: Add Paper ID Validation Tests

**Problem**: `_validate_paper_id()` in `topic_research.py` enforces a strict regex but has no direct unit test.

**File**: `tests/core/research/workflows/test_topic_research.py`

**Tests to add**:
- `test_validate_paper_id_accepts_doi()` — e.g. `"10.1234/test.2024"`
- `test_validate_paper_id_accepts_s2_hex()` — e.g. `"a1b2c3d4e5f6"`
- `test_validate_paper_id_accepts_arxiv()` — e.g. `"2301.12345"`
- `test_validate_paper_id_rejects_injection()` — e.g. `"10.1234; DROP TABLE"`
- `test_validate_paper_id_rejects_too_long()` — 257+ chars

### Item 2.5: Add OpenAlex 403 Error Test

**Problem**: OpenAlex raises `AuthenticationError` for 403, but no test validates this path.

**File**: `tests/core/research/providers/test_openalex.py`

**Tests to add**:
- `test_403_raises_authentication_error()` — mock 403 response, verify `AuthenticationError` raised

---

## FIX-3: Cleanup & Consistency

> **Scope**: ~20-40 LOC | **Risk**: None | **Priority**: LOW

### Item 3.1: Update PLAN-CHECKLIST.md Completion Status

**Problem**: `PLAN-CHECKLIST.md` has Phase 0 items unchecked (supervision first-round extraction, thin wrapper removal) even though the code shows these were implemented. The checklist doesn't reflect actual completion state.

**File**: `PLAN-CHECKLIST.md`

**Fix**: Audit each Phase 0 checklist item against the codebase and mark completed items as `[x]`.

### Item 3.2: Remove Orphan Research Output Files from Repo Root

**Problem**: Several markdown files in the repo root appear to be deep research output artifacts that were accidentally committed or left untracked:
- `compare-postgresql-vs-mysql-for-oltp-workloads-in-2024-*.md` (3 files)
- `test-progress-heartbeat*.md` (2 files)
- `test-synthesis-heartbeat*.md` (2 files)
- `what-are-the-benefits-of-renewable-energy*.md` (2 files)

**Fix**: Add to `.gitignore` or delete. These are generated outputs, not source code.

### Item 3.3: Document StructuredResearchOutput Dict Schemas

**Problem**: `StructuredResearchOutput` uses `list[dict]` for sources, findings, gaps, and contradictions, but the expected key-value pairs are undocumented. Consumers must read the `_build_structured_output()` implementation to understand the schema.

**File**: `src/foundry_mcp/core/research/models/deep_research.py`

**Fix**: Add docstring examples to `StructuredResearchOutput` showing the expected dict shape for each field. For example:
```python
sources: list[dict[str, Any]] = Field(
    default_factory=list,
    description="Denormalized source catalog. Keys: id, title, url, source_type, "
    "quality, citation_number, authors, year, venue, doi, citation_count, ..."
)
```

### Item 3.4: Add `generated_at` Timestamps to Output Models

**Problem**: `StructuredResearchOutput` and `CitationNetwork` have no timestamp indicating when they were generated. This makes it impossible to detect staleness or correlate with session timeline.

**File**: `src/foundry_mcp/core/research/models/deep_research.py`

**Fix**: Add `generated_at: Optional[str] = None` to both models. Populate with ISO 8601 timestamp at generation time in `_build_structured_output()` and `CitationNetworkBuilder.build_network()`.

---

## Estimated Scope

| Fix | Impl LOC | Test LOC | Focus |
|-----|----------|----------|-------|
| 0. Correctness & data integrity | ~30-50 | ~0 | Metadata dedup, naming, simplification |
| 1. Config validation | ~15-25 | ~20-30 | Threshold ordering, provider list |
| 2. Test coverage gaps | ~0 | ~150-250 | Citation gating, fallback, OOM, validation |
| 3. Cleanup & consistency | ~20-40 | ~0 | Checklist, orphan files, docs, timestamps |
| **Total** | **~65-115** | **~170-280** | |
