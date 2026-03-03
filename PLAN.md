# PLAN: Deep Research Post-Synthesis Quality Improvements

**Branch:** `alpha`
**Scope:** Selective academic provider usage, source relevance filtering, claim verification default, citation regex fix

---

## Motivation

Analysis of `deepres-cb42b20e188f` (a consumer credit-card research session using the `general` profile) revealed:

1. **Academic provider noise** — `PROFILE_GENERAL` defaults to `["tavily", "semantic_scholar"]`, causing Semantic Scholar to fire on every general query. This produced 14 academic sources, of which 8-10 were completely irrelevant (seaplane transport in Greece, NHL arena financing, pilot training in Zambia, etc.). The synthesis was smart enough to ignore them, but they wasted search budget, tokens, and compression time.
2. **No source relevance gate** — Once a source passes URL/title/content dedup, it is unconditionally added to state regardless of topical relevance.
3. **Claim verification defaults to off** — The `deep_research_claim_verification_enabled` config and `ResearchProfile.enable_claim_verification` both default to `False`, meaning the feature is invisible to users who don't read config docs.
4. **Citation year-as-number bug** — `_CITATION_RE = re.compile(r"\[(\d+)\](?!\()")` matches `[2025]` and `[2026]` year references in report text as citation numbers, producing false "dangling citation" warnings.

---

## Phase 1: Selective Academic Provider Usage

**Goal:** Academic providers (semantic_scholar, openalex, crossref) should only activate when the query has academic relevance. The existing adaptive hint system in `brief.py` already adds providers based on discipline keywords — the problem is that `PROFILE_GENERAL` includes `semantic_scholar` unconditionally.

### Changes

#### 1a. Remove `semantic_scholar` from `PROFILE_GENERAL` default providers

**File:** `src/foundry_mcp/core/research/models/deep_research.py` (lines 983-988)

```python
# Before
PROFILE_GENERAL = ResearchProfile(
    name="general",
    providers=["tavily", "semantic_scholar"],
    ...
)

# After
PROFILE_GENERAL = ResearchProfile(
    name="general",
    providers=["tavily"],
    ...
)
```

The adaptive hint system (`BriefPhaseMixin._extract_provider_hints` + `_apply_provider_hints` in `brief.py:435-528`) will re-add `semantic_scholar` or `openalex` when the research brief contains discipline keywords like "biomedical", "clinical", "computer science", "education", etc. This is the right behavior — academic providers should be demand-driven, not always-on.

#### 1b. Expand discipline keyword coverage in `_DISCIPLINE_PROVIDER_MAP`

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py` (lines 413-432)

The current keyword map only covers 4 discipline groups. Add coverage for additional academic domains to ensure the hint system activates academic providers when genuinely needed:

```python
_DISCIPLINE_PROVIDER_MAP: list[tuple[list[str], str]] = [
    # Existing groups (keep as-is)
    (["biomedical", "clinical", "health", "medical", "medicine", ...], "semantic_scholar"),
    (["computer science", "machine learning", "artificial intelligence", ...], "semantic_scholar"),
    (["education", "pedagogy", "curriculum", ...], "openalex"),
    (["social science", "sociology", "economics", ...], "openalex"),

    # New groups
    (["physics", "chemistry", "biology", "ecology", "geology",
      "environmental science", "climate"], "semantic_scholar"),
    (["mathematics", "statistics", "operations research"], "semantic_scholar"),
    (["psychology", "cognitive science", "neuroscience",
      "behavioral science"], "semantic_scholar"),
    (["engineering", "robotics", "materials science",
      "electrical engineering"], "semantic_scholar"),
    (["law", "jurisprudence", "legal analysis", "regulation",
      "policy analysis"], "openalex"),
    (["history", "philosophy", "literature review",
      "systematic review", "meta-analysis"], "openalex"),
]
```

#### 1c. Tests

- Unit test: `PROFILE_GENERAL.providers` does not contain academic providers by default.
- Unit test: `_extract_provider_hints` returns `["semantic_scholar"]` for briefs containing "machine learning" but `[]` for "best credit card for travel rewards".
- Integration test: A general-profile session with a consumer query produces `active_providers == ["tavily"]` after brief phase.

---

## Phase 2: Source Relevance Filtering

**Goal:** Add a lightweight relevance gate that runs during source collection to detect and flag sources that are topically irrelevant to the research query. Irrelevant sources are retained in state (for provenance/audit) but deprioritized for compression and synthesis.

### Design

Insert relevance scoring into the existing `_dedup_and_add_source()` flow in `topic_research.py:354-428`. After a source passes dedup checks (Phase 3 atomic commit), compute a relevance score before final addition.

#### 2a. Add `relevance_score` field to `ResearchSource`

**File:** `src/foundry_mcp/core/research/models/sources.py`

Add `relevance_score: float | None = None` field to `ResearchSource`. Values:
- `None` = not yet scored (default)
- `0.0 – 1.0` = scored, where 0.0 = irrelevant, 1.0 = highly relevant

#### 2b. Implement keyword-based relevance scoring

**File:** `src/foundry_mcp/core/research/workflows/deep_research/source_quality.py` (new function)

```python
def compute_source_relevance(
    source_title: str,
    source_content: str | None,
    reference_text: str,
    *,
    source_type: str = "web",
) -> float:
    """Score source relevance against reference text (brief + sub-query).

    Uses weighted keyword overlap between source title/content and the
    reference text. Academic sources (source_type="academic") receive
    a stricter scoring curve since they are more likely to be tangential
    hits from broad academic search APIs.

    Returns:
        Float in [0.0, 1.0] — higher means more relevant.
    """
```

Approach:
1. Tokenize reference text and source title+content into keyword sets (lowercase, strip stopwords).
2. Compute Jaccard similarity between keyword sets.
3. Weight title overlap more heavily than content overlap (title is a stronger relevance signal).
4. For `source_type == "academic"`, apply a stricter threshold curve (multiply raw score by 0.7) to account for academic search APIs returning tangentially related papers.
5. Return clamped `[0.0, 1.0]` score.

#### 2c. Integrate into `_dedup_and_add_source`

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (lines 410-428)

In Phase 3 (atomic commit), after quality scoring but before `state.append_source(source)`:

```python
# Compute relevance against sub-query + brief
reference_text = sub_query.query
if state.research_brief:
    reference_text = state.research_brief + "\n" + reference_text
source.relevance_score = compute_source_relevance(
    source_title=source.title or "",
    source_content=(source.content or "")[:2000],  # Cap content for performance
    reference_text=reference_text,
    source_type=source.source_type or "web",
)
```

#### 2d. Add config option for relevance threshold

**File:** `src/foundry_mcp/config/research.py`

```python
deep_research_source_relevance_threshold: float = 0.05
    # Sources scoring below this threshold are flagged as irrelevant.
    # They remain in state for provenance but are deprioritized during
    # compression and synthesis. Set to 0.0 to disable filtering.
```

#### 2e. Deprioritize low-relevance sources in compression

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`

When building the compression prompt, sources with `relevance_score` below the threshold should be excluded from the compression input (they add noise and waste tokens). Log the exclusion for audit.

#### 2f. Tests

- Unit test: `compute_source_relevance("PESTLE Analysis of Seaplane Transport in Greece", None, "best credit card for travel rewards maximizing business class flights")` returns a low score (< 0.1).
- Unit test: `compute_source_relevance("Chase Sapphire Reserve 2025 Review", None, "best credit card for travel rewards")` returns a high score (> 0.5).
- Unit test: Academic source with same-topic keywords scores lower than web source with same keywords (stricter curve).
- Integration test: Sources below threshold are excluded from compression input but remain in `state.sources`.

---

## Phase 3: Default Claim Verification to True

**Goal:** Claim verification should be on by default so users benefit without needing to discover the config option.

### Changes

#### 3a. Change config default

**File:** `src/foundry_mcp/config/research.py`

```python
# Before
deep_research_claim_verification_enabled: bool = False

# After
deep_research_claim_verification_enabled: bool = True
```

#### 3b. Change ResearchProfile default

**File:** `src/foundry_mcp/core/research/models/deep_research.py`

```python
# Before (in ResearchProfile class)
enable_claim_verification: bool = False

# After
enable_claim_verification: bool = True
```

#### 3c. Update built-in profiles

Since the ResearchProfile default is now `True`, built-in profiles that shouldn't run verification (if any) need an explicit `enable_claim_verification=False`. Review all 5 built-in profiles — all should default to True.

#### 3d. Tests

- Update any test that asserts `enable_claim_verification == False` as the default.
- Verify existing claim verification tests still pass (they should — the pipeline logic is unchanged, only the opt-in default flips).
- Verify TOML config override `deep_research_claim_verification_enabled = false` still disables it.

---

## Phase 4: Citation Year-as-Number Fix

**Goal:** `[2025]` and `[2026]` year references in report text should not be parsed as citation numbers.

### Changes

#### 4a. Add `max_citation` bound to `extract_cited_numbers()`

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/_citation_postprocess.py` (lines 29-41)

```python
def extract_cited_numbers(report: str, *, max_citation: int | None = None) -> set[int]:
    """Extract all citation numbers referenced in the report.

    Args:
        report: The markdown report text.
        max_citation: If provided, exclude numbers above this value.
            Pass ``len(state.sources)`` or ``state.next_citation_number``
            to filter out year references like [2025].

    Returns:
        Set of cited integer citation numbers.
    """
    numbers = {int(m.group(1)) for m in _CITATION_RE.finditer(report)}
    if max_citation is not None:
        numbers = {n for n in numbers if n <= max_citation}
    return numbers
```

#### 4b. Thread `max_citation` through call sites

**File:** `_citation_postprocess.py` — `postprocess_citations()` (line 290)

```python
# Before
cited_numbers = extract_cited_numbers(report)

# After
max_cn = max(valid_numbers) if valid_numbers else 0
cited_numbers = extract_cited_numbers(report, max_citation=max_cn)
```

Also update the second call at line 305 (after dangling removal recomputation).

**File:** `phases/claim_verification.py` — if `extract_cited_numbers` is used there, thread `max_citation` similarly.

#### 4c. Tests

- Unit test: `extract_cited_numbers("published in [2025] and updated [2026]", max_citation=61)` returns `set()`.
- Unit test: `extract_cited_numbers("[1] and [2] and [2025]", max_citation=61)` returns `{1, 2}`.
- Unit test: Without `max_citation`, behavior is unchanged (backward-compatible).
- Unit test: `postprocess_citations` no longer reports year references as dangling.

---

## Implementation Order

1. **Phase 4** (citation fix) — smallest, no dependencies, can land immediately
2. **Phase 1** (selective providers) — profile default change + keyword expansion
3. **Phase 3** (claim verification default) — simple default flip, but test after Phase 1 to avoid interaction issues
4. **Phase 2** (source relevance filtering) — largest change, depends on Phase 1 being stable

Phases 1 and 4 can be developed in parallel. Phase 3 is independent. Phase 2 builds on Phase 1.

---

## Files Touched (Summary)

| File | Phases |
|------|--------|
| `src/foundry_mcp/core/research/models/deep_research.py` | 1a, 3b |
| `src/foundry_mcp/core/research/models/sources.py` | 2a |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py` | 1b |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` | 2c |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py` | 2e |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/_citation_postprocess.py` | 4a, 4b |
| `src/foundry_mcp/core/research/workflows/deep_research/source_quality.py` | 2b |
| `src/foundry_mcp/config/research.py` | 2d, 3a |
| `tests/` (various) | 1c, 2f, 3d, 4c |
