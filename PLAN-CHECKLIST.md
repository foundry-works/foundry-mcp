# PLAN-CHECKLIST: Deep Research Post-Synthesis Quality Improvements

## Phase 4: Citation Year-as-Number Fix
- [x] Add `max_citation: int | None = None` parameter to `extract_cited_numbers()` in `_citation_postprocess.py`
- [x] Filter out numbers > `max_citation` when parameter is provided
- [x] Update `postprocess_citations()` to pass `max(max(valid_numbers), 999)` as `max_citation` (both call sites: line ~290 and ~305)
- [x] Add `max_citation` parameter to `remove_dangling_citations()` to preserve year references during removal
- [x] Check `claim_verification.py` for any `extract_cited_numbers` usage — confirmed not used there
- [x] Add unit tests:
  - [x] Year references excluded when `max_citation` is set
  - [x] Mixed citations + years correctly filtered
  - [x] Boundary value (equal to max_citation kept, above excluded)
  - [x] Without `max_citation`, behavior unchanged (backward-compatible)
  - [x] `postprocess_citations` no longer reports years as dangling

## Phase 1: Selective Academic Provider Usage

### 1a. Profile default change
- [ ] Change `PROFILE_GENERAL.providers` from `["tavily", "semantic_scholar"]` to `["tavily"]` in `deep_research.py`
- [ ] Verify `PROFILE_TECHNICAL` already excludes academic providers (currently `["tavily", "google"]` — no change needed)
- [ ] Verify `PROFILE_ACADEMIC`, `PROFILE_SYSTEMATIC_REVIEW`, `PROFILE_BIBLIOMETRIC` still include academic providers (no change needed)

### 1b. Expand discipline keyword map
- [ ] Add physics/chemistry/biology/ecology/environmental science group → `semantic_scholar`
- [ ] Add mathematics/statistics/operations research group → `semantic_scholar`
- [ ] Add psychology/cognitive science/neuroscience group → `semantic_scholar`
- [ ] Add engineering/robotics/materials science group → `semantic_scholar`
- [ ] Add law/jurisprudence/policy analysis group → `openalex`
- [ ] Add history/philosophy/literature review/systematic review group → `openalex`

### 1c. Tests
- [ ] Unit test: `PROFILE_GENERAL.providers` does not contain `semantic_scholar` or `openalex`
- [ ] Unit test: `_extract_provider_hints` returns `["semantic_scholar"]` for academic brief text
- [ ] Unit test: `_extract_provider_hints` returns `[]` for consumer/general brief text
- [ ] Unit test: `_apply_provider_hints` adds `semantic_scholar` to general profile when hint present
- [ ] Unit test: `_apply_provider_hints` does NOT modify custom profiles

## Phase 3: Default Claim Verification to True

### 3a. Config default
- [ ] Change `deep_research_claim_verification_enabled` default from `False` to `True` in `research.py`

### 3b. ResearchProfile default
- [ ] Change `enable_claim_verification` default from `False` to `True` in `ResearchProfile` class

### 3c. Built-in profiles
- [ ] Review all 5 built-in profiles — confirm all should have claim verification enabled
- [ ] Add explicit `enable_claim_verification=True` to any profile that currently inherits the default (for clarity)

### 3d. Tests
- [ ] Update tests asserting `enable_claim_verification == False` as default
- [ ] Verify TOML config `deep_research_claim_verification_enabled = false` still disables
- [ ] Run existing claim verification test suite — confirm no regressions

## Phase 2: Source Relevance Filtering

### 2a. Data model
- [ ] Add `relevance_score: float | None = None` field to `ResearchSource` in `sources.py`
- [ ] Verify serialization round-trip (JSON/dict) preserves the field

### 2b. Relevance scoring function
- [ ] Implement `compute_source_relevance()` in `source_quality.py`
- [ ] Tokenization: lowercase, strip stopwords, extract keyword sets
- [ ] Scoring: weighted Jaccard similarity (title 0.7 weight, content 0.3 weight)
- [ ] Academic penalty: multiply raw score by 0.7 for `source_type == "academic"`
- [ ] Return clamped `[0.0, 1.0]` score

### 2c. Integration into source collection
- [ ] Add relevance scoring call in `_dedup_and_add_source()` Phase 3 (after quality scoring, before `state.append_source`)
- [ ] Build reference text from `sub_query.query` + `state.research_brief`
- [ ] Cap source content to 2000 chars for performance
- [ ] Log relevance scores at DEBUG level

### 2d. Config option
- [ ] Add `deep_research_source_relevance_threshold: float = 0.05` to `ResearchConfig`
- [ ] Add validation: `0.0 <= threshold <= 1.0`
- [ ] Document: set to `0.0` to disable filtering

### 2e. Compression integration
- [ ] Exclude sources with `relevance_score < threshold` from compression input
- [ ] Log excluded source count and IDs at INFO level
- [ ] Ensure excluded sources remain in `state.sources` (provenance preserved)

### 2f. Tests
- [ ] Unit test: irrelevant academic paper scores < 0.1 against consumer query
- [ ] Unit test: relevant web source scores > 0.5 against matching query
- [ ] Unit test: academic sources score lower than web sources with same keywords
- [ ] Unit test: `relevance_score=None` sources pass through unfiltered (backward-compatible)
- [ ] Unit test: sources below threshold excluded from compression input
- [ ] Unit test: excluded sources still present in `state.sources`
- [ ] Integration test: end-to-end flow with irrelevant sources correctly deprioritized

## Final Validation
- [ ] Run full test suite: `pytest tests/ -x`
- [ ] Run deep research contract tests
- [ ] Manual smoke test: run a general-profile consumer query and verify no academic noise
- [ ] Manual smoke test: run an academic-profile query and verify academic providers activate
- [ ] Verify claim verification runs by default on new sessions
