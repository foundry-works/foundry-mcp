# Deep Research Post-Synthesis Remediation Checklist

## Phase 1: Fix provider leakage in gathering and supervision

- [ ] Update `gathering.py:340-344` to read `state.metadata["active_providers"]` with config fallback
- [ ] Update `supervision.py:1231-1235` to read `state.metadata["active_providers"]` with config fallback
- [ ] Change fallback default in both locations from `["tavily", "google", "semantic_scholar"]` to `["tavily"]`
- [ ] Add unit test: active_providers metadata respected by gathering phase
- [ ] Add unit test: active_providers metadata respected by supervision phase
- [ ] Add unit test: fallback to config when active_providers absent (backward compat)
- [ ] Verify existing gathering/supervision tests still pass

## Phase 2: Filter bibliography to cited-only sources

- [x] Update `_citation_postprocess.py:337` to pass `cited_only=True` and `cited_numbers=cited_numbers`
- [x] Add unit test: bibliography only contains sources cited in body text
- [x] Add unit test: APA format mode works with cited_only filtering
- [x] Add unit test: provenance/export still returns all sources regardless of citation status
- [x] Verify existing citation postprocess tests still pass

## Phase 3: Fix claim verification timeout on large reports

- [x] Add `max_tokens=16384` to extraction call in `claim_verification.py:781-787`
- [x] Add report truncation guard (30K char cap) before extraction prompt construction
- [x] Update `_build_extraction_user_prompt()` or add wrapper to accept truncated report
- [x] Update default `deep_research_claim_verification_timeout` from 120.0 to 180.0 in `config/research.py:335`
- [x] Update TOML defaults dict timeout value in `config/research.py:488`
- [x] Add unit test: extraction call receives max_tokens parameter
- [x] Add unit test: large reports are truncated before extraction
- [x] Add unit test: truncation preserves report body, drops bibliography first
- [x] Verify existing claim verification tests still pass

## Final Validation

- [x] Run full test suite (`pytest tests/`)
- [ ] Run contract tests (`pytest tests/contract/`)
- [ ] Smoke test: general-profile deep research session confirms only Tavily in provider stats
- [ ] Smoke test: report bibliography matches inline citations
- [ ] Smoke test: claim_verification object is populated on completion
