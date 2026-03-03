# Remediation Checklist: Alpha Branch Review

## P0: Critical — Silent Config Data Loss

- [x] Add 10 missing claim verification fields to `from_toml_dict()` constructor in `config/research.py`
  - [x] `deep_research_claim_verification_enabled`
  - [x] `deep_research_claim_verification_sample_rate`
  - [x] `deep_research_claim_verification_provider`
  - [x] `deep_research_claim_verification_model`
  - [x] `deep_research_claim_verification_timeout`
  - [x] `deep_research_claim_verification_max_claims`
  - [x] `deep_research_claim_verification_max_concurrent`
  - [x] `deep_research_claim_verification_max_corrections`
  - [x] `deep_research_claim_verification_annotate_unsupported`
  - [x] `deep_research_claim_verification_max_input_tokens`
- [x] Add `_validate_claim_verification_config()` to `__post_init__` in `config/research.py`
  - [x] Validate `sample_rate` in `[0.0, 1.0]`
  - [x] Validate `timeout` > 0
  - [x] Validate `max_claims` >= 1
  - [x] Validate `max_concurrent` >= 1
  - [x] Validate `max_corrections` >= 0
  - [x] Validate `max_input_tokens` > 0
- [x] Add claim verification fields to `DeepResearchSettings` in `config/research_sub_configs.py`
- [x] Add regression test: assert all `dataclasses.fields(ResearchConfig)` names appear in `from_toml_dict`
- [x] Fix timeout type: `deep_research_claim_verification_timeout` from `int` to `float`
- [x] Fix timeout type: `deep_research_summarization_timeout` from `int` to `float`

## P1: High — Concurrency & Error Handling

- [x] Change `asyncio.gather` to `return_exceptions=True` in `claim_verification.py:496`
  - [x] Handle exception instances in result processing loop
- [x] Snapshot dict iteration in `workflow_execution.py:662`: `list(self._active_sessions.items())`
- [x] Add comment documenting shared-state constraint at daemon thread entry point in `background_tasks.py`

## P2: Medium — Correctness & Security

- [x] Validate OpenAlex reference/related IDs against `_OPENALEX_WORK_ID_RE` before filter join (`openalex.py:372-377, 406-411`)
- [x] Route SIGTERM handler through public `cancel()` API (`infrastructure.py:192`)
- [x] Add `claims_partially_supported` counter to `ClaimVerificationResult` (`claim_verification.py:808-815`)
- [x] Pass `structured_blocks` to `_compression_output_is_valid` in inline compression path (`topic_research.py:1543-1547`)
- [x] Fix `budget_truncate_text` to return empty string when budget <= 0 (`_token_budget.py:77-82`)

## P3: Low — Hygiene & Hardening

### Repo cleanup
- [x] Delete `conversation_assessment_research.md` and add gitignore pattern
- [x] Delete `dev_docs/plans/specs/.autonomy/context/context.json` and add `.autonomy/` to `.gitignore`

### CI
- [x] Update `softprops/action-gh-release@v1` to `@v2` or pin to SHA (`.github/workflows/publish-alpha.yml`)
- [x] Narrow tag pattern from `v*a*` to `v*a[0-9]*` (`.github/workflows/publish-alpha.yml`)

### Code quality
- [x] Remove `TypeError` from catch clause in `parse_researcher_response` (`models/deep_research.py:702-706`)
- [x] Remove redundant `import time` in `background_tasks.py:281`
- [x] Change f-string logger calls to `%s` formatting in `docx_extractor.py:421, 471-474, 519`
- [x] Change `int(raw_score)` to `round(raw_score)` in `evaluator.py:206`

## Verification

- [x] Run full test suite: `python -m pytest tests/ -x --timeout=60`
- [x] Run config tests specifically: `python -m pytest tests/unit/test_config_*.py -v`
- [x] Verify no new test failures introduced
