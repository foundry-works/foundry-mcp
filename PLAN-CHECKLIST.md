# Deep Research Timeout & Zero-Source Fix — Checklist

## Phase 1: Cap Timeout Retries

- [x] **1.1** Add `max_timeout_retries: int = 0` parameter to `_execute_provider_async` in `base.py`
- [x] **1.2** In the `ProviderTimeoutError` handler (`base.py:479-497`), check `timeout_retries_used < max_timeout_retries` instead of `attempt < max_retries`; track `timeout_retries_used` separately
- [x] **1.3** In the `asyncio.TimeoutError` handler (`base.py:499-517`), apply the same timeout retry cap
- [x] **1.4** Pass `max_timeout_retries=0` from `execute_llm_call` in `_lifecycle.py` (all deep research LLM calls get the new behavior)
- [x] **1.5** Add test: verify timeout errors break immediately (no retries) when `max_timeout_retries=0`
- [x] **1.6** Add test: verify non-timeout errors still retry up to `max_retries`
- [x] **1.7** Add test: verify `max_timeout_retries=1` allows exactly one timeout retry

## Phase 2: Zero-Source Synthesis Guard

- [ ] **2.1** In `workflow_execution.py`, before the synthesis `_run_phase` call, check `len(state.sources) == 0`
- [ ] **2.2** When zero sources: log warning, emit `zero_source_synthesis_warning` audit event, set `state.metadata["ungrounded_synthesis"] = True`
- [ ] **2.3** After synthesis completes with `ungrounded_synthesis=True`, prepend disclaimer to `state.report`
- [ ] **2.4** Re-save the report markdown file (if `report_output_path` is set) after prepending disclaimer
- [ ] **2.5** Add test: zero-source state triggers warning metadata and disclaimer in report
- [ ] **2.6** Add test: non-zero-source state does NOT trigger warning or disclaimer

## Phase 3: Topic Research Timeout Config

- [ ] **3.1** Add `deep_research_topic_research_timeout: float = 90.0` to `ResearchConfig` dataclass in `research.py`
- [ ] **3.2** Add parsing in `ResearchConfig.from_dict()` for the new field
- [ ] **3.3** Add to `_FLOAT_FIELDS` list for TOML config merging
- [ ] **3.4** In `_execute_researcher_llm_call` (`topic_research.py:966`), replace `self.config.deep_research_reflection_timeout` with `self.config.deep_research_topic_research_timeout`
- [ ] **3.5** Update `samples/foundry-mcp.toml` with the new config key and documentation comment
- [ ] **3.6** Add test: verify topic research uses the new timeout config value

## Verification

- [ ] **V.1** Run existing test suites: `pytest tests/core/research/workflows/test_timeout_resilience.py -x`
- [ ] **V.2** Run existing test suites: `pytest tests/core/research/workflows/test_deep_research.py -x`
- [ ] **V.3** Run full deep research test directory: `pytest tests/core/research/workflows/deep_research/ -x`
- [ ] **V.4** Manual smoke test: run a deep research session with Tavily API key unset (forces timeout/failure) and verify zero-source disclaimer appears
- [ ] **V.5** Verify no regressions in existing timeout/resilience behavior for non-deep-research workflows
