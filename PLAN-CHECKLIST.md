# Deep Research ODR Alignment — Checklist

Cross-reference: [PLAN.md](PLAN.md) for detailed rationale and file references.

## Phase 1: Cost-Tiered Model Defaults

- [ ] **1a** Set `summarization_provider`/`summarization_model` defaults in `ModelRoleConfig` to a cheap-tier model
  - File: `src/foundry_mcp/config/research_sub_configs.py`
  - Verify: `SourceSummarizer` resolves to cheap model when no explicit override is set
  - Test: Unit test that `safe_resolve_model_for_role(config, "summarization")` returns cheap default

- [ ] **1b** Evaluate and optionally set `compression_provider`/`compression_model` defaults to cheap tier
  - File: `src/foundry_mcp/config/research_sub_configs.py`
  - Decision: Run compression quality comparison (cheap vs capable model) on 3-5 sample topics
  - If quality holds: set cheap default. If not: keep at research tier, document why.

- [ ] **1c** Document cost-tier model strategy
  - File: `dev_docs/guides/` (new guide or section)
  - Cover: which roles are cheap-eligible, example configs, cost/quality trade-offs

## Phase 2: Supervisor Delegation Scaling Heuristics

- [x] **2a** Add query-type scaling rules to `_build_delegation_system_prompt()`
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py:1204`
  - Rules: simple→1-2, comparison→1-per-element, complex→3-5
  - Verify: prompt changes don't break existing test fixtures

- [x] **2b** Add complexity signal to `_build_delegation_user_prompt()`
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py:1241`
  - Heuristic: sub-query count + brief length → simple/moderate/complex label
  - Test: Unit test for complexity classification function

## Phase 3: Researcher Tool-Call Parse Resilience

- [x] **3a** Add retry-on-parse-failure to researcher ReAct loop
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`
  - Max retries: 2 (matching ODR's `stop_after_attempt=3` total)
  - Retry prompt suffix: clarify expected JSON format
  - Test: Unit test with mock LLM that returns invalid JSON on first call, valid on retry

- [x] **3b** Add `tool_parse_failures` counter to topic research metrics
  - File: `src/foundry_mcp/core/research/models/deep_research.py`
  - Expose in audit artifacts for observability

## Phase 4: Search Result Presentation Format

- [ ] **4a** Standardize numbered-source format in researcher search tool responses
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`
  - Format: `--- SOURCE N: Title ---\nURL: ...\nNOVELTY: [tag]\nSUMMARY: ...\nKEY EXCERPTS: ...`
  - Test: Snapshot test of formatted search result output

- [ ] **4b** Add batch header to search results (count, domains, novelty summary)
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`
  - Format: `Found N results from M domains. N new, M related.`

## Phase 5: Compression Token-Limit Retry Strategy

- [ ] **5a** Implement message-boundary-aware truncation for compression retries
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`
  - Strategy: drop oldest complete messages first, preserve last 2 think + latest search results
  - Test: Unit test with oversized message history that triggers retry, verify newest messages retained

- [ ] **5b** Record truncation metadata in compression output
  - Files: `compression.py`, `models/deep_research.py`
  - Fields: `messages_dropped`, `retry_count`, `original_message_count`
  - Verify: fidelity tracking picks up the new fields
