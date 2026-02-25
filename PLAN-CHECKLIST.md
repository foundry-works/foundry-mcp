# Deep Research ODR Alignment — Checklist

Cross-reference: [PLAN.md](PLAN.md) for detailed rationale, ODR patterns, and code references.

All four phases are independent and can be developed in parallel.

---

## Phase 1: Extract-Content Visibility in ReAct Loop

- [ ] **1a** Summarize extracted content and return to researcher
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`
  - Location: `_handle_extract_tool()` (~lines 1092-1154)
  - After successful extraction, call `_summarize_search_results()` on newly extracted sources
  - Replace confirmation-only response with `_format_search_results_batch()` output
  - Skip summarization for sources with content < 300 chars
  - Preserve confirmation prefix ("Extracted content from N of M URL(s).")
  - Test: Researcher message history shows formatted source blocks after extract calls
  - Test: Sources with short content (<300 chars) skip summarization

- [ ] **1b** Apply novelty scoring to extracted sources
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`
  - Location: `_handle_extract_tool()` after summarization
  - Call `compute_novelty_tag()` on extracted sources (same as search results)
  - Include novelty tags in formatted output
  - Test: Extracted sources get NEW/RELATED/DUPLICATE tags

- [ ] **1c** Guard against redundant summarization
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`
  - Location: `_handle_extract_tool()`
  - Skip summarization for sources that were already in `state.sources` before extraction
  - Only format newly-added sources in the response message
  - Test: Re-extracting an already-known URL doesn't produce duplicate formatted blocks

---

## Phase 2: Supervisor-Optimized Research Summaries

- [ ] **2a** Add `supervisor_summary` field to TopicResearchResult
  - File: `src/foundry_mcp/core/research/models/deep_research.py`
  - Location: `TopicResearchResult` class (~lines 27-121)
  - New field: `supervisor_summary: Optional[str]` with description
  - Test: Field serializes/deserializes correctly in persistence

- [ ] **2b** Generate supervisor_summary during compression
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`
  - Location: Compression system prompt and response parsing
  - Append "## SUPERVISOR BRIEF" instruction block to compression prompt
  - Parse response to split `compressed_findings` from `supervisor_summary`
  - Graceful fallback: if section missing, leave supervisor_summary as None
  - Store both on TopicResearchResult
  - Test: Compression output contains SUPERVISOR BRIEF section
  - Test: Fallback works when model doesn't produce the section

- [ ] **2c** Use supervisor_summary in coverage data
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
  - Location: `_build_per_query_coverage()` (~lines 2516-2600)
  - Prefer `supervisor_summary` over `compressed_findings[:2000]` when available
  - Fall back to current truncation when supervisor_summary is None
  - Test: Coverage data uses structured summary instead of raw truncation
  - Test: Resume compatibility — older sessions without supervisor_summary still work

- [ ] **2d** Use supervisor_summary in evidence inventory
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
  - Location: `_build_evidence_inventory()` (~lines 1235-1325)
  - Include key findings from supervisor_summary in evidence inventory
  - Stay within existing 500-char budget
  - Test: Evidence inventory includes findings when supervisor_summary available
  - Test: Evidence inventory unchanged when supervisor_summary is None

---

## Phase 3: Native Search Provider Support

- [ ] **3a** Implement AnthropicNativeSearchProvider
  - File: `src/foundry_mcp/core/research/providers/anthropic_native.py` (new)
  - Extends `SearchProvider` ABC
  - Implements `search(query, max_results)` via Anthropic Messages API with `web_search_20250305`
  - Parses response citations into `ResearchSource` objects
  - Provider name: `"anthropic_native"`
  - Implements `health_check()` and `classify_error()`
  - Test: Returns valid ResearchSource list for a known query
  - Test: Handles rate limit (429) and auth (401) errors correctly
  - Test: Health check returns False when API key is missing

- [ ] **3b** Implement OpenAINativeSearchProvider
  - File: `src/foundry_mcp/core/research/providers/openai_native.py` (new)
  - Extends `SearchProvider` ABC
  - Implements `search(query, max_results)` via OpenAI API with `web_search_preview`
  - Parses `tool_outputs` for web_search_call results into `ResearchSource` objects
  - Provider name: `"openai_native"`
  - Implements `health_check()` and `classify_error()`
  - Test: Returns valid ResearchSource list for a known query
  - Test: Handles rate limit and auth errors correctly
  - Test: Health check returns False when API key is missing

- [ ] **3c** Register providers and add config
  - File: `src/foundry_mcp/core/research/providers/__init__.py` — export new providers
  - File: `src/foundry_mcp/config/research.py` — add `"anthropic_native"` and `"openai_native"` as valid provider values
  - File: `src/foundry_mcp/core/research/providers/resilience/config.py` — add default resilience configs (1.0 RPS, burst=3, circuit_failure_threshold=5)
  - Test: Both providers can be instantiated via provider registry
  - Test: Resilience wrappers (rate limit, circuit breaker) apply to new providers

- [ ] **3d** Add health check and error classification
  - Files: `anthropic_native.py`, `openai_native.py`
  - Map HTTP 429 → rate_limit, 401/403 → auth, 5xx → server_error
  - Map model-specific errors (tool unavailable) → provider_error
  - Test: Error classification returns expected ErrorType for each status code

---

## Phase 4: Evaluation Dimensions — Practical Value & Balance

- [ ] **4a** Add PRACTICAL_VALUE dimension
  - File: `src/foundry_mcp/core/research/evaluation/dimensions.py`
  - Add `EvaluationDimension` with name="practical_value", 1-5 rubric for actionability
  - Add to `DIMENSIONS` tuple
  - Test: `DIMENSION_BY_NAME["practical_value"]` resolves correctly
  - Test: `len(DIMENSIONS) == 8`

- [ ] **4b** Add BALANCE dimension
  - File: `src/foundry_mcp/core/research/evaluation/dimensions.py`
  - Add `EvaluationDimension` with name="balance", 1-5 rubric for multi-perspective objectivity
  - Add to `DIMENSIONS` tuple
  - Test: `DIMENSION_BY_NAME["balance"]` resolves correctly

- [ ] **4c** Verify composite scoring handles 8 dimensions
  - File: `src/foundry_mcp/core/research/evaluation/scoring.py`
  - No code change expected — `compute_composite` uses `1/len(dimension_scores)`
  - Verify: equal weight becomes 1/8 = 0.125 per dimension
  - Verify: `_parse_evaluation_response` gracefully handles missing new dimensions (defaults to score 3)
  - Test: Composite score with 8 dimensions matches expected weighted average
  - Test: Composite score with 6 returned + 2 missing dimensions still computes correctly

- [ ] **4d** Verify evaluation prompt auto-includes new dimensions
  - File: `src/foundry_mcp/core/research/evaluation/evaluator.py`
  - Location: `_build_evaluation_prompt()` — confirm rubric section is built dynamically from DIMENSIONS
  - Confirm JSON schema section lists all 8 dimension names
  - Test: Run evaluation on a completed research session — all 8 dimensions scored
  - Test: Evaluation response JSON contains practical_value and balance keys
