# Deep Research Enhancements — Implementation Checklist

Companion to [PLAN.md](PLAN.md). Each item is a discrete, testable unit of work.
Mark items `[x]` as completed. Items within a phase are ordered by dependency.

---

## Phase 5: Progressive Token-Limit Recovery (independent — do first)

- [x] **5.1** Add `TOKEN_LIMITS` registry to `providers/base.py` mapping known model names → context window sizes
- [x] **5.2** Add `truncate_to_token_estimate(text, max_tokens) -> str` to `_helpers.py`
- [x] **5.3** Add provider-specific error detection patterns to `ContextWindowError` classification in `_lifecycle.py`
  - [x] OpenAI: `BadRequestError` + "token/context/length" keywords
  - [x] Anthropic: `BadRequestError` + "prompt is too long"
  - [x] Google: `ResourceExhausted` exception type
- [x] **5.4** Enhance `execute_llm_call()` in `_lifecycle.py` with retry loop:
  - [x] On `ContextWindowError`, estimate limit from `TOKEN_LIMITS`
  - [x] Truncate user prompt by 10% per attempt
  - [x] Retry up to 3 times
  - [x] Track retries in `PhaseMetrics.metadata["token_limit_retries"]`
  - [x] Fall back to existing hard-error if all retries fail
- [x] **5.5** Write `tests/research/test_token_limit_recovery.py` (39 tests passing):
  - [x] Test progressive truncation with mock provider errors
  - [x] Test fallback to hard error after 3 retries
  - [x] Test provider-specific error detection (OpenAI, Anthropic, Google)
  - [x] Test that system prompt is never truncated
- [x] **5.6** Verify existing deep research contract tests still pass

---

## Phase 1: Fetch-Time Source Summarization

- [x] **1.1** Add `raw_content: Optional[str]` field to `ResearchSource` in `models/sources.py`
  - [x] Ensure Pydantic serialization backward-compat (default `None`, excluded from compact repr)
- [x] **1.2** Add `summarization_model` and `summarization_provider` to `config.py`
  - [x] Sensible defaults (cheapest available or same as research if unset)
  - [x] Document in config schema
- [x] **1.3** Create `SourceSummarizer` in `providers/shared.py`:
  - [x] Input: raw content string + model config
  - [x] Output: `SummarizationResult(executive_summary: str, key_excerpts: list[str])`
  - [x] Prompt: extract executive summary (25-30% of original) + up to 5 verbatim key excerpts
  - [x] 60-second timeout per source with fallback to original content
  - [x] Token tracking for summarization calls
- [x] **1.4** Add optional `summarizer` hook to `SearchProvider` base in `providers/base.py`
  - [x] Default: no-op (returns content unchanged)
  - [x] Providers opt in by setting summarizer during initialization
- [x] **1.5** Wire summarization into `TavilySearchProvider.search()`:
  - [x] After raw results returned, run `SourceSummarizer` on each source's content
  - [x] Parallel execution bounded by `max_concurrent`
  - [x] Store original in `raw_content`, summary in `content`
  - [x] Store excerpts in `metadata["excerpts"]`
  - [x] Gate behind `config.fetch_time_summarization` flag (default `True`)
- [x] **1.6** Add `fetch_time_summarization: bool = True` to `config.py`
- [x] **1.7** Track summarization tokens separately in `PhaseMetrics` (tracked via source metadata `summarization_input_tokens`/`summarization_output_tokens`)
- [x] **1.8** Write `tests/research/test_source_summarization.py`:
  - [x] Test `SourceSummarizer` with mock LLM responses
  - [x] Test timeout fallback to original content
  - [x] Test parallel summarization with multiple sources
  - [x] Test opt-out via config flag
  - [x] Test `raw_content` preservation
  - [x] Test backward-compat: existing `ResearchSource` without `raw_content` deserializes
- [x] **1.9** Verify existing deep research contract tests still pass

---

## Phase 2: Forced Reflection in Topic Research

- [x] **2.1** Add `parse_reflection_decision(text) -> ReflectionDecision` to `_helpers.py`:
  - [x] Structured extraction: `{continue_searching: bool, refined_query: str|null, research_complete: bool, rationale: str}`
  - [x] Fallback parsing if JSON extraction fails (regex for key fields)
- [x] **2.2** Update reflection prompt in `_topic_reflect()`:
  - [x] Include current source count and quality distribution
  - [x] Explicit sufficiency criteria ("3+ relevant sources from distinct domains = likely sufficient")
  - [x] Add `research_complete` option to signal early exit
  - [x] Request structured JSON output
- [x] **2.3** Restructure `_execute_topic_research_async` loop in `topic_research.py`:
  - [x] Make reflection mandatory after every search (not conditional on source count)
  - [x] Parse reflection response via `parse_reflection_decision()`
  - [x] Exit loop early if `research_complete=True`
  - [x] Exit loop if `continue_searching=False`
  - [x] Respect `max_searches` as hard cap regardless of reflection decision
- [x] **2.4** Update `TopicResearchResult` to include `early_completion: bool` and `completion_rationale: str`
- [x] **2.5** Write `tests/research/test_topic_reflection.py`:
  - [x] Test mandatory reflection after each search iteration
  - [x] Test early exit on `research_complete=True`
  - [x] Test `continue_searching=False` behavior
  - [x] Test hard cap `max_searches` override of reflection decision
  - [x] Test fallback parsing when JSON extraction fails
  - [x] Test reflection prompt includes source count and quality info
- [x] **2.6** Verify existing deep research contract tests still pass

---

## Phase 3: Per-Topic Compression Before Aggregation

- [x] **3.1** Add `compressed_findings: Optional[str]` to `TopicResearchResult` in `models/deep_research.py`
  - [x] Ensure Pydantic serialization backward-compat
- [x] **3.2** Add `compression_model` and `compression_provider` to `config.py`
  - [x] Default: same as research model/provider
- [x] **3.3** Implement per-topic compression step in `phases/gathering.py`:
  - [x] After all topic researchers complete, iterate `TopicResearchResult` list
  - [x] For each topic: collect its sources from `state.sources` by `sub_query_id`
  - [x] Build compression prompt: "Reformat findings with inline citations [1], [2]. DO NOT summarize. Preserve all relevant information."
  - [x] Call LLM with compression provider/model
  - [x] Store result in `TopicResearchResult.compressed_findings`
  - [x] Progressive token-limit handling (3 retries, 10% truncation — reuse Phase 5 infra)
  - [x] Fallback: if compression fails, leave `compressed_findings=None` (analysis uses raw sources)
- [x] **3.4** Parallel compression across topics, bounded by `max_concurrent`
- [x] **3.5** Track compression tokens in `PhaseMetrics`
- [x] **3.6** Update `phases/analysis.py` to prefer `compressed_findings` when available:
  - [x] If all topics have `compressed_findings`, use those as primary analysis input
  - [x] Adjust budget allocation to account for pre-compressed content
  - [x] Fall through to existing raw-source analysis when `compressed_findings` is `None`
- [x] **3.7** Write `tests/research/test_topic_compression.py` (33 tests passing):
  - [x] Test compression prompt includes correct sources per topic
  - [x] Test progressive truncation on token limit error
  - [x] Test fallback to raw sources when compression fails
  - [x] Test analysis phase uses compressed findings when available
  - [x] Test analysis phase falls back when compressed findings absent
  - [x] Test parallel compression across multiple topics
  - [x] Test citation numbering consistency between compression and analysis
- [x] **3.8** Verify existing deep research contract tests still pass

---

## Phase 4: Structured Clarification Gate

- [x] **4.1** Define `ClarificationDecision` schema in `_helpers.py`:
  - [x] `{need_clarification: bool, question: str, verification: str}`
  - [x] Extraction via `extract_json()` with retry-on-parse-failure (3 attempts)
- [x] **4.2** Add `execute_structured_llm_call()` variant to `phases/_lifecycle.py`:
  - [x] Accepts expected JSON schema
  - [x] Requests JSON-mode output from provider
  - [x] Validates response against schema
  - [x] Retries up to 3 times on validation failure
  - [x] Falls back to unstructured call on exhaustion
- [x] **4.3** Update `phases/clarification.py`:
  - [x] Use `execute_structured_llm_call()` with `ClarificationDecision` schema
  - [x] On `need_clarification=True`: return question (existing flow)
  - [x] On `need_clarification=False`: store `verification` in `state.clarification_constraints`, log audit event, proceed to planning
  - [x] On parse failure fallback: treat as "no clarification needed" (existing behavior)
- [x] **4.4** Write `tests/research/test_clarification_structured.py`:
  - [x] Test structured output parsing for both `need_clarification` values
  - [x] Test `verification` stored in state
  - [x] Test retry on parse failure
  - [x] Test fallback to unstructured on exhaustion
  - [x] Test audit event logged on verification
- [x] **4.5** Verify existing deep research contract tests still pass

---

## Phase 6: Multi-Model Cost Optimization

- [x] **6.1** Formalize model role hierarchy in `config.py`:
  - [x] `research_model` / `research_provider` — analysis, planning, clarification (strongest)
  - [x] `summarization_model` / `summarization_provider` — fetch-time (cheapest)
  - [x] `compression_model` / `compression_provider` — per-topic compression
  - [x] `reflection_model` / `reflection_provider` — think-tool pauses
  - [x] `report_model` / `report_provider` — final synthesis
  - [x] All default to `None` (falls back to phase-level, then global default)
- [x] **6.2** Add `resolve_model_for_role(role: str) -> tuple[str, str]` to config:
  - [x] Resolution chain: role-specific → phase-level → global default
  - [x] Returns `(provider_id, model)`
- [x] **6.3** Add `role: Optional[str]` parameter to `execute_llm_call()` in `_lifecycle.py`:
  - [x] When `role` is provided, resolve provider/model from config via `resolve_model_for_role()`
  - [x] Explicit `provider_id`/`model` parameters still override role-based resolution
- [x] **6.4** Update all phase callsites to pass appropriate roles:
  - [x] `gathering.py` (summarization calls) → `role="summarization"`
  - [x] `gathering.py` (compression calls) → `role="compression"`
  - [x] `topic_research.py` (reflection calls) → `role="topic_reflection"`
  - [x] `analysis.py` (finding extraction) → `role="research"`
  - [x] `synthesis.py` (report generation) → `role="report"`
  - [x] `clarification.py` (structured decision) → `role="clarification"`
  - [x] `planning.py` (sub-query generation) → `role="research"`
- [x] **6.5** Add cost tracking per role in `PhaseMetrics.metadata["role"]`:
  - [x] Role stored per-call in PhaseMetrics metadata
  - [x] `get_model_role_costs()` aggregation on DeepResearchState: `{role: {provider, model, input_tokens, output_tokens, calls}}`
- [x] **6.6** Update `server(action="capabilities")` to include model roles in response
- [x] **6.7** Write `tests/core/research/workflows/test_model_routing.py`:
  - [x] Test role resolution chain (role-specific → phase → global)
  - [x] Test explicit provider/model overrides role-based resolution
  - [x] Test cost tracking per role
  - [x] Test backward-compat when no role-specific config provided
  - [x] Test all phase callsites pass expected roles (32 tests)
- [x] **6.8** Verify existing deep research contract tests still pass (1759 passed, 6 skipped)

---

## Cross-Cutting Validation

- [ ] **V.1** Run full test suite (`pytest tests/`) — all existing tests pass
- [ ] **V.2** Run contract tests (`pytest tests/contract/`) — envelope schemas valid
- [ ] **V.3** Manual end-to-end test: run a deep research session with all features enabled
- [ ] **V.4** Compare token usage before/after on a reference query (document in PR)
- [ ] **V.5** Verify backward-compat: load a pre-existing saved research session, confirm it deserializes and can be resumed
- [ ] **V.6** Review all new config fields have sensible defaults and documentation
