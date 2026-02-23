# Deep Research Enhancements — Implementation Checklist

Companion to [PLAN.md](PLAN.md). Each item is a discrete, testable unit of work.
Mark items `[x]` as completed. Items within a phase are ordered by dependency.

---

## Phase 5: Progressive Token-Limit Recovery (independent — do first)

- [ ] **5.1** Add `TOKEN_LIMITS` registry to `providers/base.py` mapping known model names → context window sizes
- [ ] **5.2** Add `truncate_to_token_estimate(text, max_tokens) -> str` to `_helpers.py`
- [ ] **5.3** Add provider-specific error detection patterns to `ContextWindowError` classification in `_lifecycle.py`
  - [ ] OpenAI: `BadRequestError` + "token/context/length" keywords
  - [ ] Anthropic: `BadRequestError` + "prompt is too long"
  - [ ] Google: `ResourceExhausted` exception type
- [ ] **5.4** Enhance `execute_llm_call()` in `_lifecycle.py` with retry loop:
  - [ ] On `ContextWindowError`, estimate limit from `TOKEN_LIMITS`
  - [ ] Truncate user prompt by 10% per attempt
  - [ ] Retry up to 3 times
  - [ ] Track retries in `PhaseMetrics.metadata["token_limit_retries"]`
  - [ ] Fall back to existing hard-error if all retries fail
- [ ] **5.5** Write `tests/research/test_token_limit_recovery.py`:
  - [ ] Test progressive truncation with mock provider errors
  - [ ] Test fallback to hard error after 3 retries
  - [ ] Test provider-specific error detection (OpenAI, Anthropic, Google)
  - [ ] Test that system prompt is never truncated
- [ ] **5.6** Verify existing deep research contract tests still pass

---

## Phase 1: Fetch-Time Source Summarization

- [ ] **1.1** Add `raw_content: Optional[str]` field to `ResearchSource` in `models/sources.py`
  - [ ] Ensure Pydantic serialization backward-compat (default `None`, excluded from compact repr)
- [ ] **1.2** Add `summarization_model` and `summarization_provider` to `config.py`
  - [ ] Sensible defaults (cheapest available or same as research if unset)
  - [ ] Document in config schema
- [ ] **1.3** Create `SourceSummarizer` in `providers/shared.py`:
  - [ ] Input: raw content string + model config
  - [ ] Output: `SummarizationResult(executive_summary: str, key_excerpts: list[str])`
  - [ ] Prompt: extract executive summary (25-30% of original) + up to 5 verbatim key excerpts
  - [ ] 60-second timeout per source with fallback to original content
  - [ ] Token tracking for summarization calls
- [ ] **1.4** Add optional `summarizer` hook to `SearchProvider` base in `providers/base.py`
  - [ ] Default: no-op (returns content unchanged)
  - [ ] Providers opt in by setting summarizer during initialization
- [ ] **1.5** Wire summarization into `TavilySearchProvider.search()`:
  - [ ] After raw results returned, run `SourceSummarizer` on each source's content
  - [ ] Parallel execution bounded by `max_concurrent`
  - [ ] Store original in `raw_content`, summary in `content`
  - [ ] Store excerpts in `metadata["excerpts"]`
  - [ ] Gate behind `config.fetch_time_summarization` flag (default `True`)
- [ ] **1.6** Add `fetch_time_summarization: bool = True` to `config.py`
- [ ] **1.7** Track summarization tokens separately in `PhaseMetrics` (add `summarization_tokens` field or use metadata)
- [ ] **1.8** Write `tests/research/test_source_summarization.py`:
  - [ ] Test `SourceSummarizer` with mock LLM responses
  - [ ] Test timeout fallback to original content
  - [ ] Test parallel summarization with multiple sources
  - [ ] Test opt-out via config flag
  - [ ] Test `raw_content` preservation
  - [ ] Test backward-compat: existing `ResearchSource` without `raw_content` deserializes
- [ ] **1.9** Verify existing deep research contract tests still pass

---

## Phase 2: Forced Reflection in Topic Research

- [ ] **2.1** Add `parse_reflection_decision(text) -> ReflectionDecision` to `_helpers.py`:
  - [ ] Structured extraction: `{continue_searching: bool, refined_query: str|null, research_complete: bool, rationale: str}`
  - [ ] Fallback parsing if JSON extraction fails (regex for key fields)
- [ ] **2.2** Update reflection prompt in `_topic_reflect()`:
  - [ ] Include current source count and quality distribution
  - [ ] Explicit sufficiency criteria ("3+ relevant sources from distinct domains = likely sufficient")
  - [ ] Add `research_complete` option to signal early exit
  - [ ] Request structured JSON output
- [ ] **2.3** Restructure `_execute_topic_research_async` loop in `topic_research.py`:
  - [ ] Make reflection mandatory after every search (not conditional on source count)
  - [ ] Parse reflection response via `parse_reflection_decision()`
  - [ ] Exit loop early if `research_complete=True`
  - [ ] Exit loop if `continue_searching=False`
  - [ ] Respect `max_searches` as hard cap regardless of reflection decision
- [ ] **2.4** Update `TopicResearchResult` to include `early_completion: bool` and `completion_rationale: str`
- [ ] **2.5** Write `tests/research/test_topic_reflection.py`:
  - [ ] Test mandatory reflection after each search iteration
  - [ ] Test early exit on `research_complete=True`
  - [ ] Test `continue_searching=False` behavior
  - [ ] Test hard cap `max_searches` override of reflection decision
  - [ ] Test fallback parsing when JSON extraction fails
  - [ ] Test reflection prompt includes source count and quality info
- [ ] **2.6** Verify existing deep research contract tests still pass

---

## Phase 3: Per-Topic Compression Before Aggregation

- [ ] **3.1** Add `compressed_findings: Optional[str]` to `TopicResearchResult` in `models/deep_research.py`
  - [ ] Ensure Pydantic serialization backward-compat
- [ ] **3.2** Add `compression_model` and `compression_provider` to `config.py`
  - [ ] Default: same as research model/provider
- [ ] **3.3** Implement per-topic compression step in `phases/gathering.py`:
  - [ ] After all topic researchers complete, iterate `TopicResearchResult` list
  - [ ] For each topic: collect its sources from `state.sources` by `sub_query_id`
  - [ ] Build compression prompt: "Reformat findings with inline citations [1], [2]. DO NOT summarize. Preserve all relevant information."
  - [ ] Call LLM with compression provider/model
  - [ ] Store result in `TopicResearchResult.compressed_findings`
  - [ ] Progressive token-limit handling (3 retries, 10% truncation — reuse Phase 5 infra)
  - [ ] Fallback: if compression fails, leave `compressed_findings=None` (analysis uses raw sources)
- [ ] **3.4** Parallel compression across topics, bounded by `max_concurrent`
- [ ] **3.5** Track compression tokens in `PhaseMetrics`
- [ ] **3.6** Update `phases/analysis.py` to prefer `compressed_findings` when available:
  - [ ] If all topics have `compressed_findings`, use those as primary analysis input
  - [ ] Adjust budget allocation to account for pre-compressed content
  - [ ] Fall through to existing raw-source analysis when `compressed_findings` is `None`
- [ ] **3.7** Write `tests/research/test_topic_compression.py`:
  - [ ] Test compression prompt includes correct sources per topic
  - [ ] Test progressive truncation on token limit error
  - [ ] Test fallback to raw sources when compression fails
  - [ ] Test analysis phase uses compressed findings when available
  - [ ] Test analysis phase falls back when compressed findings absent
  - [ ] Test parallel compression across multiple topics
  - [ ] Test citation numbering consistency between compression and analysis
- [ ] **3.8** Verify existing deep research contract tests still pass

---

## Phase 4: Structured Clarification Gate

- [ ] **4.1** Define `ClarificationDecision` schema in `_helpers.py`:
  - [ ] `{need_clarification: bool, question: str, verification: str}`
  - [ ] Extraction via `extract_json()` with retry-on-parse-failure (3 attempts)
- [ ] **4.2** Add `execute_structured_llm_call()` variant to `phases/_lifecycle.py`:
  - [ ] Accepts expected JSON schema
  - [ ] Requests JSON-mode output from provider
  - [ ] Validates response against schema
  - [ ] Retries up to 3 times on validation failure
  - [ ] Falls back to unstructured call on exhaustion
- [ ] **4.3** Update `phases/clarification.py`:
  - [ ] Use `execute_structured_llm_call()` with `ClarificationDecision` schema
  - [ ] On `need_clarification=True`: return question (existing flow)
  - [ ] On `need_clarification=False`: store `verification` in `state.clarification_constraints`, log audit event, proceed to planning
  - [ ] On parse failure fallback: treat as "no clarification needed" (existing behavior)
- [ ] **4.4** Write `tests/research/test_clarification_structured.py`:
  - [ ] Test structured output parsing for both `need_clarification` values
  - [ ] Test `verification` stored in state
  - [ ] Test retry on parse failure
  - [ ] Test fallback to unstructured on exhaustion
  - [ ] Test audit event logged on verification
- [ ] **4.5** Verify existing deep research contract tests still pass

---

## Phase 6: Multi-Model Cost Optimization

- [ ] **6.1** Formalize model role hierarchy in `config.py`:
  - [ ] `research_model` / `research_provider` — analysis, synthesis (strongest)
  - [ ] `summarization_model` / `summarization_provider` — fetch-time (cheapest)
  - [ ] `compression_model` / `compression_provider` — per-topic compression
  - [ ] `reflection_model` / `reflection_provider` — think-tool pauses
  - [ ] `report_model` / `report_provider` — final synthesis
  - [ ] All default to `None` (falls back to phase-level, then global default)
- [ ] **6.2** Add `resolve_model_for_role(role: str) -> tuple[str, str]` to config:
  - [ ] Resolution chain: role-specific → phase-level → global default
  - [ ] Returns `(provider_id, model)`
- [ ] **6.3** Add `role: Optional[str]` parameter to `execute_llm_call()` in `_lifecycle.py`:
  - [ ] When `role` is provided, resolve provider/model from config via `resolve_model_for_role()`
  - [ ] Explicit `provider_id`/`model` parameters still override role-based resolution
- [ ] **6.4** Update all phase callsites to pass appropriate roles:
  - [ ] `gathering.py` (topic research LLM calls) → `role="reflection"`
  - [ ] `gathering.py` (compression calls) → `role="compression"`
  - [ ] `topic_research.py` (reflection calls) → `role="reflection"`
  - [ ] `analysis.py` (finding extraction) → `role="research"`
  - [ ] `synthesis.py` (report generation) → `role="report"`
  - [ ] `clarification.py` (structured decision) → `role="research"`
  - [ ] `planning.py` (sub-query generation) → `role="research"`
  - [ ] Summarization (Phase 1 code) → `role="summarization"`
- [ ] **6.5** Add cost tracking per role in `PhaseMetrics.metadata["model_roles"]`:
  - [ ] `{role: {provider, model, input_tokens, output_tokens, calls}}`
- [ ] **6.6** Update `server(action="capabilities")` to include model roles in response
- [ ] **6.7** Write `tests/research/test_model_routing.py`:
  - [ ] Test role resolution chain (role-specific → phase → global)
  - [ ] Test explicit provider/model overrides role-based resolution
  - [ ] Test cost tracking per role
  - [ ] Test backward-compat when no role-specific config provided
  - [ ] Test all phase callsites pass expected roles
- [ ] **6.8** Verify existing deep research contract tests still pass

---

## Cross-Cutting Validation

- [ ] **V.1** Run full test suite (`pytest tests/`) — all existing tests pass
- [ ] **V.2** Run contract tests (`pytest tests/contract/`) — envelope schemas valid
- [ ] **V.3** Manual end-to-end test: run a deep research session with all features enabled
- [ ] **V.4** Compare token usage before/after on a reference query (document in PR)
- [ ] **V.5** Verify backward-compat: load a pre-existing saved research session, confirm it deserializes and can be resumed
- [ ] **V.6** Review all new config fields have sensible defaults and documentation
