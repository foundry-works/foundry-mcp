# Foundry Deep Research — Implementation Checklist

Tracks progress against [PLAN.md](./PLAN.md). Check items as completed.

---

## Phase 1: Quick Wins

### 1. Query Clarification Phase
- [x] Add `CLARIFICATION` to `DeepResearchPhase` enum in `models/deep_research.py`
- [x] Add `clarification_constraints` dict field to `DeepResearchState`
- [x] Add `CLARIFIER` to `AgentRole` enum in `orchestration.py`
- [x] Create `phases/clarification.py` with `ClarificationPhaseMixin`
  - [x] `_execute_clarification_async()` — single LLM call
  - [x] `_build_clarification_system_prompt()` — structured JSON output instructions
  - [x] `_build_clarification_user_prompt()` — original query + any user context
  - [x] `_parse_clarification_response()` — extract questions / constraints
- [x] Wire into `workflow_execution.py` before planning phase
- [x] Register mixin in `core.py` (`DeepResearchWorkflow` class)
- [x] Add config keys to `.foundry-mcp.toml`:
  - [x] `deep_research_allow_clarification` (default: `true`)
  - [x] `deep_research_clarification_provider`
- [x] Handle "skip" path — if user doesn't answer, proceed with original query
- [x] Unit tests for clarification parsing
- [x] Integration test: query → clarification → planning flow

### 4. Proactive Content Digest
- [x] Add `"proactive"` to `deep_research_digest_policy` validation
- [x] Add `PROACTIVE` to `DigestPolicy` enum in `document_digest/config.py`
- [x] Update `_is_eligible()` and `_get_skip_reason()` in digestor to handle PROACTIVE
- [x] In `workflow_execution.py`, add post-gather digest step:
  - [x] Check if policy is `proactive`
  - [x] Call `_execute_digest_step_async` on gathered sources (reuses existing digest pipeline)
  - [x] Store digest results on source objects
  - [x] Audit event for proactive digest completion (`proactive_digest_complete`)
- [x] Ensure analysis phase uses pre-digested content when available (skip re-digest)
- [x] Update config documentation for new policy option
- [x] Unit test: gathering with proactive digest (`test_proactive_digest.py`)
- [x] Verify token counting uses digested content length

### 6. End-to-End Citation Tracking
- [x] Add `citation_number: Optional[int]` to source model
- [x] Assign citation numbers sequentially in `state.add_source()` or gathering phase
- [x] Update `_build_synthesis_user_prompt()`:
  - [x] Present findings with `[N]` citation markers
  - [x] Include citation legend mapping N → source title + URL
- [x] Update synthesis system prompt to instruct inline `[N]` citation usage
- [x] Auto-generate `## Sources` section post-synthesis:
  - [x] Build from state sources (not LLM output)
  - [x] Format: `[N] Title — URL`
- [x] Create `_citation_postprocess.py`:
  - [x] Scan report for `[N]` references
  - [x] Verify all referenced N exist in sources
  - [x] Warn on unreferenced sources (optional)
  - [x] Remove dangling citations
- [x] Unit tests for citation assignment and post-processing
- [x] Verify citations survive refinement iterations (re-synthesis)

---

## Phase 2: Core Architecture

### 2. LLM-Driven Supervisor Reflection
- [x] Add config keys:
  - [x] `deep_research_enable_reflection` (default: `false`)
  - [x] `deep_research_reflection_provider`
  - [x] `deep_research_reflection_timeout`
- [x] Add `async_think_pause()` method to `SupervisorOrchestrator`:
  - [x] Accept state + reflection prompt
  - [x] Call LLM (fast model) with structured output schema
  - [x] Parse: `{"quality_assessment", "proceed", "adjustments", "rationale"}`
  - [x] Return `ReflectionDecision` object
- [x] Wire `async_think_pause()` into `workflow_execution.py`:
  - [x] Call `_maybe_reflect()` after each phase completion
  - [x] If `proceed: false`, log adjustment suggestions (don't retry in v1)
- [x] Update `evaluate_phase_completion()`:
  - [x] LLM reflection runs alongside heuristic evaluation (coexisting, not replacing)
  - [x] Preserve hardcoded thresholds as fallback when reflection is disabled
- [x] Record reflection decisions in audit trail
- [x] Unit tests: reflection enabled vs disabled paths
- [x] Integration test: verify reflection doesn't break existing workflow (1491 passed)

---

## Phase 3: Advanced

### 3. Parallel Sub-Topic Researcher Agents
- [x] Design `TopicResearchResult` model:
  - [x] `sub_query_id`, `searches_performed`, `sources_found`, `per_topic_summary`
  - [x] `reflection_notes` (from per-topic reflect step)
- [x] Create `phases/topic_research.py` with `TopicResearchMixin`:
  - [x] `_execute_topic_research_async(sub_query)` — single topic ReAct loop
  - [x] `_topic_search()` — search scoped to one sub-query
  - [x] `_topic_reflect()` — fast LLM evaluates results, suggests refinement
  - [x] Loop: search → reflect → (refine query → search)* → compile summary
  - [x] Max iterations per topic: configurable (`deep_research_topic_max_searches`)
- [x] Modify `phases/gathering.py`:
  - [x] When `deep_research_enable_topic_agents = true`, delegate to topic researchers
  - [x] Run topic researchers in parallel with `asyncio.gather()` + semaphore
  - [x] Collect `TopicResearchResult` per sub-query
  - [x] Merge sources and per-topic summaries into state
- [x] Budget splitting: divide search budget across sub-queries
- [x] Add config keys:
  - [x] `deep_research_enable_topic_agents` (default: `false`)
  - [x] `deep_research_topic_max_searches` (default: `3`)
  - [x] `deep_research_topic_reflection_provider`
- [x] Per-topic audit events
- [x] Unit tests: single topic research loop
- [x] Integration test: multi-topic parallel execution
- [x] Verify deduplication across topic researchers

### 5. Contradiction Detection
- [x] Add `Contradiction` model to `models/deep_research.py`:
  - [x] `id`, `finding_ids: list[str]`, `description`, `resolution`, `preferred_source_id`
  - [x] `severity: str` (major/minor)
- [x] Add `contradictions: list[Contradiction]` to `DeepResearchState`
- [x] Add post-analysis contradiction detection in `phases/analysis.py`:
  - [x] After findings extraction, send findings to LLM
  - [x] System prompt: identify conflicting claims between findings
  - [x] Parse structured JSON response
  - [x] Store contradictions in state
- [x] Update `phases/synthesis.py`:
  - [x] Include contradictions section in synthesis prompt
  - [x] Instruct LLM to address contradictions explicitly in report
  - [x] Suggest resolution approach (prefer higher-quality source, note uncertainty)
- [x] Optionally create contradiction-type gaps for refinement
- [x] Audit events for detected contradictions
- [x] Unit tests for contradiction parsing
- [x] Integration test: contradictory sources → report addresses them

---

## Cross-Cutting

- [x] Update `.foundry-mcp.toml` with all new config keys (with comments)
- [x] Update foundry-sandbox config documentation (`docs/configuration.md`)
- [x] Run full test suite after each phase completion
- [x] Update CHANGELOG.md with new capabilities
