# Foundry Deep Research — Implementation Checklist

Tracks progress against [PLAN.md](./PLAN.md). Check items as completed.

---

## Phase 1: Quick Wins

### 1. Query Clarification Phase
- [ ] Add `CLARIFICATION` to `DeepResearchPhase` enum in `models/deep_research.py`
- [ ] Add `clarification_constraints` dict field to `DeepResearchState`
- [ ] Add `CLARIFIER` to `AgentRole` enum in `orchestration.py`
- [ ] Create `phases/clarification.py` with `ClarificationPhaseMixin`
  - [ ] `_execute_clarification_async()` — single LLM call
  - [ ] `_build_clarification_system_prompt()` — structured JSON output instructions
  - [ ] `_build_clarification_user_prompt()` — original query + any user context
  - [ ] `_parse_clarification_response()` — extract questions / constraints
- [ ] Wire into `workflow_execution.py` before planning phase
- [ ] Register mixin in `core.py` (`DeepResearchWorkflow` class)
- [ ] Add config keys to `.foundry-mcp.toml`:
  - [ ] `deep_research_allow_clarification` (default: `true`)
  - [ ] `deep_research_clarification_provider`
- [ ] Handle "skip" path — if user doesn't answer, proceed with original query
- [ ] Unit tests for clarification parsing
- [ ] Integration test: query → clarification → planning flow

### 4. Proactive Content Digest
- [ ] Add `"proactive"` to `deep_research_digest_policy` validation
- [ ] In `phases/gathering.py`, add post-gather digest step:
  - [ ] Check if policy is `proactive`
  - [ ] Call `DocumentDigestor` on each newly gathered source
  - [ ] Store digest results on source objects
  - [ ] Audit event for proactive digest completion
- [ ] Ensure analysis phase uses pre-digested content when available (skip re-digest)
- [ ] Update config documentation for new policy option
- [ ] Unit test: gathering with proactive digest
- [ ] Verify token counting uses digested content length

### 6. End-to-End Citation Tracking
- [ ] Add `citation_number: Optional[int]` to source model
- [ ] Assign citation numbers sequentially in `state.add_source()` or gathering phase
- [ ] Update `_build_synthesis_user_prompt()`:
  - [ ] Present findings with `[N]` citation markers
  - [ ] Include citation legend mapping N → source title + URL
- [ ] Update synthesis system prompt to instruct inline `[N]` citation usage
- [ ] Auto-generate `## Sources` section post-synthesis:
  - [ ] Build from state sources (not LLM output)
  - [ ] Format: `[N] Title — URL`
- [ ] Create `_citation_postprocess.py`:
  - [ ] Scan report for `[N]` references
  - [ ] Verify all referenced N exist in sources
  - [ ] Warn on unreferenced sources (optional)
  - [ ] Remove dangling citations
- [ ] Unit tests for citation assignment and post-processing
- [ ] Verify citations survive refinement iterations (re-synthesis)

---

## Phase 2: Core Architecture

### 2. LLM-Driven Supervisor Reflection
- [ ] Add config keys:
  - [ ] `deep_research_enable_reflection` (default: `false`)
  - [ ] `deep_research_reflection_provider`
  - [ ] `deep_research_reflection_timeout`
- [ ] Add `async_think_pause()` method to `SupervisorOrchestrator`:
  - [ ] Accept state + reflection prompt
  - [ ] Call LLM (fast model) with structured output schema
  - [ ] Parse: `{"quality_assessment", "proceed", "adjustments", "rationale"}`
  - [ ] Return decision object
- [ ] Wire `async_think_pause()` into `workflow_execution.py`:
  - [ ] Call after each phase completion (where `hooks.emit_phase_complete` already fires)
  - [ ] If `proceed: false`, log adjustment suggestions (don't retry in v1)
- [ ] Update `evaluate_phase_completion()`:
  - [ ] When reflection is enabled, use LLM assessment instead of hardcoded thresholds
  - [ ] Preserve hardcoded thresholds as fallback when reflection is disabled
- [ ] Record reflection decisions in audit trail
- [ ] Unit tests: reflection enabled vs disabled paths
- [ ] Integration test: verify reflection doesn't break existing workflow

---

## Phase 3: Advanced

### 3. Parallel Sub-Topic Researcher Agents
- [ ] Design `TopicResearchResult` model:
  - [ ] `sub_query_id`, `searches_performed`, `sources_found`, `per_topic_summary`
  - [ ] `reflection_notes` (from per-topic reflect step)
- [ ] Create `phases/topic_research.py` with `TopicResearchMixin`:
  - [ ] `_execute_topic_research_async(sub_query)` — single topic ReAct loop
  - [ ] `_topic_search()` — search scoped to one sub-query
  - [ ] `_topic_reflect()` — fast LLM evaluates results, suggests refinement
  - [ ] Loop: search → reflect → (refine query → search)* → compile summary
  - [ ] Max iterations per topic: configurable (`deep_research_topic_max_searches`)
- [ ] Modify `phases/gathering.py`:
  - [ ] When `deep_research_enable_topic_agents = true`, delegate to topic researchers
  - [ ] Run topic researchers in parallel with `asyncio.gather()` + semaphore
  - [ ] Collect `TopicResearchResult` per sub-query
  - [ ] Merge sources and per-topic summaries into state
- [ ] Budget splitting: divide search budget across sub-queries
- [ ] Add config keys:
  - [ ] `deep_research_enable_topic_agents` (default: `false`)
  - [ ] `deep_research_topic_max_searches` (default: `3`)
  - [ ] `deep_research_topic_reflection_provider`
- [ ] Per-topic audit events
- [ ] Unit tests: single topic research loop
- [ ] Integration test: multi-topic parallel execution
- [ ] Verify deduplication across topic researchers

### 5. Contradiction Detection
- [ ] Add `Contradiction` model to `models/deep_research.py`:
  - [ ] `id`, `finding_ids: list[str]`, `description`, `resolution`, `preferred_source_id`
  - [ ] `severity: str` (major/minor)
- [ ] Add `contradictions: list[Contradiction]` to `DeepResearchState`
- [ ] Add post-analysis contradiction detection in `phases/analysis.py`:
  - [ ] After findings extraction, send findings to LLM
  - [ ] System prompt: identify conflicting claims between findings
  - [ ] Parse structured JSON response
  - [ ] Store contradictions in state
- [ ] Update `phases/synthesis.py`:
  - [ ] Include contradictions section in synthesis prompt
  - [ ] Instruct LLM to address contradictions explicitly in report
  - [ ] Suggest resolution approach (prefer higher-quality source, note uncertainty)
- [ ] Optionally create contradiction-type gaps for refinement
- [ ] Audit events for detected contradictions
- [ ] Unit tests for contradiction parsing
- [ ] Integration test: contradictory sources → report addresses them

---

## Cross-Cutting

- [ ] Update `.foundry-mcp.toml` with all new config keys (with comments)
- [ ] Update foundry-sandbox config documentation (`docs/configuration.md`)
- [ ] Run full test suite after each phase completion
- [ ] Update CHANGELOG.md with new capabilities
