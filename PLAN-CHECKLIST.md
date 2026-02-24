# PLAN-CHECKLIST: Deep Research Architecture — Brief, Compression & Delegation

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-24

---

## Phase 1: Research Brief Generation

- [ ] **1.1** Add `BRIEF` to `DeepResearchPhase` enum
  - Insert between `CLARIFICATION` and `PLANNING`
  - Update `advance_phase()` logic in `DeepResearchState`
  - Ensure backward compatibility (existing states without BRIEF still work)
- [ ] **1.2** Add `research_brief` field to `DeepResearchState`
  - `research_brief: Optional[str] = None`
  - Add to serialization/deserialization (to_dict/from_dict)
  - Available to all downstream phases
- [ ] **1.3** Create `phases/brief.py` with `BriefPhaseMixin`
  - `_execute_brief_async()` — main orchestrator
  - `_build_brief_system_prompt()` — adapted from open_deep_research's `transform_messages_into_research_topic_prompt`:
    - Maximize specificity of the research question
    - Fill unstated dimensions as open-ended (not assumed)
    - Specify source preferences (primary/official sources, peer-reviewed, language-native)
    - Include scope boundaries (what to investigate, what to exclude)
  - `_build_brief_user_prompt()` — includes original query, clarification constraints (from `state.clarification_constraints`), current date
  - Uses `execute_structured_llm_call()` for parse-retry reliability
  - Non-fatal: if brief generation fails, log warning and proceed with original query
- [ ] **1.4** Add `BRIEFER` agent role to orchestration
  - Add to `PHASE_TO_AGENT` mapping in `orchestration.py`
  - Add to `AgentRole` enum if applicable
  - Add to `get_phase_timeout()` with appropriate timeout
- [ ] **1.5** Wire BRIEF phase into `workflow_execution.py`
  - Add BRIEF handling after CLARIFICATION, before PLANNING
  - Include phase-boundary reflection hook (if enabled)
  - Add `_execute_brief_async` to TYPE_CHECKING stubs
  - Emit phase audit events (phase.started, phase.complete)
- [ ] **1.6** Update planning to consume research brief
  - In `_execute_planning_async()`: when `state.research_brief` is not None, use it as decomposition input
  - Skip the existing inline brief-refinement sub-step when dedicated brief exists
  - Fall back to original query when `state.research_brief` is None (config disabled or generation failed)
- [ ] **1.7** Update synthesis to include research brief in context
  - In `_build_synthesis_user_prompt()`: add research brief as context section
  - Label: "## Research Brief" with the enriched question
  - Helps synthesis align report scope to the enriched question
- [ ] **1.8** Update supervision to reference research brief
  - In `_build_supervision_user_prompt()`: include research brief scope boundaries
  - Coverage assessment can reference which dimensions of the brief are covered vs. missing
- [ ] **1.9** Add config fields to `ResearchConfig`
  - `deep_research_enable_brief: bool = True`
  - `deep_research_brief_provider: Optional[str] = None` (defaults to research-tier)
  - `deep_research_brief_model: Optional[str] = None`
  - Add `brief` role to `_ROLE_RESOLUTION_CHAIN` (falls back to research → default)
  - Add `from_toml_dict()` parsing for new keys
- [ ] **1.10** Add `phases/__init__.py` export for BriefPhaseMixin
- [ ] **1.11** Add tests for research brief generation
  - Test: brief enriches a vague query with dimensions and source preferences
  - Test: brief preserves explicit user constraints from clarification
  - Test: planning uses research_brief when available
  - Test: planning falls back to original_query when brief is None
  - Test: brief skipped when `deep_research_enable_brief=False`
  - Test: synthesis prompt includes research brief
  - Test: supervision prompt includes brief scope boundaries
  - Test: brief generation failure is non-fatal (proceeds with original query)
  - Test: audit events emitted for brief phase
  - Test: phase ordering correct (CLARIFICATION → BRIEF → PLANNING)
  - Test: backward compatibility (existing states without research_brief field)

---

## Phase 2: Inline Per-Topic Compression Before Supervision

- [x] **2.1** Extract reusable per-topic compression helper
  - Factor `_compress_single_topic_async()` from `compression.py` per-topic logic
  - Accept: topic_result, state, timeout
  - Return: (input_tokens, output_tokens, success) tuple
  - Reusable from both gathering phase (inline) and compression phase (fallback)
- [x] **2.2** Add inline compression to `_execute_topic_research_async()`
  - After ReAct loop completes and before returning `TopicResearchResult`
  - Call `_compress_single_topic_async()` with the topic's sources and ReAct context
  - Populate `TopicResearchResult.compressed_findings` inline
  - Use compression-tier model (cheap, fast)
  - Non-fatal: on failure, leave `compressed_findings = None` and log warning
  - Track compression tokens in topic result metadata
- [x] **2.3** Update `_build_supervision_user_prompt()` for content-aware assessment
  - For each completed sub-query: include truncated compressed findings (max ~2000 chars)
  - Format: `**Key findings:**\n{compressed_findings[:2000]}`
  - Fall back to findings_summary format when compressed findings unavailable
- [x] **2.4** Update `_build_supervision_system_prompt()` for content assessment
  - Instruct LLM to assess content coverage, not just source diversity
  - New guidance: "Evaluate whether the findings substantively address the research brief's dimensions"
  - "Identify specific content gaps where important perspectives or evidence are missing"
  - "Consider both quantitative coverage (source count/diversity) and qualitative coverage (finding depth)"
- [x] **2.5** Refactor global compression phase (`_execute_global_compression_async`)
  - Batch compression in gathering.py skips already-compressed topics
  - Global compression already reads from `TopicResearchResult.compressed_findings`
  - Falls back to re-compressing from raw sources if `compressed_findings` is None for any topic
- [x] **2.6** Add `deep_research_inline_compression: bool = True` config flag
  - When False, per-topic compression deferred to batch step after all topics complete
  - Update `from_toml_dict()` parsing
- [x] **2.7** Add tests for inline compression
  - Test: `compressed_findings` populated in `TopicResearchResult` after gathering
  - Test: supervision prompt includes compressed content excerpts
  - Test: supervision coverage assessment references actual findings
  - Test: global compression reads pre-compressed findings (no double compression)
  - Test: inline compression failure is non-fatal
  - Test: config flag disables inline compression (falls back to separate phase)
  - Test: supervision falls back to metadata-only when compressed findings unavailable
  - Test: token usage tracking includes inline compression tokens

---

## Phase 3: Iteration Budget + Extract Tool for Researchers

- [x] **3.1** Raise iteration budget defaults
  - `deep_research_topic_max_searches`: 5 → 10
  - `deep_research_max_supervision_rounds`: 3 → 6
  - Update docstrings and config comments
  - Early-exit heuristic unchanged (prevents over-searching on simple queries)
- [x] **3.2** Rename `deep_research_topic_max_searches` to `deep_research_topic_max_tool_calls`
  - Backward compatibility: accept old name in `from_toml_dict()` with deprecation warning
  - Update all references in topic_research.py, config, tests
  - Reflects that search + extract both count toward budget
- [x] **3.3** Add `_topic_extract()` method to topic research
  - Input: list of URLs to extract (max `deep_research_extract_max_per_iteration`)
  - Uses Tavily Extract API via existing provider infrastructure
  - Summarizes extracted content via `SourceSummarizer` (50K char cap)
  - Creates `ResearchSource` entries with full/summarized content
  - Respects concurrency semaphore and state lock for dedup
  - Non-fatal: extraction failures logged, research continues with search sources
- [x] **3.4** Update reflection decision schema
  - Add `urls_to_extract: Optional[list[str]] = None` field (max 2)
  - Update `parse_reflection_decision()` to handle new field
  - Backward compatible: field is optional, default None
- [x] **3.5** Update reflection system prompt for extract awareness
  - Add guidance: "If a search result snippet suggests rich content behind a URL (e.g., detailed technical documentation, comparison tables, research papers), recommend extracting it"
  - "Only recommend extraction for URLs where the snippet indicates valuable detail you cannot get from the snippet alone"
  - "Limit extraction recommendations to 2 URLs per iteration"
- [x] **3.6** Update ReAct loop to support extraction
  - After reflection, if `urls_to_extract` is non-empty and extract enabled:
    - Call `_topic_extract()` with recommended URLs
    - Add extracted sources to state (with dedup)
    - Count extraction as 1 tool call toward `max_tool_calls` budget
  - Continue to next iteration (search or reflect) after extraction
- [x] **3.7** Add config fields
  - `deep_research_enable_extract: bool = True`
  - `deep_research_extract_max_per_iteration: int = 2`
  - Add `from_toml_dict()` parsing
- [x] **3.8** Add tests for iteration budget and extract
  - Test: higher iteration budget allows deeper research on complex queries
  - Test: early-exit heuristic still fires on simple queries (no regression)
  - Test: extract tool fetches and summarizes URL content
  - Test: extracted sources deduplicated against search sources
  - Test: extraction failures non-fatal
  - Test: reflection can recommend URLs for extraction
  - Test: total tool calls (search + extract) respect `max_tool_calls` cap
  - Test: backward compatibility for old config key name
  - Test: supervision respects new round limit (6)
  - Test: extract disabled when config flag False

---

## Phase 4: Supervisor Delegation Model

- [x] **4.1** Add `ResearchDirective` dataclass to state model
  - Fields: `research_topic: str` (paragraph-length), `perspective: str`, `evidence_needed: str`, `priority: int`
  - Add `directives: list[ResearchDirective]` to supervision state tracking
  - Add serialization support
- [x] **4.2** Add `deep_research_max_concurrent_research_units: int = 5` config
  - Caps parallel researchers per delegation round
  - Add `deep_research_delegation_model: bool = True` flag
  - When disabled, falls back to existing query-generation supervision
  - Add `from_toml_dict()` parsing
- [x] **4.3** Refactor `_execute_supervision_async()` — delegation loop
  - Replace single LLM assessment call with multi-step loop:
    1. **Think**: analyze compressed findings, identify gaps (existing think step)
    2. **Delegate**: generate `ResearchDirective` objects from gap analysis (new)
    3. **Execute**: spawn parallel topic researchers for directives (new)
    4. **Compress**: inline compression of new results (Phase 2 infrastructure)
    5. **Assess**: evaluate coverage and decide continue/complete (updated)
  - Loop bounded by `max_supervision_rounds`
  - Exit on: `ResearchComplete` signal, all gaps filled, or round limit reached
- [x] **4.4** Add `_build_delegation_prompt()` to supervision
  - System prompt: "You are a research lead delegating tasks to specialized researchers"
  - Input: research brief, compressed findings from all topics, gap analysis from think step
  - Output format: JSON array of `ResearchDirective` objects
  - Guidance: paragraph-length directives, specify approach and evidence type
  - Limit: max `max_concurrent_research_units` directives per round
  - Include `ResearchComplete` signal option (when no more gaps)
- [x] **4.5** Add `_parse_delegation_response()`
  - Extract `ResearchDirective` objects from LLM JSON response
  - Validate: research_topic non-empty, priority 1-3
  - Handle `research_complete: true` signal
  - Cap directives at `max_concurrent_research_units`
  - Graceful fallback: on parse failure, generate single directive from gap analysis
- [x] **4.6** Add `_execute_directives_async()` to supervision
  - Convert each `ResearchDirective` into a `SubQuery` (directive topic as query text)
  - Spawn parallel `_execute_topic_research_async()` calls (reuse gathering infrastructure)
  - Bounded by `asyncio.Semaphore(max_concurrent_research_units)`
  - Collect `TopicResearchResult` objects with inline compression
  - Add new sources/findings to state under lock
  - Non-fatal: individual directive failures don't block others
- [x] **4.7** Add think-before-delegate and think-after-results
  - Before delegation: think step articulates specific gaps (existing from prior plan)
  - After results return: think step assesses what was learned and what remains
  - Both use reflection-tier model
  - Think outputs logged in supervision_history
- [x] **4.8** Update `workflow_execution.py` for delegation model
  - Supervision can now trigger inline gathering (not just follow-up query append)
  - New results merged into state alongside original gathering results
  - Phase transition: after supervision completes, proceed to ANALYSIS (unchanged)
- [x] **4.9** Ensure backward compatibility
  - When `deep_research_delegation_model=False`, use existing query-generation path
  - Existing tests pass without modification
  - State model additions are optional fields with defaults
- [x] **4.10** Add tests for supervisor delegation
  - Test: supervisor generates paragraph-length directives (not single-sentence queries)
  - Test: directives target specific gaps identified in think output
  - Test: parallel researcher execution respects concurrency limit
  - Test: supervisor sees compressed findings from directive execution
  - Test: think step runs before delegation and after results
  - Test: `ResearchComplete` signal terminates supervision loop
  - Test: round limit triggers automatic completion
  - Test: fallback to query-generation when delegation disabled
  - Test: directive priorities influence execution order
  - Test: individual directive failure doesn't block others
  - Test: new sources properly merged into state with dedup
  - Test: total search budget respected across delegation rounds

