# PLAN: Post-Review Fixes for Deep Research Pipeline

## Context

Senior code review of `tyler/foundry-mcp-20260223-0747` (123 files, +44K/-8K lines) identified critical, high, and medium issues across security, state management, configuration, and API surface areas. This plan addresses findings in priority order across 5 phases.

---

## Phase 1 — Critical: Timeout Budget & Execution Model (HIGH)

The default `deep_research_timeout=600s` is smaller than `supervision_wall_clock_timeout=1800s` and smaller than the worst-case planning phase (3 x 360s = 1080s). The switch from `background=True` to `background=False` in the handler compounds this by blocking MCP tool calls for the full duration.

### 1.1 Restore background execution with polling
- **File:** `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`
- **Lines:** 100-115
- Revert to `background=True` so the handler returns immediately with `research_id`
- Keep the existing status/polling handler path functional
- Remove or update vestigial polling guidance in status handler (lines 162-173)

### 1.2 Fix timeout budget inversion
- **File:** `src/foundry_mcp/config/research.py`
- **Lines:** 175, 1052-1060
- Raise `deep_research_timeout` default from 600s to at least 2400s (must exceed supervision wall-clock + planning + synthesis)
- OR lower `deep_research_supervision_wall_clock_timeout` and per-phase timeouts to fit within 600s
- Add validation: warn if `deep_research_timeout < deep_research_supervision_wall_clock_timeout`

### 1.3 Add aggregate timeout guard to planning phase
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/planning.py`
- Planning makes 3 sequential LLM calls each with `timeout` parameter — add wall-clock check between calls
- Divide phase timeout among calls or add cumulative elapsed-time check

### 1.4 Deprecate or wire `max_iterations`
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py`
- **Lines:** 300-338
- `decide_iteration()` unconditionally returns `should_iterate=False`, making `max_iterations` a no-op
- Either emit a deprecation warning when `max_iterations > 1` is configured, or implement the iteration loop in `workflow_execution.py`

---

## Phase 2 — Critical: Bounded State Growth (HIGH)

Unbounded accumulation of sources, topic research results, and task objects causes memory pressure and serialization bottlenecks in long-running servers.

### 2.1 Cap `state.sources` with eviction
- **File:** `src/foundry_mcp/core/research/models/deep_research.py`
- **Lines:** 775, 1050, 1069
- Add a global source count cap (e.g., 500) to `add_source()` / `append_source()`
- When cap is reached, evict oldest sources by dropping their `content` field (keep URL/title/metadata)
- OR move source content to separate storage and keep only references in state

### 2.2 Clear message histories after compression
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
- **Line:** 1187
- After compression succeeds for a `TopicResearchResult`, clear its `message_history` field
- The `compressed_findings` captures essential content — full history is no longer needed

### 2.3 Cap `topic_research_results` list
- **File:** `src/foundry_mcp/core/research/models/deep_research.py`
- **Line:** 782
- Add a hard cap (e.g., 50) on `topic_research_results` — drop oldest when exceeded
- OR trim `message_history` on each result to a summary after compression

### 2.4 Auto-clean `_tasks` class-level dict
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/core.py`
- **Lines:** 139-140
- Call `cleanup_stale_tasks()` at the start of each `_start_background_task`
- Prevents accumulation of `BackgroundTask` objects from crashed threads

### 2.5 Cap `supervision_messages` entry count
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
- **Lines:** 507-518
- Add a hard cap on `len(supervision_messages)` (e.g., 100) that drops oldest entries when exceeded
- Character-budget truncation keeps serialized size bounded, but entry count should also be capped

### 2.6 Bound `_active_research_sessions` module-level dict
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/infrastructure.py`
- **Lines:** 27-28
- Add a max sessions cap or periodic cleanup sweep
- Ensure sessions are removed from dict even if `finally` block doesn't run (e.g., atexit handler should `.clear()`)

---

## Phase 3 — Security: SSRF, Sanitization, Injection (MEDIUM)

### 3.1 Unify SSRF validators
- **Files:**
  - `src/foundry_mcp/core/research/workflows/deep_research/_injection_protection.py` (lines 26-35, 82-84)
  - `src/foundry_mcp/core/research/providers/tavily_extract.py` (line 268)
- Add missing blocked networks: `0.0.0.0/8`, `ff00::/8` (IPv6 multicast)
- Add hostname pattern blocking: `.local`, `.internal`, `.localhost` subdomains
- Consider consolidating into a single `validate_url()` function used by both modules
- Rename one if keeping separate to avoid name collision (`validate_extract_url` exists in both)

### 3.2 Fix double HTML entity encoding bypass
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/_injection_protection.py`
- **Lines:** 161-167
- Loop `html.unescape()` until output stabilizes (no more entities to decode), or call it twice
- Prevents `&amp;lt;system&amp;gt;` from surviving sanitization

### 3.3 Sanitize `report` and `query` in evaluation prompt
- **File:** `src/foundry_mcp/core/research/evaluation/evaluator.py`
- **Lines:** 128-139, 287-289
- Pass `state.report` through `sanitize_external_content()` before interpolation
- Pass `state.original_query` through `sanitize_external_content()` for defense-in-depth

### 3.4 Expand zero-width character stripping
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/_injection_protection.py`
- **Lines:** 161-163
- Add: `U+00AD` (Soft Hyphen), `U+034F` (Combining Grapheme Joiner), `U+2060` (Word Joiner), `U+2061-2064` (invisible math operators), `U+180E` (Mongolian Vowel Separator)

### 3.5 Fix `\b` word boundary allowing underscore-extended tags
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/_injection_protection.py`
- **Lines:** 117-124
- Change `\b` to `(?=[\s/>])` or `(?:\b|_)` to catch `<system_prompt>`, `<instructions_override>` etc.

---

## Phase 4 — Correctness: Orchestration & Config Validation (MEDIUM)

### 4.1 Validate `coverage_confidence_weights` schema
- **File:** `src/foundry_mcp/config/research.py`
- **Lines:** 110, 415, 741-788
- Add validation in `_validate_supervision_config()`: all values must be numeric, keys must be from known set (`source_adequacy`, `domain_diversity`, `query_completion_rate`)

### 4.2 Add supervision quality gate
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py`
- **Lines:** 271-279
- `evaluate_phase_completion` for SUPERVISION always returns `quality_ok=True`
- Add minimum quality check: at least some sources collected, at least one round completed

### 4.3 Fix cancellation rollback — clean partial data or check on resume
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`
- **Lines:** 441-475
- Add `rollback_note` checking in `_validate_state_for_resume()` in `session_management.py`
- OR clean up partial-iteration data (orphaned sources/findings) during rollback

### 4.4 Handle overlapping redundancy indices in critique
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/planning.py`
- **Lines:** 697-713
- Detect overlapping index sets across redundancy groups
- Skip later groups whose indices have already been removed

### 4.5 Fix gap priority parsing — uncaught ValueError
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/planning.py`
- **Line:** 636
- Wrap `int(g.get("priority", 1))` in `try/except (ValueError, TypeError)` with default of 1
- Consistent with sub-query parsing at lines 473-476

### 4.6 Add `deep_research_timeout` to timeout validation
- **File:** `src/foundry_mcp/config/research.py`
- **Lines:** 175, 694-714
- Include `deep_research_timeout` in the `timeout_fields` validation list
- Prevents zero/negative timeout from TOML config

### 4.7 Add PLANNING to `PHASE_TO_AGENT` mapping
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py`
- **Lines:** 57-67
- Add `DeepResearchPhase.PLANNING: AgentRole.PLANNER` (or document intentional omission)

---

## Phase 5 — Code Quality & Observability (LOW)

### 5.1 Add debug logging for `resolve_phase_provider` fallthrough
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/_model_resolution.py`
- **Lines:** 243-262
- Log at debug level which attributes were checked when no phase-specific provider is found

### 5.2 Inject `max_sub_queries` into critique system prompt
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/planning.py`
- Critique prompt says "2-7 after changes" but actual cap is `max_sub_queries` (default 5)
- Replace hardcoded range with actual constraint value

### 5.3 Add warning when `elif` guard fires in `advance_phase`
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`
- **Lines:** 274-275
- Log warning when the catch-all `elif` fires for an unexpected phase value

### 5.4 Update status handler to remove polling-specific guidance
- **File:** `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`
- **Lines:** 162-173
- Remove "checks remaining" and polling guidance if execution is synchronous
- OR keep if Phase 1.1 restores background execution (coordinate with Phase 1)

### 5.5 Build compression history incrementally
- **File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`
- **Lines:** 37-117
- Build `history_block` from the end, stopping once budget is reached, rather than materializing full block then truncating
- Avoids transient memory spikes with large message histories
