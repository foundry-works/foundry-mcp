# PLAN-CHECKLIST: Deep Research — Supervisor Orchestration, Compression & Synthesis Alignment

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-24

---

## Phase 1: Unify Supervision as the Research Orchestrator

- [x] **1.1** Remove `deep_research_supervisor_owned_decomposition` config flag
  - Delete field from `research.py` config class *(already removed in prior phases)*
  - Remove from `from_toml_dict()` parsing *(already removed in prior phases)*
  - Remove any default value assignments *(already removed in prior phases)*
  - Ensure old config files with this key don't cause runtime errors (ignore unknown keys) *(verified: `data.get()` silently ignores unknown keys)*
- [x] **1.2** Make BRIEF → SUPERVISION the sole default transition in `workflow_execution.py`
  - Verify lines 194-199 unconditionally skip PLANNING/GATHERING for new workflows *(confirmed)*
  - Remove any remaining conditional checks on the deleted config flag *(none found — flag only in PLAN docs)*
  - Add comment: "PLANNING and GATHERING are legacy-resume-only phases" *(added)*
- [x] **1.3** Guard PLANNING phase as legacy-resume-only
  - PLANNING removed from DeepResearchPhase enum entirely — no legacy resume possible
  - planning.py code retained for backward compat but unreachable from workflow_execution.py
- [x] **1.4** Guard GATHERING phase as legacy-resume-only
  - GATHERING block only executes if `state.phase == DeepResearchPhase.GATHERING` on workflow entry *(confirmed)*
  - Added deprecation warning log and audit event when legacy GATHERING phase runs
  - gathering.py code retained for backward compat
- [x] **1.5** Review and harden first-round decomposition prompts
  - Verified first-round think prompt includes open_deep_research guidance:
    - "Bias toward single researcher for simple queries" *(present in think system prompt)*
    - "Parallelize for explicit comparisons (one per comparison element)" *(present in delegation system prompt)*
    - "2-5 directives for typical queries" *(present in delegation system prompt)*
  - Verified first-round delegate prompt includes:
    - Priority assignment (1=critical, 2=important, 3=nice-to-have) *(present)*
    - Specificity guidance: each directive yields targeted results *(present)*
    - Self-critique: "Verify no redundant directives and no missing perspectives" *(present)*
- [x] **1.6** Verify round 0 → round 1 handoff
  - After round 0 directives execute, round 1 heuristic (`_assess_coverage_heuristic`) assesses decomposition results *(confirmed — heuristic early-exit only runs at round > 0)*
  - Heuristic sees round 0's topic_research_results and sources *(confirmed via test)*
  - Round 0 counted toward `max_supervision_rounds` *(confirmed)*
- [x] **1.7** Clean up dead references to config flag
  - Searched codebase for `supervisor_owned_decomposition` — only found in PLAN docs (expected)
  - No conditionals, comments, or test fixtures reference the flag in source code
- [x] **1.8** Add tests for unified supervisor orchestration
  - Test: new workflow goes BRIEF → SUPERVISION → SYNTHESIS (no PLANNING/GATHERING) *(TestUnifiedSupervisorOrchestration)*
  - Test: PLANNING phase not in enum (never executes for new or legacy workflows) *(test_planning_phase_not_in_enum)*
  - Test: GATHERING phase never executes for new workflows *(test_gathering_only_runs_from_legacy_resume)*
  - Test: legacy saved state at GATHERING resumes correctly *(test_legacy_gathering_resume_enters_gathering_block)*
  - Test: round 0 always delegates, round 1 assesses results *(test_round_zero_always_delegates, test_round_one_assesses_round_zero_results)*
  - Test: round 0 results visible to round 1 heuristic *(test_coverage_data_includes_round_zero_results)*
  - Test: deprecation log emitted when legacy phase runs *(test_deprecation_log_emitted_for_legacy_gathering)*
  - Test: old config files with `supervisor_owned_decomposition` key don't crash *(test_old_config_key_supervisor_owned_decomposition_ignored)*
- [x] **1.9** Verify existing deep research tests still pass *(348 passed, 0 failures)*

---

## Phase 2: Pass Full Message History to Compression

- [x] **2.1** Add `message_history` field to `TopicResearchResult`
  - Field: `message_history: list[dict[str, str]] = Field(default_factory=list)` *(added to models/deep_research.py)*
  - Default empty list ensures Pydantic backward-compat
  - Roundtrip serialization verified via test
- [x] **2.2** Store message history on result in `topic_research.py`
  - At end of ReAct loop, `result.message_history = list(message_history)` *(added before compile/compression)*
  - Includes all entries: assistant responses, tool results (web_search, think, extract, research_complete)
- [x] **2.3** Update compression prompt to use message history when available
  - In `_compress_single_topic_async()` (compression.py):
    - When `topic_result.message_history` non-empty → `_build_message_history_prompt()` formats chronological conversation
    - `max_content_length` cap applied, oldest messages truncated first (preserves recent reasoning)
    - When `message_history` empty → `_build_structured_metadata_prompt()` fallback (no breaking change)
- [x] **2.4** Align compression system prompt with open_deep_research structure
  - `<Task>`: Clean up findings, preserve ALL relevant information verbatim *(present)*
  - `<Guidelines>`: 6 rules matching open_deep_research's compress_research_system_prompt *(present)*
  - `<Output Format>`: "Queries and Tool Calls Made" → "Fully Comprehensive Findings" → "Sources" *(present)*
  - `<Citation Rules>`: Sequential `[1] Source Title: URL`, no gaps *(present)*
  - Critical Reminder: "preserved verbatim ... don't rewrite, don't summarize, don't paraphrase" *(present)*
- [x] **2.5** Add tests for message-history-based compression
  - Test: message_history field default, set/get, model_dump, backward compat deserialization *(TestMessageHistoryField — 4 tests)*
  - Test: compression prompt includes raw message history when available *(TestCompressionPromptDispatch — 2 tests)*
  - Test: compression prompt falls back to structured metadata when message_history empty *(TestCompressionPromptDispatch)*
  - Test: message history truncated to max_content_length, oldest dropped first *(TestMessageHistoryTruncation + TestBuildMessageHistoryPrompt — 3 tests)*
  - Test: citation format `[N] Title: URL` in system prompt *(TestCompressionSystemPromptAlignment — 5 tests)*
  - Test: "Queries and Tool Calls Made" in output format *(TestCompressionSystemPromptAlignment)*
  - Test: existing compression tests pass with new prompt structure *(131 passed, 0 failed)*
  - Total new tests: 29 in test_message_history_compression.py
- [x] **2.6** Verify existing deep research tests still pass *(2010 passed, 6 skipped, 0 failures)*

---

## Phase 3: Align Synthesis Prompt with open_deep_research

- [ ] **3.1** Add section verbosity expectation to synthesis system prompt
  - Add: "Each section should be as long as necessary to deeply answer the question with the information gathered. Sections are expected to be thorough and detailed. You are writing a deep research report and users expect comprehensive answers."
  - Place after the "Writing Quality" section
- [ ] **3.2** Soften structure prescriptiveness
  - Add after structure guidance: "These are suggestions. Section is a fluid concept — you can structure your report however you think is best, including in ways not listed above. Make sure sections are cohesive and make sense for the reader."
  - Keep query-type hints as starting points, not rigid templates
- [ ] **3.3** Make Analysis subsections optional
  - Change mandatory "Analysis" section with "Supporting Evidence", "Conflicting Information", and "Limitations" subsections
  - Replace with: "Include analysis of conflicting information and limitations where they exist, but integrate them naturally into the relevant sections rather than forcing separate subsections."
- [ ] **3.4** Add citation importance emphasis
  - Add to Citations section: "Citations are extremely important. Pay careful attention to getting these right. Users will often use citations to find more information on specific points."
- [ ] **3.5** Strengthen language matching
  - Add a second language-matching instruction at the end of the system prompt:
    - "REMEMBER: The research and brief may be in English, but the final report MUST be written in the same language as the user's original query. This is critical — the user will only understand the answer if it matches their input language."
- [ ] **3.6** Add per-section writing rules
  - "Use ## for each section title (Markdown format)"
  - "Write in paragraph form by default; use bullet points only when listing discrete items"
  - "Do not refer to yourself or comment on the report itself — just write the report"
- [ ] **3.7** Add tests for synthesis prompt changes
  - Test: system prompt includes verbosity expectation ("thorough", "detailed", "comprehensive")
  - Test: system prompt includes structure flexibility ("however you think is best")
  - Test: "Supporting Evidence" / "Conflicting Information" / "Limitations" NOT mandatory
  - Test: citation section includes importance emphasis
  - Test: language matching instruction appears at least twice
  - Test: per-section writing rules present (## headers, paragraph form, no self-reference)
  - Test: query-type classification still works (comparison, enumeration, howto, explanation)
- [ ] **3.8** Verify existing synthesis tests still pass

---

## Phase 4: Message-Aware Token Limit Recovery

- [ ] **4.1** Add `truncate_prompt_for_retry()` helper to `_lifecycle.py`
  - Signature: `truncate_prompt_for_retry(prompt: str, attempt: int, max_attempts: int = 3) -> str`
  - Attempt 1: remove first 20% of content (preserve tail)
  - Attempt 2: remove first 30% of content
  - Attempt 3: remove first 40% of content
  - Never truncate below a minimum threshold (e.g., 1000 chars)
  - Returns the truncated prompt string
- [ ] **4.2** Add retry loop to `_compress_single_topic_async`
  - Wrap the `execute_llm_call` in a retry loop (max 3 attempts)
  - On token limit error (ContextWindowError):
    - Apply `truncate_prompt_for_retry()` to user prompt
    - Log: `"Compression retry %d/%d: truncating prompt by %d%%"`
    - Retry the LLM call
  - On success: break out of retry loop
  - After 3 failures: return `(0, 0, False)` — non-fatal, skip compression for this topic
  - Record retry count in audit event
- [ ] **4.3** Add retry loop to `_execute_synthesis_async`
  - Wrap the `execute_llm_call` in a retry loop (max 3 attempts)
  - On token limit error:
    - Apply `truncate_prompt_for_retry()` to user prompt
    - For synthesis specifically: drop lowest-priority topics' findings first, then truncate remaining
    - Log each retry with truncation percentage
  - After 3 failures: generate partial report with whatever content fit
  - Record retry count in audit event
- [ ] **4.4** Verify provider-specific token limit error detection
  - OpenAI: `BadRequestError` + "maximum context length" / "too many tokens" / "token"
  - Anthropic: `BadRequestError` + "prompt is too long" / "too many tokens"
  - Google: `ResourceExhausted` / `InvalidArgument` with "token" keyword
  - Verify existing `ContextWindowError` classification covers these patterns
  - Add any missing patterns
- [ ] **4.5** Add tests for token limit recovery
  - Test: compression retries on simulated token limit error
  - Test: synthesis retries on simulated token limit error
  - Test: progressive truncation (20% → 30% → 40%)
  - Test: system prompt never truncated (only user prompt content)
  - Test: most recent content preserved (oldest content truncated first)
  - Test: max 3 retries, then graceful fallback
  - Test: non-token-limit errors NOT retried (e.g., auth errors, rate limits)
  - Test: retry metadata recorded in audit events
  - Test: provider-specific error detection for OpenAI, Anthropic, Google
  - Test: `truncate_prompt_for_retry()` unit tests (boundary cases, minimum threshold)
- [ ] **4.6** Verify existing deep research tests still pass
