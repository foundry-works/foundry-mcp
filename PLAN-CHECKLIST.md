# PLAN CHECKLIST: Deep Research Alignment Improvements

## Phase 1 — Supervisor Message Accumulation
- [x] 1.1 Add `supervision_messages: list[dict]` field to `DeepResearchState`
- [x] 1.2 After each directive completes, format its `compressed_findings` as a tool-result message and append to `supervision_messages`
- [x] 1.3 Pass accumulated `supervision_messages` to delegation LLM in `_run_delegation_async`
- [x] 1.4 Keep structured coverage data as supplementary context (don't remove it)
- [x] 1.5 Add oldest-first message truncation guard when supervisor history approaches model context limit
- [x] 1.6 Test: supervisor sees prior round findings in message history
- [x] 1.7 Test: supervisor doesn't re-delegate already-covered topics
- [x] 1.8 Test: token truncation preserves most recent rounds

## Phase 2 — Researcher Forced Reflection
- [x] 2.1 Update `_build_researcher_system_prompt`: after each `web_search`/`extract_content`, researcher MUST call `think` before next search
- [x] 2.2 Add rule: `think` may NOT be called in parallel with search tools
- [x] 2.3 Allow multiple `web_search` calls in first turn only (initial broadening)
- [x] 2.4 Add soft validation in ReAct loop: if previous turn had search and current turn doesn't start with `think`, inject synthetic reflection prompt
- [x] 2.5 Log warning when reflection is injected (not hard-block)
- [x] 2.6 Test: researcher alternates search → think → search pattern
- [x] 2.7 Test: first-turn parallel searches are allowed
- [x] 2.8 Test: synthetic reflection injection works when researcher skips think

## Phase 3 — Synthesis Source Format Alignment
- [ ] 3.1 Update `_build_synthesis_system_prompt`: first reference to source uses `[Title](URL) [N]`, subsequent uses `[N]`
- [ ] 3.2 Verify `postprocess_citations` regex preserves `[Title](URL) [N]` patterns
- [ ] 3.3 Fix regex if `[N]` adjacent to `)` is incorrectly stripped
- [ ] 3.4 Test: inline markdown links survive post-processing
- [ ] 3.5 Test: auto-appended Sources section still correct with inline links present

## Phase 4 — Token Limit Recovery in Report Generation
- [ ] 4.1 Add try/except around synthesis LLM call for token-limit errors
- [ ] 4.2 First retry: truncate findings to `model_token_limit * 4` characters
- [ ] 4.3 Subsequent retries: reduce by 10% per attempt, max 3 retries
- [ ] 4.4 Keep system prompt and source reference intact during truncation
- [ ] 4.5 Fall back to lifecycle-level recovery if findings truncation insufficient
- [ ] 4.6 Log truncation metrics (chars dropped, retry count) to audit trail
- [ ] 4.7 Test: synthesis succeeds after findings truncation
- [ ] 4.8 Test: lifecycle recovery kicks in as final fallback
- [ ] 4.9 Test: audit trail records truncation details

## Phase 5 — Researcher Stop Heuristics
- [ ] 5.1 Add "Stop Immediately When" block to `_build_researcher_system_prompt`:
  - (a) can answer comprehensively
  - (b) 3+ high-quality relevant sources found
  - (c) last 2 searches returned substantially similar results
- [ ] 5.2 Add checklist to think-tool prompt: "Before next search: Do I have 3+ sources? Did last 2 searches overlap? Can I answer now?"
- [ ] 5.3 Test: researcher calls `research_complete` when 3+ sources found
- [ ] 5.4 Test: researcher stops after 2 overlapping searches

## Phase 6 — Supervisor think_tool as Conversation
- [ ] 6.1 After think phase, inject gap analysis as assistant message in `supervision_messages` (depends on Phase 1)
- [ ] 6.2 Evaluate single-call approach: system prompt + history + "analyze gaps then generate directives" — compare quality vs two-call
- [ ] 6.3 If single-call quality matches: merge think + delegate into one LLM call
- [ ] 6.4 If single-call quality worse: keep two calls but ensure think output flows through message history
- [ ] 6.5 Test: supervisor references prior gap analysis in delegation rationale
- [ ] 6.6 Test: latency reduction measured if single-call adopted
