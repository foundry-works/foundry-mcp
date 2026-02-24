# PLAN: Deep Research Alignment Improvements

## Context

Comparison of foundry-mcp's deep research workflow against `open_deep_research` (upstream reference) revealed several areas where alignment can improve research quality. This plan addresses the actionable gaps, ordered by expected impact.

---

## Phase 1 — Supervisor Message Accumulation

**Problem:** open_deep_research's supervisor accumulates its conversation across rounds — it sees its own prior AIMessage + ToolMessages containing compressed research results from each delegation. foundry-mcp rebuilds context from structured coverage data each round (source counts, quality distributions, findings excerpts truncated to 2000 chars). This means the supervisor never sees its own prior reasoning or the full compressed output from earlier rounds.

**Why it matters:** The supervisor's gap analysis quality degrades when it can't reference its own prior thinking. Coverage metrics are a lossy proxy for the actual research content. The supervisor may re-delegate topics it already covered because it only sees truncated excerpts, not the full compressed findings.

**Changes:**

1. **Accumulate supervisor messages across rounds.** Maintain a `supervision_messages: list[dict]` on `DeepResearchState` that grows across rounds:
   - Round N system prompt + user prompt → AIMessage (delegation response) → synthetic ToolMessages with compressed findings from each executed directive
   - Pass full message list to delegation LLM on round N+1
   - Keep the existing structured coverage data as a supplement, not the sole context

2. **Inject compressed findings as tool-result messages.** After each directive's topic researcher completes and compresses, format its `compressed_findings` as a ToolMessage-style entry in the supervisor's message history. This mirrors how open_deep_research feeds `ConductResearch` results back.

3. **Token-limit guard on supervisor history.** Apply the same message-aware truncation strategy (oldest-first removal) if the accumulated supervisor messages approach the model's context limit. The structured coverage data serves as the fallback when messages are truncated.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` — accumulate and pass messages
- `src/foundry_mcp/core/research/workflows/deep_research/models/deep_research.py` — add `supervision_messages` field to state
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py` — token guard for supervisor context

---

## Phase 2 — Researcher Forced Reflection via think_tool

**Problem:** open_deep_research explicitly requires the researcher to call `think_tool` after every search before doing anything else. The prompt says "CRITICAL: Use think_tool after each search to reflect on results and plan next steps" and the supervisor prompt forbids calling think_tool in parallel with other tools. foundry-mcp's researcher treats `think` as optional and allows multiple tool calls per turn, meaning the researcher can skip reflection entirely and chain searches without pausing.

**Why it matters:** Forced reflection between searches is a core quality mechanism — it makes the researcher assess what it found before deciding the next query. Without it, researchers tend to exhaust their budget on broad, overlapping searches rather than progressively narrowing.

**Changes:**

1. **Add reflection enforcement to the researcher prompt.** Update `_build_researcher_system_prompt` in `topic_research.py`:
   - After each `web_search` or `extract_content` call, the researcher MUST call `think` before issuing another search
   - `think` may NOT be called in parallel with `web_search` or `extract_content`
   - Multiple `web_search` calls in a single turn are allowed ONLY if they are the first tool calls of the turn (initial broadening)

2. **Validate tool call ordering in the ReAct loop.** In `_execute_react_turn` (or equivalent), check that if the previous turn contained a search tool, the current turn starts with a `think` call. If not, inject a synthetic think prompt asking the researcher to reflect before continuing. This is a soft enforcement — log a warning and inject rather than hard-block.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — prompt update + turn validation

---

## Phase 3 — Synthesis Source Format Alignment

**Problem:** open_deep_research's final report prompt tells the model to use `[Title](URL)` markdown links inline AND to build a `### Sources` section with `[N] Source Title: URL` format. foundry-mcp tells the model NOT to generate a Sources section (it's auto-appended) and uses `[N]` inline only. The auto-appended section uses `[N] [Title](URL)` markdown link format.

The net effect is similar, but there's a gap: open_deep_research's prompt explicitly asks the model to use `[Title](URL)` inline references (clickable links), while foundry-mcp only asks for bare `[N]` numbers. This means foundry-mcp reports have less navigable inline text — readers must scroll to the Sources section to find URLs.

**Why it matters:** Inline markdown links are significantly more useful in rendered markdown environments. A reader seeing `[MIT Tech Review](https://...)` mid-paragraph gets immediate context about the source authority without scrolling.

**Changes:**

1. **Add inline markdown link guidance to synthesis prompt.** Update `_build_synthesis_system_prompt`:
   - Instruct the model to use `[Title](URL) [N]` format for first reference to each source
   - Subsequent references to the same source use just `[N]`
   - Keep the auto-appended Sources section as the canonical reference

2. **Update postprocess_citations to preserve inline links.** The current regex `\[(\d+)\](?!\()` already avoids markdown links. Verify that `[Title](URL) [N]` patterns survive post-processing without the `[N]` being stripped as dangling.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` — prompt update
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_citation_postprocess.py` — verify/fix regex

---

## Phase 4 — Token Limit Recovery in Report Generation

**Problem:** open_deep_research handles token limits during report generation with a clear 3-tier strategy: first try full context, then truncate findings to `model_token_limit * 4` characters, then progressively reduce by 10% per retry (max 3 retries). foundry-mcp delegates token recovery to the lifecycle layer's message-aware truncation, which operates on message history rather than findings content. For synthesis specifically, this is suboptimal — the findings string is what needs truncating, not the message structure.

**Why it matters:** If synthesis hits a token limit, the current recovery truncates messages (which may remove the system prompt or research context) rather than trimming the findings payload. open_deep_research's approach of specifically truncating the findings string while keeping the prompt intact is more targeted.

**Changes:**

1. **Add findings-specific truncation in synthesis phase.** In the synthesis execution path, catch token-limit errors and implement:
   - First retry: truncate the user prompt's findings section to `model_token_limit * 4` characters
   - Subsequent retries: reduce by 10% per attempt
   - Max 3 retries
   - Keep the system prompt and source reference intact; only trim findings text
   - Fall back to the existing lifecycle-level recovery if findings truncation alone doesn't resolve it

2. **Log truncation metrics.** Record how much content was dropped and at which retry, so the audit trail shows synthesis fidelity degradation.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` — add try/retry with findings truncation
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py` — ensure lifecycle recovery doesn't conflict

---

## Phase 5 — Researcher Stop Heuristics Alignment

**Problem:** open_deep_research's researcher prompt has explicit "Stop Immediately When" rules:
- You can answer comprehensively
- You have 3+ relevant examples/sources
- Last 2 searches returned similar info (diminishing returns)

foundry-mcp's researcher prompt says to call `research_complete` "when confident or diminishing returns" but lacks the specific heuristics. The `early_completion` flag exists but the decision criteria are vague.

**Why it matters:** Without concrete stop rules, researchers either over-search (wasting budget) or under-search (missing important sources). The "3+ examples" and "last 2 searches similar" rules are simple, effective heuristics.

**Changes:**

1. **Add explicit stop heuristics to researcher system prompt.** Update `_build_researcher_system_prompt`:
   - "Stop and call research_complete when ANY of: (a) you can answer the research question comprehensively, (b) you have found 3+ high-quality, relevant sources, (c) the last 2 searches returned substantially similar results"

2. **Carry the heuristics into the think prompt.** When the researcher calls `think`, include a checklist: "Before your next search, check: Do I have 3+ relevant sources? Did my last 2 searches overlap? Can I answer comprehensively now?"

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — prompt updates

---

## Phase 6 — Supervisor think_tool Pattern

**Problem:** open_deep_research's supervisor uses `think_tool` as a first-class tool call — the supervisor decides when to reflect, and the reflection is recorded in the message history as a ToolMessage. This means the supervisor's reasoning about gaps is visible in the conversation and feeds into subsequent rounds. foundry-mcp's supervisor has a separate `_execute_supervision_think_async` phase that runs a dedicated LLM call for gap analysis, but the output is passed as structured data, not accumulated in a message history.

This is closely related to Phase 1 (message accumulation). The improvement here is specifically about making the supervisor's gap-analysis reasoning conversational rather than ephemeral.

**Why it matters:** When the supervisor's think output is a standalone structured blob, it's disconnected from the delegation decision. When it's part of the message history, the LLM can reference its own prior reasoning ("In my earlier analysis, I identified X gap — the new findings from researcher 3 partially address it but Y remains open").

**Changes:**

1. **Integrate think output into supervisor message history.** After the think phase produces gap analysis text, inject it as an assistant message in the supervisor's accumulated messages (from Phase 1). The delegation prompt then naturally follows in the same conversation.

2. **Make think and delegate a single conversation turn.** Instead of two separate LLM calls (think → delegate), consider making think the first part of a single call: system prompt + accumulated history + "First, analyze coverage gaps. Then, generate directives." This reduces latency and ensures the delegation is directly informed by the gap analysis without prompt-engineering the handoff. Evaluate whether single-call quality matches two-call quality before committing.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` — restructure think+delegate flow
- Dependent on Phase 1 (message accumulation)
