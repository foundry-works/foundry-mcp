# Deep Research ODR Alignment — Phase 2 Plan

Follow-on from the Phase 1 ODR alignment plan (Phases 1-5, all complete).
Addresses three remaining priority gaps identified by comparing foundry-mcp's
deep research workflow against [open_deep_research](https://github.com/langchain-ai/open_deep_research).

**Prerequisite context (already implemented):**
- Synthesis phase *does* have findings-specific token-limit retry (synthesis.py:329-496)
  with the ODR pattern (model_token_limit×4 chars → 10% reduction per retry).
- Per-topic compression retries use message-boundary-aware truncation.
- Supervision messages accumulate think/delegation/tool_result entries across rounds.

**What's still missing:**
1. No `raw_notes` capture — when compression degrades or fails, there is no
   unprocessed fallback data for synthesis or evaluation.
2. Supervisor sees only compressed summaries from prior rounds — detailed
   evidence, specific URLs, and nuanced reasoning from directive researchers
   are discarded before the next delegation cycle.
3. Synthesis has no supplementary data path — when token-limit retries force
   findings truncation, the report loses depth with no way to recover detail.

---

## Phase 1: Raw Notes Pipeline

**ODR pattern:** Every researcher captures `raw_notes` alongside `compressed_research`.
`raw_notes` is populated by `filter_messages(researcher_messages, include_types=["tool", "ai"])`
— the unprocessed concatenation of all tool responses and AI reasoning from the
ReAct loop. These flow upward: researcher → supervisor (`raw_notes` list on
`SupervisorState`) → `AgentState.raw_notes` → available to both `final_report_generation`
and the groundedness evaluator.

**Current state:** `TopicResearchResult` stores `message_history` (full ReAct
conversation) and `compressed_findings` (post-compression summary). After
compression, `message_history` is retained on the model but is never aggregated
into a flat `raw_notes` field. Synthesis reads `compressed_findings` or the
global `compressed_digest` — it has no access to uncompressed researcher output.

**Changes:**

### 1a. Add `raw_notes` field to `TopicResearchResult`

Add a `raw_notes: Optional[str]` field to `TopicResearchResult`. Populated after
the ReAct loop completes (before compression), by concatenating the content of
all tool-result and assistant messages from `message_history`. This is the
researcher-level raw capture, matching ODR's per-researcher `raw_notes`.

**File:** `src/foundry_mcp/core/research/models/deep_research.py`

### 1b. Populate `raw_notes` in topic research completion

At the end of `_execute_topic_research_async()`, after the ReAct loop exits but
before compression, build `raw_notes` from `message_history`:

```python
raw_notes_parts = []
for msg in topic_result.message_history:
    if msg.get("role") in ("assistant", "tool_result"):
        raw_notes_parts.append(msg.get("content", ""))
topic_result.raw_notes = "\n".join(raw_notes_parts)
```

This ensures raw notes survive even if compression fails or drops content.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

### 1c. Add aggregated `raw_notes` field to `DeepResearchState`

Add `raw_notes: list[str]` to `DeepResearchState`. This is the session-level
aggregation of all researcher raw notes, populated during the gathering and
supervision phases as topic researchers complete.

**File:** `src/foundry_mcp/core/research/models/deep_research.py`

### 1d. Aggregate raw notes after topic research and directive execution

In `_execute_gathering_async()` (gathering phase) and `_execute_directives_async()`
(supervision phase), after topic researchers return results, append each
`topic_result.raw_notes` to `state.raw_notes`:

```python
for result in topic_results:
    if result.raw_notes:
        state.raw_notes.append(result.raw_notes)
```

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/gathering.py`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

---

## Phase 2: Supervisor Context Preservation

**ODR pattern:** In ODR, the supervisor accumulates `supervisor_messages` with the
full LangGraph message objects (including complete tool call results from researchers).
`raw_notes` is separately aggregated and never compressed within the supervisor loop.
The supervisor always sees both its compressed notes (`notes` list) and has access
to the full raw researcher output.

**Current state:** After directives execute, their results are compressed inline
(`_compress_directive_results_inline`), and only the compressed text is appended to
`supervision_messages` as a `tool_result` entry (supervision.py:347-360). The raw
researcher output — full message histories, individual search results, source-level
detail — is discarded from the supervisor's view. On round N+1, the supervisor
sees compressed summaries from round N. This causes two problems:

1. **Re-investigation:** The supervisor may re-target gaps that were actually
   addressed in the raw findings but whose detail was lost during compression.
2. **Evidence loss:** Specific URLs, data points, and quotes that the supervisor
   could reference when formulating targeted directives are unavailable.

**Changes:**

### 2a. Append raw evidence summaries alongside compressed findings

After directives execute and inline compression runs, build a concise
**evidence inventory** from each directive result's `raw_notes` (from Phase 1)
or `source_ids` + source metadata. Append this as a separate `evidence_inventory`
message to `supervision_messages`, alongside the existing compressed `tool_result`:

```python
# After existing compressed findings append (supervision.py:347-360):
if result.raw_notes or result.source_ids:
    inventory = self._build_evidence_inventory(result, state)
    state.supervision_messages.append({
        "role": "tool_result",
        "type": "evidence_inventory",
        "round": state.supervision_round,
        "directive_id": result.sub_query_id,
        "content": inventory,
    })
```

The evidence inventory is a structured, compact summary (not the full raw notes)
listing: sources found (URL + title), key data points extracted, and which
sub-queries they address. This gives the supervisor specific evidence to reason
about without the full token cost of raw notes.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

### 2b. Build `_build_evidence_inventory()` helper

Create a helper that produces a compact evidence summary from a directive result:

```
Sources: 4 found, 3 unique domains
- [1] "Title A" (example.com) — covers: pricing, features
- [2] "Title B" (docs.example.com) — covers: API reference
- [3] "Title C" (blog.example.com) — covers: user experience
Key data points: 7 extracted
Topics addressed: pricing comparison, feature matrix
```

This is structured enough for the supervisor to know what evidence exists
without reading the full compressed findings, and specific enough to prevent
re-investigation of already-covered topics.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

### 2c. Include evidence inventories in supervisor think prompt

Update `_build_think_prompt()` and `_build_combined_think_delegate_user_prompt()`
to render `evidence_inventory` messages distinctly from `research_findings`
messages. The think prompt should present inventories as a "what evidence we
have" section separate from the "what we found" section:

```
### [Round 1] Evidence Inventory (directive dir-abc123)
Sources: 4 found, 3 unique domains
- [1] "Title A" (example.com) — covers: pricing, features
...

### [Round 1] Research Findings (directive dir-abc123)
[compressed findings text]
```

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

### 2d. Token budget for evidence inventories in supervision messages

Evidence inventories add to the supervision message history growth. To prevent
token overflow:
- Cap each evidence inventory at 500 chars (roughly 125 tokens).
- When `truncate_supervision_messages()` runs, evidence inventories from the
  oldest rounds are dropped before compressed findings from recent rounds.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py`

---

## Phase 3: Synthesis Raw-Notes Fallback

**ODR pattern:** `final_report_generation()` builds findings from
`"\n".join(notes)` — the compressed research. But `raw_notes` is available on
`AgentState` and is used by the evaluator (`eval_groundedness`) as ground truth.
When token retry truncates findings, the raw notes provide a safety net: the
evaluator can assess whether the report still reflects the full evidence base.

**Current state:** Synthesis builds its prompt from `compressed_digest` (preferred),
per-topic `compressed_findings`, or analysis `findings`. When token-limit retries
force truncation, the findings section shrinks progressively. There is no
supplementary data path — truncated findings mean lost detail with no recovery.

**Changes:**

### 3a. Inject raw notes as supplementary context for synthesis (token-budget permitting)

After building the primary synthesis prompt from compressed findings, calculate
remaining token headroom. If headroom exists (>10% of context window unused),
append a `## Supplementary Research Notes` section with excerpts from
`state.raw_notes`, truncated to fit the available budget:

```python
headroom = estimate_remaining_tokens(system_prompt, user_prompt, provider_id)
if headroom > supplementary_threshold:
    raw_notes_text = "\n---\n".join(state.raw_notes)
    raw_notes_truncated = truncate_at_boundary(raw_notes_text, headroom_chars)
    user_prompt += f"\n\n## Supplementary Research Notes\n{raw_notes_truncated}"
```

This gives the synthesizer access to uncompressed detail when budget allows,
improving report depth without risking token-limit failures.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

### 3b. Use raw notes as degraded-mode fallback when compressed findings are empty

In `_execute_synthesis_async()`, extend the "no findings" check (synthesis.py:236-240)
to fall back on `state.raw_notes` before generating an empty report:

```python
has_compressed = any(tr.compressed_findings for tr in state.topic_research_results)
has_raw_notes = bool(state.raw_notes)

if not state.findings and not state.compressed_digest and not has_compressed:
    if has_raw_notes:
        # Degraded mode: synthesize directly from raw notes
        logger.warning("No compressed findings, falling back to raw notes for synthesis")
        # Build synthesis prompt from raw_notes instead
        ...
    else:
        # No data at all: generate empty report
        ...
```

This prevents the "empty report" path when compression failed but raw data exists.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

### 3c. Include raw notes in evaluation groundedness check

When the evaluation framework (`evaluation/evaluator.py`) runs LLM-as-judge
scoring, pass `state.raw_notes` as the ground-truth context for the
groundedness dimension. This matches ODR's `eval_groundedness` which uses
`outputs["raw_notes"]` as context to assess whether the report is supported
by the evidence gathered.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/evaluation/evaluator.py`

---

## Dependency Graph

```
Phase 1 (raw notes pipeline)
    ├──→ Phase 2 (supervisor context) — uses raw_notes for evidence inventory
    └──→ Phase 3 (synthesis fallback) — uses raw_notes for supplementary/degraded paths
```

Phase 1 is the foundation. Phases 2 and 3 can be implemented in parallel once
Phase 1 is complete.

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | `raw_notes` increases `DeepResearchState` serialization size | Cap per-topic raw_notes at `max_content_length` (50k chars default); gc oldest on state save |
| 2 | Evidence inventories add token overhead to supervision | Hard cap at 500 chars/inventory; drop oldest-round inventories first during truncation |
| 3a | Supplementary notes in synthesis could cause token overflow | Only inject when headroom > threshold; raw notes are always truncated to fit |
| 3b | Raw notes are noisier than compressed findings | System prompt instructs synthesizer to prefer compressed findings; raw notes are labeled "supplementary" |
