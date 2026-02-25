# Deep Research ODR Alignment Phase 2 — Checklist

Cross-reference: [PLAN.md](PLAN.md) for detailed rationale, ODR patterns, and code references.

**Note:** Synthesis token-limit retry already exists (synthesis.py:329-496).
These phases address the remaining gaps: raw data capture, supervisor context
preservation, and synthesis fallback paths.

---

## Phase 1: Raw Notes Pipeline

- [x] **1a** Add `raw_notes: Optional[str]` field to `TopicResearchResult`
  - File: `src/foundry_mcp/core/research/models/deep_research.py`
  - Field: `Optional[str]`, default `None`, described as unprocessed concatenation of tool+assistant messages
  - Test: Verify serialization round-trip with raw_notes populated

- [x] **1b** Populate `raw_notes` in topic research completion
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`
  - Location: End of `_execute_topic_research_async()`, after ReAct loop, before compression
  - Logic: Concatenate `content` from all `message_history` entries where `role in ("assistant", "tool")`
  - Cap: Truncate to `config.deep_research_max_content_length` (default 50k chars)
  - Test: Unit test that raw_notes is populated after a mocked ReAct loop with 3+ messages

- [x] **1c** Add `raw_notes: list[str]` field to `DeepResearchState`
  - File: `src/foundry_mcp/core/research/models/deep_research.py`
  - Field: `list[str]`, default `[]`, session-level aggregation of all researcher raw notes
  - Test: Verify serialization round-trip

- [x] **1d** Aggregate raw notes after topic research completion (gathering phase)
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/gathering.py`
  - Location: After `_execute_topic_research_async()` returns, append `result.raw_notes` to `state.raw_notes`
  - Test: Integration test verifying `state.raw_notes` has entries after gathering completes

- [x] **1e** Aggregate raw notes after directive execution (supervision phase)
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
  - Location: After `_execute_directives_async()` returns (near line 347), append each result's raw_notes
  - Test: Unit test verifying directive results' raw_notes flow into `state.raw_notes`

---

## Phase 2: Supervisor Context Preservation

Depends on: Phase 1 (1a, 1b) for `raw_notes` on `TopicResearchResult`.

- [x] **2a** Append evidence inventory messages to `supervision_messages`
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
  - Location: After compressed findings append (line ~360), add `evidence_inventory` message
  - Message format: `{"role": "tool_result", "type": "evidence_inventory", "round": N, "directive_id": "...", "content": "..."}`
  - Test: Unit test verifying evidence_inventory messages appear in supervision_messages after directive execution

- [x] **2b** Implement `_build_evidence_inventory()` helper
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
  - Input: `TopicResearchResult` + `DeepResearchState` (for source metadata lookup)
  - Output: Compact string listing sources (URL + title + coverage), data point count, topics addressed
  - Cap: 500 chars max per inventory
  - Test: Unit test with mock TopicResearchResult containing 5 sources, verify output format and char cap

- [x] **2c** Render evidence inventories in supervisor think prompt
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
  - Update: `_build_combined_think_delegate_user_prompt()` and `_build_delegation_user_prompt()`
  - Render `evidence_inventory` type messages with distinct header: `### [Round N] Evidence Inventory`
  - Test: Snapshot test of think prompt with both research_findings and evidence_inventory messages

- [x] **2d** Add evidence inventory awareness to `truncate_supervision_messages()`
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py`
  - Rule: When truncating for token limits, drop `evidence_inventory` messages from oldest rounds before `research_findings`
  - Test: Unit test with 20+ supervision messages, verify oldest inventories dropped first

---

## Phase 3: Synthesis Raw-Notes Fallback

Depends on: Phase 1 (1c, 1d, 1e) for `state.raw_notes` populated.

- [x] **3a** Inject raw notes as supplementary context in synthesis prompt
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`
  - Location: In `_build_synthesis_user_prompt()` or after `_build_synthesis_tail()`
  - Logic: Estimate remaining token headroom after primary prompt; if >10% window free, append `## Supplementary Research Notes` with truncated raw_notes
  - Guard: Never exceed 80% of context window with supplementary content
  - Test: Unit test with mock state containing raw_notes, verify supplementary section appears when headroom exists and is absent when budget is tight

- [x] **3b** Fall back to raw notes when compressed findings are empty
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`
  - Location: `_execute_synthesis_async()` near line 240, extend empty-findings check
  - Logic: If no compressed findings but `state.raw_notes` exist, build synthesis prompt from raw_notes directly
  - Add `degraded_mode: bool` to synthesis audit event and WorkflowResult metadata
  - Test: Unit test with empty compressed_findings but populated raw_notes, verify report is generated (not empty report)

- [x] **3c** Pass raw notes to evaluation groundedness scorer
  - File: `src/foundry_mcp/core/research/workflows/deep_research/evaluation/evaluator.py`
  - Logic: When computing groundedness dimension, set context = `"\n".join(state.raw_notes)` if available, else fall back to compressed findings
  - Test: Unit test verifying groundedness evaluator receives raw_notes as context
