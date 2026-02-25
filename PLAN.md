# Deep Research ODR Alignment — Improvement Plan

Improvements inspired by patterns observed in [open_deep_research](https://github.com/langchain-ai/open_deep_research) that foundry-mcp either lacks or implements partially.

**Out of scope** (foundry-mcp innovations to keep as-is):
- Two-step think→act supervision with confidence scoring
- Cross-topic deduplication and dual-pass compression
- Novelty-tagged search results (NOVEL/RELATED/DUPLICATE)
- Content-similarity deduplication (Jaccard threshold)

---

## Phase 1: Cost-Tiered Model Defaults

**ODR pattern:** Four explicit models with cost tiers — `summarization_model=gpt-4.1-mini` (cheap/fast), `research_model=gpt-4.1`, `compression_model=gpt-4.1`, `final_report_model=gpt-4.1`. Summarization runs on a ~10x cheaper model because it's high-volume, low-complexity work.

**Current state:** `ModelRoleConfig` (in `config/research_sub_configs.py:72-98`) defines 8+ role-specific `provider`/`model` fields (including `summarization_provider`, `summarization_model`, `compression_provider`, `compression_model`). All default to `None`, falling back to `config.default_provider` — meaning summarization, compression, and research all hit the same (expensive) model.

**Changes:**

### 1a. Set recommended cheap-model defaults for summarization
- In `ModelRoleConfig`, change `summarization_provider` and `summarization_model` defaults to point at a fast/cheap tier (e.g., `gemini-2.0-flash` or equivalent).
- The `SourceSummarizer` in `providers/shared.py:724` already respects `safe_resolve_model_for_role(config, "summarization")` — no wiring changes needed, only defaults.

### 1b. Set recommended cheap-model defaults for compression
- Per-topic compression (in `compression.py`) is also high-volume, moderate-complexity work. ODR uses the same-tier model, but a cheaper option would reduce cost without quality loss.
- Evaluate whether `compression_provider`/`compression_model` should default to the same cheap tier as summarization or stay at the research tier.

### 1c. Document the cost-tier pattern
- Add a section to the deep research config docstring or a dev_docs guide explaining the tiered model strategy: which roles are high-volume/low-complexity (summarization, compression) vs. low-volume/high-complexity (research, supervision, synthesis).
- Include example configurations showing cost/quality trade-offs.

**Files:**
- `src/foundry_mcp/config/research_sub_configs.py` — defaults
- `src/foundry_mcp/config/research.py` — docstring
- `dev_docs/guides/` — new or updated guide

---

## Phase 2: Supervisor Delegation Scaling Heuristics

**ODR pattern:** The `lead_researcher_prompt` includes explicit query-type→agent-count rules:
- Simple fact-finding: 1 agent
- Comparisons: 1 agent per compared element
- Complex multi-dimensional topics: multiple agents for sub-topics
- Bias toward fewer agents unless clear need for parallelization

**Current state:** Our delegation system prompt (`supervision.py:1210-1239`) instructs "Maximum 5 directives per round" with priority levels (1=critical, 2=important, 3=nice-to-have) but has no explicit scaling guidance based on query type. The supervisor can generate 1-5 directives every round regardless of query complexity.

**Changes:**

### 2a. Add scaling heuristics to delegation system prompt
- Add query-type-aware guidance to `_build_delegation_system_prompt()`:
  - Simple factual queries: 1-2 directives maximum
  - Comparison queries: 1 directive per compared element
  - Complex multi-dimensional: 3-5 directives targeting distinct dimensions
  - Bias toward fewer, more focused directives over many shallow ones

### 2b. Inject query complexity signal into delegation user prompt
- In `_build_delegation_user_prompt()`, add a heuristic complexity classification of the original query (based on sub-query count, brief length, or keyword patterns) so the supervisor has an explicit signal to calibrate directive count.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` — prompts

---

## Phase 3: Researcher Tool-Call Parse Resilience

**ODR pattern:** Uses LangChain's `model.with_structured_output(Schema).with_retry(stop_after_attempt=3)` for all key decision points — clarification, brief generation, research completion, search tool binding. Parse failures automatically retry with the same prompt.

**Current state:** Researcher ReAct loop in `topic_research.py` parses tool calls via `extract_json()` from raw LLM text output. Parse failures fall through to error handling but don't automatically retry the same LLM call. The supervision phase uses `execute_structured_llm_call()` (in `_lifecycle.py`) which does have retry logic, but the researcher loop doesn't share this pattern.

**Changes:**

### 3a. Add retry-on-parse-failure to researcher tool-call extraction
- When `extract_json()` fails to parse a valid tool call from the researcher's response, retry the LLM call (up to 2 retries) with the original prompt plus a clarifying suffix: "Your previous response was not valid JSON. Please respond with the exact JSON format specified."
- This matches ODR's `with_retry(stop_after_attempt=3)` pattern without requiring a framework change.

### 3b. Track parse failure rate in phase metrics
- Add a `tool_parse_failures` counter to `TopicResearchResult` or phase metrics so we can measure how often this happens and whether specific models are worse.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — ReAct loop
- `src/foundry_mcp/core/research/models/deep_research.py` — metrics

---

## Phase 4: Search Result Presentation Format

**ODR pattern:** Tavily results presented to researchers in a structured, citation-friendly format:
```
--- SOURCE 1: Title ---
URL: https://...

SUMMARY:
<summary>concise narrative</summary>
<key_excerpts>verbatim quotes</key_excerpts>
```
This format makes it easy for the researcher to reference specific sources and for the compression step to preserve citations.

**Current state:** Search results are presented to the researcher as part of the tool response message. Sources have `executive_summary` and `key_excerpts` fields (from `SourceSummarizer`), but the formatting of these into the researcher's context may not be as structured as ODR's numbered-source layout.

**Changes:**

### 4a. Standardize search result formatting in researcher context
- When building the tool response for `web_search` results in the ReAct loop, format each source with:
  - Sequential source number
  - Title and URL prominently displayed
  - Summary and key excerpts in labeled sections
  - Novelty tag (from Phase 3) included inline
- This creates a consistent "cite by number" pattern that flows through to compression and synthesis.

### 4b. Include source count and domain summary in search tool response header
- Prepend a brief header to each search result batch: "Found N results from M domains. N new, M related to prior results."
- Gives the researcher an at-a-glance novelty signal before reading individual results.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — search result formatting

---

## Phase 5: Compression Token-Limit Retry Strategy

**ODR pattern:** When compression hits a token limit, ODR removes messages progressively — dropping the oldest messages first while keeping the most recent ones — and retries up to 3 times. Each retry reduces the input by removing messages up to the last AI message boundary.

**Current state:** Our compression retry (in `compression.py` and `_lifecycle.py`) uses a pre-truncation strategy: the user prompt is truncated by 10% increments on each retry. This is functional but less targeted than ODR's approach of removing whole messages from the oldest end.

**Changes:**

### 5a. Implement message-boundary-aware truncation for compression retries
- On token limit failure, instead of blanket 10% truncation, identify message boundaries in the `message_history` list and drop the oldest complete messages first.
- Preserve: the most recent 2 think messages, the most recent search results, and the research_complete summary (if present).
- This keeps the highest-value context (most recent findings, final assessment) while shedding older, potentially superseded information.

### 5b. Add truncation metadata to compressed output
- When messages are dropped during retry, record which messages were dropped and how many retries were needed, in the `TopicResearchResult` or compression metadata.
- This feeds into the existing fidelity tracking system.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py` — retry logic
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py` — if shared retry helpers need updating
