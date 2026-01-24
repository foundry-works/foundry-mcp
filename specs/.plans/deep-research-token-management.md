# Deep Research: Robust Token Management & Content Fidelity

## Problem Statement

The current deep research system uses reactive context management (catches errors after overflow) and character-based truncation (loses semantic content). It also treats model limits as static input budgets without reserving output tokens or accounting for dynamic overhead (tool schema, messages, system prompts). This can lead to:
- Phase failures on context overflow
- Lost information from hard truncation
- No visibility into what content was compressed/dropped
- Budget math that still overflows or over-summarizes in combined input/output models

## Solution Overview

Transform the system to use **proactive token management** with **intelligent summarization**, **provider-aware budgeting**, and **graceful degradation**. Add explicit response-contract updates for fidelity metadata, define summarization prompts/output schemas, and add state versioning for new persistence fields.

> Note: Content archive security governance (encryption/redaction/retention enforcement) is out of scope for this spec and should be handled in a follow-up security-focused spec.

---

## Simplification Summary

This spec applies a YAGNI approach with the following simplifications:

### 1. Fixed Safety Margin (no calibration)
- Single `token_safety_margin` config value (default 15%) instead of EMA-based calibration
- Removed: `TokenUsageObservation`, `record_token_observation()`, calibration persistence, drift tracking

### 2. User-Configured Runtime Overhead
- Single `runtime_overhead` config value instead of inferred CLI overhead dict
- User sets based on their environment (Claude Code: ~60000, others: ~40000)
- Removed: `CLI_SYSTEM_PROMPT_OVERHEAD` dict, `runtime_id()` method, inference logic

### 3. Simplified Model Limits
- Python dict (`DEFAULT_MODEL_LIMITS`) as single source of truth
- One config override dict: `model_context_overrides`
- Simple function: `get_model_limits(provider_id, config)`
- Removed: `model_limits.json` file, `ModelContextRegistry` class, limits probe command

### 4. Minimal Config Surface (~12 options)
**Kept**: `token_management_enabled`, `token_safety_margin`, `runtime_overhead`, `summarization_provider`, `summarization_providers`, `summarization_timeout`, `summarization_cache_enabled`, `content_archive_enabled`, `content_archive_ttl_hours`, `research_archive_dir`, `allow_content_dropping`, `model_context_overrides`

**Hardcoded**: analysis/synthesis budget percentages, retry/delay settings, min fidelity guardrails, truncate fallback

### 5. Items Kept As-Is
- Content Archive (Phase 6) - disabled by default
- Multi-layer Warning System - per-item, per-phase, and response-level warnings
- Summarization System - 4-level summarization with provider chains, map-reduce, caching
- State Versioning - migrations for backwards compatibility
- Fidelity Tracking - `content_fidelity` and `dropped_content_ids` in state/response

---

## Implementation Phases

### Phase 0: Contract + State Versioning

**Modify**: `dev_docs/codebase_standards/mcp_response_schema.md`, `docs/concepts/deep_research_workflow.md`, fixtures/docs where response examples live

Add explicit response schema changes for:
- `content_fidelity` (per item, per phase)
- `dropped_content_ids`
- `content_fidelity_schema_version` and `content_archive_hashes` (if enabled)

**Placement (decision)**:
- `data.content_fidelity`, `data.dropped_content_ids`, `data.content_archive_hashes`
- `meta.warnings` for degradation/partial-result warnings

**Compatibility/rollout**:
- Additive fields to `data` only; `meta.version` remains `response-v2`.
- Always include `content_fidelity_schema_version`, `content_fidelity`, `dropped_content_ids`, `content_archive_hashes` with empty defaults when token management is disabled to keep fixtures deterministic.
- Include `warning_details` as an empty list when no warnings.
- Warnings appear in `meta.warnings` when degradation occurs; otherwise omit the field.
- Fixtures/tests should assert default-empty fields are present even when token management is disabled.

**Contract matrix (affected responses)**:
- `deep-research-report`: includes fidelity fields + optional `meta.warnings`.
- `deep-research-start/status/list/delete`: unchanged payload; no new fields.

Example deep-research-report envelope:
```json
{
  "success": true,
  "data": {
    "report": "...",
    "content_fidelity_schema_version": "v1",
    "content_fidelity": {
      "source-123": {
        "phases": {
          "analysis": {"level": "key_points", "reason": "budget_limit", "warnings": []}
        }
      }
    },
    "warning_details": [
      {"code": "CONTENT_TRUNCATED", "message": "Sources summarized", "phase": "analysis"}
    ],
    "dropped_content_ids": ["source-999"],
    "content_archive_hashes": {"source-123": "<sha256>"},
  },
  "meta": {"version": "response-v2", "warnings": ["CONTENT_TRUNCATED"]}
}
```

Define **warning/error semantics** for degraded results (summarization/dropping) using `meta.warnings` and standardized warning codes (e.g., `CONTENT_TRUNCATED`, `CONTENT_DROPPED`, `SUMMARY_PROVIDER_FAILED`).

**Schema fragments (response-v2 additions)**:
```json
{
  "content_fidelity_schema_version": {"type": "string"},
  "warning_details": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "phase": {"type": "string"},
        "item_id": {"type": "string"}
      },
      "required": ["code", "message"]
    }
  },
  "content_fidelity": {
    "type": "object",
    "additionalProperties": {
      "type": "object",
      "properties": {
        "phases": {
          "type": "object",
          "additionalProperties": {
            "type": "object",
            "properties": {
              "level": {"enum": ["raw", "condensed", "key_points", "headline"]},
              "reason": {"enum": ["budget_limit", "provider_error", "manual_override"]},
              "warnings": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["level"]
          }
        }
      },
      "required": ["phases"]
    }
  },
  "dropped_content_ids": {"type": "array", "items": {"type": "string"}},
  "content_archive_hashes": {"type": "object", "additionalProperties": {"type": "string"}}
}
```

**Warning taxonomy (meta.warnings strings)**:
- `CONTENT_TRUNCATED`: low-priority content truncated
- `CONTENT_DROPPED`: content dropped to meet budget
- `PRIORITY_SUMMARIZED`: high-priority item summarized
- `SUMMARY_PROVIDER_FAILED`: summarization provider exhausted
- `TOKEN_BUDGET_FLOORED`: phase budget floored to minimum
- `LIMITS_DEFAULTED`: provider limits missing, defaults applied
- `ARCHIVE_WRITE_FAILED`: archive write failed (when enabled)
- `PROTECTED_OVERFLOW`: protected content exceeded budget
- `STATE_MIGRATION_RECOVERED`: state migration failed but defaults applied
- `TOKEN_COUNT_ESTIMATE_USED`: preflight token count used fallback estimate

Per-item warnings live in `content_fidelity[item_id].phases[phase].warnings` for localized reporting.

`meta.warnings` stays an array of strings to match `response-v2`; omit the field when there are no warnings.

`data.warning_details` provides structured warning context (code/message/phase/item_id) when needed.

**Warning/error decision table**:
- Map-reduce chunk failure: warning + retry; if still fails and no fallback, return error.
- Provider fallback exhausted: warning if partial data exists; error if nothing usable.
- Protected overflow after headline: error with remediation.

**Per-phase warning matrix**:
- Gathering: `CONTENT_TRUNCATED` when sources summarized; error if no sources remain.
- Analysis/Synthesis: `CONTENT_TRUNCATED`/`CONTENT_DROPPED` on degradation; `PROTECTED_OVERFLOW` escalates to error.
- Refinement: `CONTENT_TRUNCATED` when prior context summarized; error only on protected overflow.

Add **state schema versioning + migrations** for persisted `DeepResearchState` so older sessions load with defaults for new fields.

**Migration matrix**:
- v0 (no fidelity fields) → v1: default empty `content_fidelity`, empty `dropped_content_ids`.
- Downgrade guidance: strip fidelity/archive fields on save when target version is older.

**Migration policy**:
- Run migrations on state load; on failure, recover with defaults + `STATE_MIGRATION_RECOVERED` warning.
- Maintain a supported version window (current and previous) and add a cross-version load test.

**Versioning plan**: keep `response-v2` as the envelope version for additive fields; bump to a new response version only for breaking changes. Update schema docs with a version history entry and regenerate golden fixtures + changelog notes in `CHANGELOG.md`.

**Helper/fixture checklist**:
- Update `src/foundry_mcp/core/responses.py` to surface `meta.warnings` consistently.
- Regenerate golden fixtures after new fields are populated (or add optional field stubs).
- Document migration expectations in `dev_docs/codebase_standards/mcp_response_schema.md`.

**Gate**: Contract/schema/examples + fixture updates are a prerequisite to implementation changes.

**Runtime guard**: When `token_management_enabled` is false, emit empty defaults and skip summarization/budgeting logic.

### Phase 1: Token Management Foundation

**New file**: `src/foundry_mcp/core/research/token_management.py`

Define `BudgetingMode` enum (input_only, combined) and store per-provider rules in `DEFAULT_MODEL_LIMITS`.

Create centralized token utilities:

```python
@dataclass(frozen=True)
class ModelContextLimits:
    provider: str
    model: str
    context_window: int      # Total context window
    max_output_tokens: int   # Max output tokens (model cap)
    budgeting_mode: str = "input_only"  # BudgetingMode: input_only | combined
    output_reserved: int = 0            # Reserved output tokens when combined

    @property
    def input_budget(self) -> int:
        """Effective input budget for one-off requests."""
        if self.budgeting_mode == "combined":
            return max(0, self.context_window - self.output_reserved)
        return self.context_window

DEFAULT_MODEL_LIMITS = {
    # OpenAI GPT-5.2-Codex models (400K context, 128K output)
    "codex:gpt-5.2-codex": ModelContextLimits("codex", "gpt-5.2-codex", 400000, 128000, "combined", 128000),
    "cursor-agent:gpt-5.2-codex": ModelContextLimits("cursor-agent", "gpt-5.2-codex", 400000, 128000, "combined", 128000),
    "opencode:openai/gpt-5.2-codex": ModelContextLimits("opencode", "openai/gpt-5.2-codex", 400000, 128000, "combined", 128000),

    # OpenAI GPT-4.1 models (1M context, 32K output)
    "codex:gpt-4.1": ModelContextLimits("codex", "gpt-4.1", 1000000, 32000, "combined", 32000),
    "cursor-agent:gpt-4.1": ModelContextLimits("cursor-agent", "gpt-4.1", 1000000, 32000, "combined", 32000),
    "opencode:openai/gpt-4.1": ModelContextLimits("opencode", "openai/gpt-4.1", 1000000, 32000, "combined", 32000),

    # OpenAI o-series reasoning models (200K context, 100K output)
    "codex:o3": ModelContextLimits("codex", "o3", 200000, 100000, "combined", 100000),
    "codex:o4-mini": ModelContextLimits("codex", "o4-mini", 200000, 100000, "combined", 100000),
    "opencode:openai/o3": ModelContextLimits("opencode", "openai/o3", 200000, 100000, "combined", 100000),
    "opencode:openai/o4-mini": ModelContextLimits("opencode", "openai/o4-mini", 200000, 100000, "combined", 100000),

    # Claude models (200K context, 64K output)
    "claude:opus": ModelContextLimits("claude", "opus", 200000, 64000, "combined", 64000),
    "claude:sonnet": ModelContextLimits("claude", "sonnet", 200000, 64000, "combined", 64000),
    "claude:haiku": ModelContextLimits("claude", "haiku", 200000, 64000, "combined", 64000),

    # Gemini 3.0 models (1M context, 64K output for Pro, 32K for Flash)
    "gemini:flash": ModelContextLimits("gemini", "flash", 1000000, 32000, "input_only", 0),
    "gemini:pro": ModelContextLimits("gemini", "pro", 1000000, 64000, "input_only", 0),

    # Conservative default for unknown models
    "_default": ModelContextLimits("unknown", "unknown", 128000, 8192, "combined", 8192),
}

# Runtime overhead is configured via `runtime_overhead` in foundry-mcp.toml.
# User sets based on their environment:
# - Claude Code: ~60000 (system prompt + tools + memory files)
# - Gemini CLI: ~40000
# - Cursor Agent: ~40000
# - Codex CLI: ~40000
# - OpenCode CLI: ~40000

def get_model_limits(provider_id: str, config: ResearchConfig) -> ModelContextLimits:
    """Look up model limits from config overrides or defaults.

    Resolution order:
    1. config.model_context_overrides[provider_id]
    2. DEFAULT_MODEL_LIMITS[provider_id]
    3. DEFAULT_MODEL_LIMITS["_default"]
    """
    # Check config overrides first
    if provider_id in config.model_context_overrides:
        override = config.model_context_overrides[provider_id]
        base = DEFAULT_MODEL_LIMITS.get(provider_id, DEFAULT_MODEL_LIMITS["_default"])
        return ModelContextLimits(
            provider=base.provider,
            model=base.model,
            context_window=override.get("context_window", base.context_window),
            max_output_tokens=override.get("max_output_tokens", base.max_output_tokens),
            budgeting_mode=override.get("budgeting_mode", base.budgeting_mode),
            output_reserved=override.get("output_reserved", base.output_reserved),
        )

    # Check defaults
    if provider_id in DEFAULT_MODEL_LIMITS:
        return DEFAULT_MODEL_LIMITS[provider_id]

    # Fall back to conservative default
    return DEFAULT_MODEL_LIMITS["_default"]

def get_effective_context(provider_id: str, config: ResearchConfig) -> int:
    """Calculate effective context after overhead and safety margin."""
    model_limits = get_model_limits(provider_id, config)
    runtime_overhead = config.runtime_overhead
    safety_margin = config.token_safety_margin

    effective = model_limits.input_budget - runtime_overhead
    return max(0, int(effective * (1 - safety_margin)))

# Example effective context calculations:
# - claude:sonnet → (200K - 64K output_reserved) - 60K overhead = 76K, with 15% margin = ~65K
# - gemini:pro → 1M - 40K overhead = 960K, with 15% margin = ~816K
#
# Sources:
# - Claude: 200K context, 64K output (platform.claude.com/docs)
# - Gemini 3.0: 1M context, 64K output Pro / 32K Flash (ai.google.dev/gemini-api/docs)
# - GPT-4.1: 1M context, 32K output (platform.openai.com/docs/models)
# - GPT-5.2-Codex: 400K context, 128K output (platform.openai.com/docs/models)
# - o3/o4-mini: 200K context, 100K output (platform.openai.com/docs/models)

def estimate_tokens(text: str, provider_id: str, provider_manager: ProviderManager) -> int:
    """Use provider-native counters when available, else tiktoken, else heuristic.

    Cache counts by content hash + provider/model.
    """

@dataclass
class TokenBudget:
    total_budget: int
    reserved_output: int = 0
    safety_margin: float = 0.0
    used_tokens: int = 0

    def effective_budget(self) -> int
    def can_fit(self, tokens: int) -> bool
    def allocate(self, tokens: int) -> bool
```

**Budget policy**:
- Resolve model limits via `get_model_limits()` (config overrides → defaults).
- Compute per-phase budgets from `input_budget` and apply `token_safety_margin`.
- Use `runtime_overhead` config value for environment overhead.

**Budgeting formula (example)**:
- Combined model: `effective_budget = (context_window - output_reserved - runtime_overhead) * (1 - token_safety_margin)`
- Input-only model: `effective_budget = (context_window - runtime_overhead) * (1 - token_safety_margin)`

**Pre-dispatch validation (required)**:
- Before provider call, run `preflight_count(payload, provider_id)`.
- Fallback order: provider-native → tokenizer → heuristic (char/4) with extra safety margin.
- Emit `TOKEN_COUNT_ESTIMATE_USED` warning when heuristic fallback is used.
- Hard-fail only if the payload still exceeds caps after final-fit adjustments.

**Token counting spec**:
- Priority order: provider-native token counts → tokenizer library → heuristic char/4.
- Apply `token_safety_margin` (default 15-20%) to absorb estimation variance.

---

### Phase 2: Summarization Layer

**New file**: `src/foundry_mcp/core/research/summarization.py`
**New file**: `src/foundry_mcp/core/research/prompts/summarization_v1.txt`

LLM-based content compression with configurable provider and retry/fallback:

```python
class SummarizationLevel(str, Enum):
    RAW = "raw"              # No compression
    CONDENSED = "condensed"  # ~50% reduction
    KEY_POINTS = "key_points"  # ~75% reduction
    HEADLINE = "headline"    # ~90% reduction

class ContentSummarizer:
    def __init__(
        self,
        config: ResearchConfig,
        provider_manager: ProviderManager,
        cache: Optional[SummaryCache] = None,
    ):
        # Uses config.summarization_provider as primary
        # Falls back through config.summarization_providers on failure
        # Follows same retry/fallback pattern as deep research phases

    # Hardcoded retry settings
    MAX_RETRIES = 2
    RETRY_DELAY = 3.0

    async def summarize(
        content: str,
        target_level: SummarizationLevel,
        context: str,  # Research query for relevance
        target_budget: Optional[int] = None,
    ) -> SummarizationResult:
        """Summarize with retry/fallback across configured providers."""
        providers = self._get_provider_chain()  # Primary + fallbacks

        for provider_id in providers:
            for attempt in range(self.MAX_RETRIES):
                try:
                    # Chunk + map-reduce if content exceeds summarizer model context.
                    # Post-check output length, step down to more aggressive levels as needed.
                    # Truncate fallback when still over budget.
                    return await self._execute_summarization(
                        content, target_level, context, provider_id, target_budget=target_budget
                    )
                except (ProviderError, TimeoutError):
                    await asyncio.sleep(self.RETRY_DELAY)

        raise SummarizationError("All providers exhausted")

    async def batch_summarize(
        items: List[ContentItem],
        target_budget: int,
    ) -> List[SummarizationResult]
```

**Provider chain** (follows existing deep research pattern):
1. Try `summarization_provider` (primary)
2. Retry up to 2 times with 3.0s delay (hardcoded)
3. On exhaustion, move to next provider in `summarization_providers` list
4. Continue until success or all providers exhausted

**Partial failure semantics**:
- If a map-reduce chunk fails, retry once; on continued failure, mark the result with `SUMMARY_PROVIDER_FAILED` warning and fall back to truncation or drop based on budget policy.

**Prompt + output schema**:
- Prompt templates live in `src/foundry_mcp/core/research/prompts/` and are versioned internally.
- Summarization returns a normalized JSON payload per level (validated/parsed) with explicit JSON schemas per `SummarizationLevel`.
- On malformed output: retry once, then downgrade level, then treat as provider failure with structured warning.
- Cache keys include content hash + context hash + level + provider.

**SummarizationResult schema (authoritative)**:
```json
{
  "level": "key_points",
  "summary": "...",
  "key_points": ["..."],
  "source_ids": ["source-123"],
  "token_count": 1234,
  "provider_id": "gemini:flash",
  "warnings": []
}
```

Level-specific validation: `headline` requires `summary` only; `key_points` requires `summary` + `key_points`.

**Prompt guardrails**:
- Summarization prompt explicitly treats content as untrusted and ignores embedded instructions.
- Preserve source provenance (e.g., include a `source_id` field in map-reduce inputs).

**Budget enforcement**:
- Chunk inputs that exceed summarizer model limits and use map-reduce summarization.
- Re-summarize at more aggressive levels when output is still over budget.
- Truncate with warnings and fidelity metadata as last resort (hardcoded: always enabled).

**Cost optimization**:
- Default to fast/cheap model (gemini:flash) via TOML config
- Cache summaries by content hash + context hash + level + provider
- Batch multiple items when possible

---

### Phase 3: Context Budget Manager

**New file**: `src/foundry_mcp/core/research/context_budget.py`

Orchestrates token allocation by priority:

```python
class ContextBudgetManager:
    async def allocate_budget(
        items: List[ContentItem],
        budget: TokenBudget,
        strategy: AllocationStrategy,
    ) -> AllocationResult

    def compute_priority(item: ContentItem, state: DeepResearchState) -> float:
        """
        Priority score (0.0-1.0) based on:
        - Source quality (40%): HIGH=1.0, MEDIUM=0.7, LOW=0.4
        - Finding confidence (30%): CONFIRMED=1.0, HIGH=0.9, etc.
        - Recency (15%): Newer content scores higher
        - Relevance (15%): Semantic similarity to query
        """
```

**AllocationResult contract**:
```json
{
  "items": ["source-123"],
  "total_tokens": 4200,
  "fidelity_metadata": {"source-123": {"phase": "analysis", "level": "raw"}},
  "warnings": ["CONTENT_TRUNCATED"],
  "dropped_ids": []
}
```

**Allocation strategy**:
1. Calculate priority scores for all items
2. Sort by priority (highest first)
3. Allocate tokens starting with highest priority at full fidelity
4. Summarize lower-priority items to fit remaining budget
5. Apply centralized graceful-degradation steps (summarize -> headline -> truncate -> drop)

**Protected content**:
- `ContentItem` includes `protected: bool` for citations/key findings that must not be dropped.

---

### Phase 4: Deep Research Integration

**Modify**: `src/foundry_mcp/core/research/workflows/deep_research.py`

**Metadata propagation plan**:
- Populate `DeepResearchState.content_fidelity`, `dropped_content_ids`, `content_archive_hashes` during phases.
- Include these fields in `WorkflowResult.metadata` for `deep-research-report`.
- Update `src/foundry_mcp/tools/unified/research.py` to emit the metadata into the `data` payload when `token_management_enabled` is true.
- Route warnings through `success_response(..., warnings=[...])` to `meta.warnings`.

#### Phase inventory + integration points
- Planning: no summarization (define overhead only)
- Gathering: summarize sources when needed
- Analysis: budget + summarize sources
- Synthesis: budget findings + sources
- Refinement: summarize prior iteration context

**Budget enforcement**:
- Phase budgets apply per provider call; recalculated each phase iteration.
- For multi-call summarization (map-reduce), use per-chunk budgets with hardcoded global caps.
- Final-fit pass: assemble the full provider payload, re-count tokens, and reallocate/summarize if needed before dispatch.
- Hard-cap enforcement order: re-summarize → truncate → drop (if allowed) → error with remediation.
- Final-fit loop capped to 2 iterations; if still over, apply hard-cap enforcement then fail.

#### Analysis Phase (~line 2900)
```python
# Hardcoded budget percentages (not in config)
ANALYSIS_BUDGET_PERCENT = 0.80
SYNTHESIS_BUDGET_PERCENT = 0.85

async def _execute_analysis_async(self, state, provider_id, timeout):
    model_limits = get_model_limits(provider_id, config)
    effective_context = get_effective_context(provider_id, config)
    phase_budget = math.floor(effective_context * ANALYSIS_BUDGET_PERCENT)
    budget = TokenBudget(
        phase_budget,
        reserved_output=model_limits.output_reserved,
        safety_margin=config.token_safety_margin,
    )

    # Convert sources to prioritized content items
    content_items = [
        ContentItem(
            content=self._format_source_for_analysis(source),
            token_count=estimate_tokens(source.content, provider_id, provider_manager),
            priority=self._compute_source_priority(source, state),
            source_id=source.id,
        )
        for source in state.sources
    ]

    # Allocate budget with summarization as needed
    allocation = await budget_manager.allocate_budget(content_items, budget)

    # Track fidelity in state (per-item records)
    state.content_fidelity.update(allocation.fidelity_metadata)
    state.dropped_content_ids.extend(allocation.dropped_ids)
```

#### Synthesis Phase (~line 3300)
- Findings get priority (typically small, include all at full fidelity)
- Source references can be compressed more aggressively
- Apply `SYNTHESIS_BUDGET_PERCENT` (hardcoded) when computing the phase budget

#### Refinement Phase (~line 3700)
- Summarize previous iteration context to prevent unbounded growth

---

### Phase 5: Graceful Degradation (Centralized)

**Add to**: `src/foundry_mcp/core/research/context_budget.py`

Fallback chain when over budget (implemented inside the budget manager pipeline):
1. Summarize all items to KEY_POINTS level
2. If still over, summarize to HEADLINE level
3. If still over, truncate with warnings (hardcoded: truncate fallback always enabled)
4. If still over and `allow_content_dropping` is true, drop lowest priority items (hardcoded: min 3 items per phase)
5. Only fail if nothing can fit (clear error message)

**Priority guardrails** (hardcoded defaults):
- Preserve minimum fidelity ("condensed") for top-5 priority items.
- Emit distinct warning codes for priority items summarized vs low-priority drops.
- Never drop `protected` content; only summarize to minimum fidelity.
- Any downgrade/truncation/drop triggers a report-level warning in `meta.warnings` plus per-item fidelity warnings.

**Chunk-level failure handling**:
- Retry only failed chunks with a more aggressive summarization level.
- Preserve successful chunk summaries and emit warnings with `item_id` + `chunk_id`.
- If protected content alone exceeds budget, fallback to HEADLINE and emit `PROTECTED_OVERFLOW`; fail only if still over budget.

---

### Phase 6: Fidelity Tracking & Content Archive

**Modify**: `src/foundry_mcp/core/research/models.py`

Add to `DeepResearchState`:
```python
content_fidelity: Dict[str, ContentFidelityRecord]  # Tracks per-phase compression per item
dropped_content_ids: List[str]  # Items that couldn't fit
```

**Note**: Use `content_fidelity` for fidelity tracking instead of `state.metadata[...]` to avoid divergence.
**Note**: Content archive security governance is out of scope for this spec.

**Fidelity semantics**:
- Structure: `content_fidelity[item_id].phases[phase] = {level, reason, warnings, timestamp}`
- Merge rule: latest phase overwrites same-phase entry, preserve prior phases for history.
- ID mapping: use stable `ContentItem.id` (source IDs for sources/findings; append `#fragment-N` for chunked items).

**New file**: `src/foundry_mcp/core/research/content_archive.py`

File-based full content storage with TTL cleanup:

```python
class ContentArchive:
    """Archives full content to disk for later retrieval."""

    def __init__(
        self,
        config: ResearchConfig,
        storage_path: Optional[Path] = None,  # Default: research_archive_dir or research_dir/archive
    ):
        base_dir = (
            storage_path
            or config.research_archive_dir
            or (config.get_research_dir() / "archive")
        )
        self.storage_path = base_dir.expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._ensure_private_permissions()
        self.ttl_hours = config.content_archive_ttl_hours

    def archive(self, content: str, metadata: Optional[dict] = None) -> str:
        """Archive content, return hash for retrieval."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        record = {
            "content": content,
            "metadata": metadata or {},
            "archived_at": datetime.utcnow().isoformat(),
        }
        file_path = self.storage_path / f"{content_hash}.json"
        file_path.write_text(json.dumps(record))
        return content_hash

    def retrieve(self, content_hash: str) -> Optional[str]:
        """Retrieve archived content by hash."""
        file_path = self.storage_path / f"{content_hash}.json"
        if file_path.exists():
            record = json.loads(file_path.read_text())
            return record["content"]
        return None

    def cleanup_expired(self) -> int:
        """Remove files older than TTL. Returns count removed."""
        cutoff = datetime.utcnow() - timedelta(hours=self.ttl_hours)
        removed = 0
        for file_path in self.storage_path.glob("*.json"):
            record = json.loads(file_path.read_text())
            archived_at = datetime.fromisoformat(record["archived_at"])
            if archived_at < cutoff:
                file_path.unlink()
                removed += 1
        return removed
```

**Integration**: Archive original content before summarization, store hash in fidelity metadata, and only when `content_archive_enabled` is true. Use the configured archive directory and ensure private permissions.

**Archive guardrails**:
- Disabled by default; on write failure or read-only environments, skip archival and emit `ARCHIVE_WRITE_FAILED` warning.
- Startup capability check: if archive path is not writable, disable archival for the session and continue without failures.
- Cache archive-disabled state for the session to avoid repeated warnings/latency.
- Store `archive_writable=false` in state/config and short-circuit all `archive()` calls.

---

### Phase 7: Configuration

**Modify**: `src/foundry_mcp/config.py` - Add to `ResearchConfig`:

```python
# Token management (essential options only)
token_management_enabled: bool = True
token_safety_margin: float = 0.15  # 15% buffer (absorbs estimation variance)
runtime_overhead: int = 60000  # Tokens consumed by host CLI (Claude Code default)

# Model limit overrides (provider:model -> context_window/output_reserved/budgeting_mode)
model_context_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

# Summarization provider (follows existing per-phase pattern)
summarization_provider: Optional[str] = None  # Primary provider
summarization_providers: List[str] = field(default_factory=list)  # Fallback list
summarization_timeout: float = 120.0  # Shorter - summaries are quick
summarization_cache_enabled: bool = True

# Graceful degradation
allow_content_dropping: bool = True

# Content archive (security governance out of scope)
content_archive_enabled: bool = False
content_archive_ttl_hours: int = 72  # Keep full content for 3 days
research_archive_dir: Optional[Path] = None  # Default: research_dir/archive

# --- Hardcoded defaults (not exposed in config) ---
# analysis_budget_percent = 0.80
# synthesis_budget_percent = 0.85
# summarization_max_retries = 2
# summarization_retry_delay = 3.0
# min_items_per_phase = 3
# min_fidelity_top_items = "condensed"
# min_fidelity_top_count = 5
# summarization_truncate_fallback = True
# summarization_require_structured_output = True
```

**Config surface rationale**:
- Keep ~12 essential options that users are likely to tune
- Hardcode sensible defaults for internal parameters
- Add options when users request them (YAGNI principle)

**Modify**: `samples/foundry-mcp.toml` - Add new section:

```toml
# -----------------------------------------------------------------------------
# Token Management
# -----------------------------------------------------------------------------
# Controls proactive token budgeting and content summarization.

# Enable proactive token management with summarization
token_management_enabled = true

# Safety margin (% of context to reserve for estimation variance)
token_safety_margin = 0.15

# Runtime environment overhead (tokens consumed by host CLI system prompt)
# Claude Code: ~60000, Gemini CLI: ~40000, Cursor/Codex/OpenCode: ~40000
runtime_overhead = 60000

# Optional model limit overrides (provider:model -> context_window/output_reserved)
# model_context_overrides = { "gemini:flash" = { context_window = 1000000, output_reserved = 8192 } }

# Primary summarization provider (fast model recommended)
summarization_provider = "[cli]gemini:flash"

# Fallback providers for summarization (ordered list)
summarization_providers = [
    "[cli]gemini:flash",
    "[cli]claude:haiku",
    "[cli]gemini:pro",
]

# Summarization timeout (shorter than analysis - summaries are quick)
summarization_timeout = 120.0

# Cache summaries to avoid redundant LLM calls
summarization_cache_enabled = true

# Allow dropping low-priority content when over budget
allow_content_dropping = true

# -----------------------------------------------------------------------------
# Content Archive (Full Content Retrieval)
# -----------------------------------------------------------------------------
# Stores original content before summarization for drill-down access.

content_archive_enabled = false
content_archive_ttl_hours = 72  # 3 days
# Default: research_dir/archive if not set
research_archive_dir = "./specs/.research_archive"
```

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/foundry_mcp/core/research/token_management.py` | CREATE | Token estimation, model limits (Python dict), budgeting |
| `src/foundry_mcp/core/research/summarization.py` | CREATE | LLM-based content compression with retry/fallback |
| `src/foundry_mcp/core/research/prompts/summarization_v1.txt` | CREATE | Versioned summarization prompt template |
| `src/foundry_mcp/core/research/context_budget.py` | CREATE | Priority-based budget allocation |
| `src/foundry_mcp/core/research/content_archive.py` | CREATE | File-based full content storage with TTL |
| `src/foundry_mcp/core/research/workflows/deep_research.py` | MODIFY | Integration in analysis/synthesis/refinement |
| `src/foundry_mcp/core/research/models.py` | MODIFY | Add fidelity tracking fields |
| `src/foundry_mcp/core/research/state_migrations.py` | CREATE | State schema versioning + migrations |
| `src/foundry_mcp/config.py` | MODIFY | Add token management config (~12 options) |
| `samples/foundry-mcp.toml` | MODIFY | Add token management + archive configuration |
| `dev_docs/codebase_standards/mcp_response_schema.md` | MODIFY | Document fidelity metadata + dropped IDs |
| `dev_docs/mcp_best_practices/07-error-semantics.md` | MODIFY | Document warning taxonomy + semantics |
| `dev_docs/codebase_standards/cli-output.md` | MODIFY | Align CLI warning output with response schema |
| `src/foundry_mcp/core/responses.py` | MODIFY | Surface meta.warnings for degraded outputs |
| `CHANGELOG.md` | MODIFY | Record response schema addition |
| `docs/concepts/deep_research_workflow.md` | MODIFY | Document new token management flow + metadata |
| `tests/fixtures/golden/*.json` | MODIFY | Update response examples if schema changes |
| `tests/core/research/test_token_management.py` | CREATE | Unit tests |
| `tests/core/research/test_summarization.py` | CREATE | Summarization tests |
| `tests/core/research/test_context_budget.py` | CREATE | Budget allocation tests |
| `tests/core/research/test_content_archive.py` | CREATE | Archive storage tests |

---

## Verification

1. **Unit tests**: Token estimation accuracy, budget math with safety margin, `get_model_limits()` resolution order
2. **Provider matrix tests**: BudgetingMode + output_reserved rules for key providers in DEFAULT_MODEL_LIMITS
3. **Summarization tests**: Chunking/map-reduce, re-summarize to tighter levels, truncation fallback behavior, schema validation
4. **Cache tests**: Context/provider changes produce cache misses; same inputs hit
5. **Archive tests**: Honors `research_archive_dir`, private permissions, TTL cleanup, corrupted JSON handling
6. **Integration tests**: Run deep research with artificially low model limits across analysis→synthesis→refinement, verify graceful degradation + min item guardrails
7. **Fidelity checks**: Verify metadata accurately reflects compression applied and dropped items
8. **Schema propagation tests**: With `token_management_enabled`, ensure response payload includes fidelity fields + `meta.warnings`
9. **Manual testing**: Run full deep research, check report quality with/without token management

---

## Key Design Decisions

1. **Simple model limits**: Python dict as single source of truth, one config override dict, simple lookup function.
2. **Fixed safety margin**: Single `token_safety_margin` config (default 15-20%) instead of dynamic calibration.
3. **User-configured runtime overhead**: Single `runtime_overhead` config value; user sets based on their CLI environment.
4. **Minimal config surface**: ~12 essential options; hardcode sensible defaults for internal parameters.
5. **Chunked summarization with fallback**: Map-reduce for oversized inputs, re-summarize aggressively, optional truncation as last resort.
6. **Context-aware caching**: Cache keys include context hash and provider/model to prevent cross-query reuse.
7. **Private, opt-in content archival**: Archive under the configured research archive dir with strict permissions.
8. **Progressive degradation with guardrails**: Summarize before dropping and honor `min_items_per_phase`.
9. **Backwards compatible**: New state fields have defaults and migrations so existing sessions load correctly.
10. **Provider-aware budgeting**: Combined vs input-only budgets are explicit in DEFAULT_MODEL_LIMITS.
