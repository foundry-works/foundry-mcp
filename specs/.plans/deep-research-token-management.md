# Deep Research: Robust Token Management & Content Fidelity

## Problem Statement

The current deep research system uses reactive context management (catches errors after overflow) and character-based truncation (loses semantic content). This can lead to:
- Phase failures on context overflow
- Lost information from hard truncation
- No visibility into what content was compressed/dropped

## Solution Overview

Transform the system to use **proactive token management** with **intelligent summarization** and **graceful degradation**.

---

## Implementation Phases

### Phase 1: Token Management Foundation

**New file**: `src/foundry_mcp/core/research/token_management.py`

Create centralized token utilities:

```python
@dataclass(frozen=True)
class ModelContextLimits:
    provider: str
    model: str
    context_window: int      # Max input tokens
    max_output_tokens: int   # Max output tokens (informational, not subtracted)

    @property
    def input_budget(self) -> int:
        """For one-off requests, full context window is available for input."""
        return self.context_window

DEFAULT_MODEL_LIMITS = {
    # OpenAI GPT-5.2-Codex models (400K context, 128K output)
    "codex:gpt-5.2-codex": ModelContextLimits("codex", "gpt-5.2-codex", 400000, 128000),
    "cursor-agent:gpt-5.2-codex": ModelContextLimits("cursor-agent", "gpt-5.2-codex", 400000, 128000),
    "opencode:openai/gpt-5.2-codex": ModelContextLimits("opencode", "openai/gpt-5.2-codex", 400000, 128000),

    # OpenAI GPT-4.1 models (1M context, 32K output)
    "codex:gpt-4.1": ModelContextLimits("codex", "gpt-4.1", 1000000, 32000),
    "cursor-agent:gpt-4.1": ModelContextLimits("cursor-agent", "gpt-4.1", 1000000, 32000),
    "opencode:openai/gpt-4.1": ModelContextLimits("opencode", "openai/gpt-4.1", 1000000, 32000),

    # OpenAI o-series reasoning models (200K context, 100K output)
    "codex:o3": ModelContextLimits("codex", "o3", 200000, 100000),
    "codex:o4-mini": ModelContextLimits("codex", "o4-mini", 200000, 100000),
    "opencode:openai/o3": ModelContextLimits("opencode", "openai/o3", 200000, 100000),
    "opencode:openai/o4-mini": ModelContextLimits("opencode", "openai/o4-mini", 200000, 100000),

    # Claude models (200K context, 64K output)
    "claude:opus": ModelContextLimits("claude", "opus", 200000, 64000),
    "claude:sonnet": ModelContextLimits("claude", "sonnet", 200000, 64000),
    "claude:haiku": ModelContextLimits("claude", "haiku", 200000, 64000),

    # Gemini 3.0 models (1M context, 64K output for Pro, 32K for Flash)
    "gemini:flash": ModelContextLimits("gemini", "flash", 1000000, 32000),
    "gemini:pro": ModelContextLimits("gemini", "pro", 1000000, 64000),

    # Conservative default for unknown models
    "_default": ModelContextLimits("unknown", "unknown", 128000, 8192),
}

# CLI System Prompt Overhead (tokens consumed before user input)
# Use these to estimate effective available context for deep research.
#
# ┌─────────────────┬──────────────┬─────────────────────────────────────────────┐
# │ CLI             │ System Prompt│ Notes                                       │
# │                 │ Overhead     │                                             │
# ├─────────────────┼──────────────┼─────────────────────────────────────────────┤
# │ Claude Code     │ ~60k         │ System prompt + tools + memory files        │
# │                 │              │ Git repo info, CLAUDE.md, conversation      │
# │                 │              │ Fresh session baseline: 45-63k observed     │
# ├─────────────────┼──────────────┼─────────────────────────────────────────────┤
# │ Gemini CLI      │ ~40k         │ Estimated (limited public data)             │
# │                 │              │ prompts.ts is ~4-5k but full context varies │
# ├─────────────────┼──────────────┼─────────────────────────────────────────────┤
# │ Cursor Agent    │ ~40k         │ Estimated (limited public data)             │
# │                 │              │ Tool definitions + dynamic context          │
# ├─────────────────┼──────────────┼─────────────────────────────────────────────┤
# │ Codex CLI       │ ~40k         │ Estimated (limited public data)             │
# │                 │              │ AGENTS.md + environment context             │
# ├─────────────────┼──────────────┼─────────────────────────────────────────────┤
# │ OpenCode CLI    │ ~40k         │ Estimated (limited public data)             │
# │                 │              │ Environment context added on top            │
# └─────────────────┴──────────────┴─────────────────────────────────────────────┘

CLI_SYSTEM_PROMPT_OVERHEAD = {
    "claude": 60000,      # Claude Code - measured baseline 45-63k
    "gemini": 40000,      # Gemini CLI - estimated (limited data)
    "cursor-agent": 40000, # Cursor Agent - estimated (limited data)
    "codex": 40000,       # Codex CLI - estimated (limited data)
    "opencode": 40000,    # OpenCode CLI - estimated (limited data)
    "_default": 40000,    # Conservative default for unknown CLIs
}

def get_effective_context(provider_id: str, registry: ModelContextRegistry) -> int:
    """Calculate effective context after CLI overhead.

    For one-off requests, output tokens don't consume input context.
    Only CLI system prompt overhead is subtracted.
    """
    model_limits = registry.resolve(provider_id)
    cli_overhead = CLI_SYSTEM_PROMPT_OVERHEAD.get(
        provider_id.split(":")[0],  # Extract CLI prefix
        CLI_SYSTEM_PROMPT_OVERHEAD["_default"]
    )
    return model_limits.context_window - cli_overhead

# Example effective context calculations (one-off requests, output doesn't consume input):
# - claude:sonnet → 200K - 60K (overhead) = 140K effective input
# - gemini:pro → 1M - 40K (overhead) = 960K effective input
# - codex:gpt-4.1 → 1M - 40K (overhead) = 960K effective input
#
# Sources:
#
# Model Limits:
# - Claude: 200K context, 64K output (platform.claude.com/docs)
# - Gemini 3.0: 1M context, 64K output Pro / 32K Flash (ai.google.dev/gemini-api/docs)
# - GPT-4.1: 1M context, 32K output (platform.openai.com/docs/models)
# - GPT-5.2-Codex: 400K context, 128K output (platform.openai.com/docs/models)
# - o3/o4-mini: 200K context, 100K output (platform.openai.com/docs/models)
#
# CLI System Prompt Overhead:
# - Claude Code: ~50k measured (45-63k observed fresh session baseline)
# - Others: ~30k estimated (limited public data, conservative default)

class ModelContextRegistry:
    def __init__(self, config: ResearchConfig, provider_manager: ProviderManager):
        self.overrides = config.model_context_overrides
        self.provider_manager = provider_manager

    def normalize_provider_id(self, provider_id: str) -> str:
        """Strip transport prefixes and resolve aliases to provider:model."""

    def resolve(self, provider_id: str) -> ModelContextLimits:
        """Provider metadata -> config overrides -> defaults -> _default."""

def estimate_tokens(text: str, provider_id: str, provider_manager: ProviderManager) -> int:
    """Use provider-native counters when available, else tiktoken, else heuristic."""

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
- Resolve model limits via the registry (provider metadata + config overrides + defaults).
- Compute per-phase budgets from `input_budget` and apply `token_safety_margin`.

---

### Phase 2: Summarization Layer

**New file**: `src/foundry_mcp/core/research/summarization.py`

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

    async def summarize(
        content: str,
        target_level: SummarizationLevel,
        context: str,  # Research query for relevance
        target_budget: Optional[int] = None,
    ) -> SummarizationResult:
        """Summarize with retry/fallback across configured providers."""
        providers = self._get_provider_chain()  # Primary + fallbacks

        for provider_id in providers:
            for attempt in range(config.summarization_max_retries):
                try:
                    # Chunk + map-reduce if content exceeds summarizer model context.
                    # Post-check output length, step down to more aggressive levels as needed.
                    # Optional truncate fallback when still over budget.
                    return await self._execute_summarization(
                        content, target_level, context, provider_id, target_budget=target_budget
                    )
                except (ProviderError, TimeoutError):
                    await asyncio.sleep(config.summarization_retry_delay)

        raise SummarizationError("All providers exhausted")

    async def batch_summarize(
        items: List[ContentItem],
        target_budget: int,
    ) -> List[SummarizationResult]
```

**Provider chain** (follows existing deep research pattern):
1. Try `summarization_provider` (primary)
2. Retry up to `summarization_max_retries` times with `summarization_retry_delay`
3. On exhaustion, move to next provider in `summarization_providers` list
4. Continue until success or all providers exhausted

**Budget enforcement**:
- Chunk inputs that exceed summarizer model limits and use map-reduce summarization.
- Re-summarize at more aggressive levels when output is still over budget.
- If still over and `summarization_truncate_fallback` is enabled, truncate with warnings and fidelity metadata.

**Cost optimization**:
- Default to fast/cheap model (gemini:flash) via TOML config
- Cache summaries by content hash + context hash + level + provider + prompt version
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

**Allocation strategy**:
1. Calculate priority scores for all items
2. Sort by priority (highest first)
3. Allocate tokens starting with highest priority at full fidelity
4. Summarize lower-priority items to fit remaining budget
5. Drop items only if even headline summary doesn't fit, `allow_content_dropping` is true, and `min_items_per_phase` is still satisfied

---

### Phase 4: Deep Research Integration

**Modify**: `src/foundry_mcp/core/research/workflows/deep_research.py`

#### Analysis Phase (~line 2900)
```python
async def _execute_analysis_async(self, state, provider_id, timeout):
    registry = ModelContextRegistry(config, provider_manager)
    normalized_id = registry.normalize_provider_id(provider_id)
    model_limits = registry.resolve(normalized_id)
    phase_budget = math.floor(model_limits.input_budget * config.analysis_budget_percent)
    budget = TokenBudget(
        phase_budget,
        reserved_output=model_limits.output_reserved,
        safety_margin=config.token_safety_margin,
    )

    # Convert sources to prioritized content items
    content_items = [
        ContentItem(
            content=self._format_source_for_analysis(source),
            token_count=estimate_tokens(..., normalized_id, provider_manager),
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
- Apply `synthesis_budget_percent` and `token_safety_margin` when computing the phase budget

#### Refinement Phase (~line 3700)
- Summarize previous iteration context to prevent unbounded growth

---

### Phase 5: Graceful Degradation

**Add to**: `src/foundry_mcp/core/research/context_budget.py`

Fallback chain when over budget:
1. Summarize all items to KEY_POINTS level
2. If still over, summarize to HEADLINE level
3. If still over and `summarization_truncate_fallback` is enabled, truncate with warnings
4. If still over and `allow_content_dropping` is true, drop lowest priority items while honoring `min_items_per_phase`
5. Only fail if nothing can fit (clear error message)

```python
class GracefulDegradationHandler:
    async def handle_overflow(
        items: List[ContentItem],
        budget: TokenBudget,
        phase: str,
    ) -> DegradedResult
```

---

### Phase 6: Fidelity Tracking & Content Archive

**Modify**: `src/foundry_mcp/core/research/models.py`

Add to `DeepResearchState`:
```python
content_fidelity: Dict[str, ContentFidelityRecord]  # Tracks per-phase compression per item
dropped_content_ids: List[str]  # Items that couldn't fit
```

**Note**: Use `content_fidelity` for fidelity tracking instead of `state.metadata[...]` to avoid divergence.

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

---

### Phase 7: Configuration

**Modify**: `src/foundry_mcp/config.py` - Add to `ResearchConfig`:

```python
# Token management
token_management_enabled: bool = True
token_safety_margin: float = 0.10  # 10% buffer

# Model limit overrides (provider:model -> context_window/output_reserved)
model_context_overrides: Dict[str, Dict[str, int]] = field(default_factory=dict)

# Per-phase budgets (% of model limit)
analysis_budget_percent: float = 0.80
synthesis_budget_percent: float = 0.85

# Summarization provider (follows existing per-phase pattern)
summarization_provider: Optional[str] = None  # Primary provider
summarization_providers: List[str] = field(default_factory=list)  # Fallback list
summarization_timeout: float = 120.0  # Shorter - summaries are quick
summarization_max_retries: int = 2
summarization_retry_delay: float = 3.0
summarization_cache_enabled: bool = True
summarization_cache_ttl_hours: int = 24
summarization_prompt_version: str = "v1"
summarization_truncate_fallback: bool = True

# Graceful degradation
allow_content_dropping: bool = True
min_items_per_phase: int = 3

# Content archive
content_archive_enabled: bool = False
content_archive_ttl_hours: int = 72  # Keep full content for 3 days
research_archive_dir: Optional[Path] = None  # Default: research_dir/archive
```

**Modify**: `samples/foundry-mcp.toml` - Add new section:

```toml
# -----------------------------------------------------------------------------
# Content Summarization (Token Management)
# -----------------------------------------------------------------------------
# Controls LLM-based content compression for context management.
# Uses fast models by default; can be overridden for quality.

# Enable proactive token management with summarization
token_management_enabled = true

# Safety margin (% of context window to reserve)
token_safety_margin = 0.10

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

# Retry settings
summarization_max_retries = 2
summarization_retry_delay = 3.0

# Cache summaries to avoid redundant LLM calls
summarization_cache_enabled = true
summarization_cache_ttl_hours = 24
summarization_prompt_version = "v1"
summarization_truncate_fallback = true

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
| `src/foundry_mcp/core/research/token_management.py` | CREATE | Token estimation, model limits, budgeting |
| `src/foundry_mcp/core/research/summarization.py` | CREATE | LLM-based content compression with retry/fallback |
| `src/foundry_mcp/core/research/context_budget.py` | CREATE | Priority-based budget allocation |
| `src/foundry_mcp/core/research/content_archive.py` | CREATE | File-based full content storage with TTL |
| `src/foundry_mcp/core/research/workflows/deep_research.py` | MODIFY | Integration in analysis/synthesis/refinement |
| `src/foundry_mcp/core/research/models.py` | MODIFY | Add fidelity tracking fields |
| `src/foundry_mcp/config.py` | MODIFY | Add token management + summarization config |
| `samples/foundry-mcp.toml` | MODIFY | Add summarization + archive configuration |
| `tests/core/research/test_token_management.py` | CREATE | Unit tests |
| `tests/core/research/test_summarization.py` | CREATE | Summarization tests |
| `tests/core/research/test_context_budget.py` | CREATE | Budget allocation tests |
| `tests/core/research/test_content_archive.py` | CREATE | Archive storage tests |

---

## Verification

1. **Unit tests**: Token estimation accuracy, provider ID normalization, budget math with safety margin + per-phase percent
2. **Summarization tests**: Chunking/map-reduce, re-summarize to tighter levels, truncation fallback behavior
3. **Cache tests**: Context/provider/prompt version changes produce cache misses; same inputs hit
4. **Archive tests**: Honors `research_archive_dir`, private permissions, TTL cleanup, corrupted JSON handling
5. **Integration tests**: Run deep research with artificially low model limits, verify graceful degradation + min item guardrails
6. **Fidelity checks**: Verify metadata accurately reflects compression applied and dropped items
7. **Manual testing**: Run full deep research, check report quality with/without token management

---

## Key Design Decisions

1. **Registry-driven model limits**: Normalize provider IDs and allow config overrides before falling back to defaults.
2. **Safety margin for token counts**: Use a buffer to absorb tokenizer/provider differences and prompt overhead.
3. **Chunked summarization with fallback**: Map-reduce for oversized inputs, re-summarize aggressively, optional truncation as last resort.
4. **Context-aware caching**: Cache keys include context hash, provider/model, and prompt version to prevent cross-query reuse.
5. **Private, opt-in content archival**: Archive under the configured research archive dir with strict permissions.
6. **Progressive degradation with guardrails**: Summarize before dropping and honor `min_items_per_phase`.
7. **Backwards compatible**: New state fields have defaults so existing sessions load correctly.
