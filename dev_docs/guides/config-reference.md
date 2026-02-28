# Configuration Reference

Complete reference for all `foundry-mcp` configuration fields.

**Quick links:**
- [Config File Locations](#config-file-locations)
- [Provider Spec Format](#provider-spec-format)
- [Sub-Table Syntax](#sub-table-syntax)
- [Environment Variables](#environment-variables)
- [Deprecated Fields](#deprecated-fields)

---

## Config File Locations

| Priority | Location | Notes |
|----------|----------|-------|
| 1 (highest) | Environment variables | Runtime overrides |
| 2 | `./foundry-mcp.toml` | Project config |
| 3 | `~/.foundry-mcp.toml` | User config (legacy) |
| 4 | `~/.config/foundry-mcp/config.toml` | User config (XDG) |
| 5 (lowest) | Built-in defaults | Hardcoded in dataclasses |

Override the config file path with `FOUNDRY_MCP_CONFIG_FILE=/path/to/config.toml`.

---

## Provider Spec Format

Provider specs identify an LLM provider and model. Format: `[cli]provider:model`

```
[cli]gemini:pro             → Gemini CLI, pro model
[cli]claude:opus            → Claude CLI, opus model
[cli]codex:gpt-5.2-codex   → Codex CLI, gpt-5.2-codex model
[cli]opencode:openai/gpt-5.2-codex → OpenCode CLI routing to OpenAI backend
```

Simple names like `"gemini"` or `"claude"` are also accepted (resolved to default model).

---

## Sub-Table Syntax

Several `[research]` sub-sections support both flat keys and nested tables. **Flat keys take priority.**

```toml
# Flat keys (take priority):
[research]
tavily_search_depth = "advanced"
deep_research_max_iterations = 5

# Equivalent sub-table form:
[research.tavily]
search_depth = "advanced"

[research.deep_research]
max_iterations = 5
```

Supported sub-tables: `[research.deep_research]`, `[research.tavily]`, `[research.perplexity]`, `[research.semantic_scholar]`.

---

## Section: [providers]

Provider aliases — short names expanded at parse time in all `*_provider` string fields, `*_providers` lists, tier values, and consultation priority lists.

```toml
[providers]
pro  = "[cli]gemini:pro"
fast = "[cli]gemini:flash"
opus = "[cli]claude:opus"
```

Then: `default_provider = "pro"` expands to `"[cli]gemini:pro"`.

---

## Section: [workspace]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `specs_dir` | str | `None` (auto-detected) | `FOUNDRY_MCP_SPECS_DIR` | Spec files directory |
| `research_dir` | str | `None` (→ specs_dir/.research) | `FOUNDRY_MCP_RESEARCH_DIR` | Research state storage |

---

## Section: [logging]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `level` | str | `"INFO"` | `FOUNDRY_MCP_LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR |
| `structured` | bool | `true` | — | JSON (true) or human-readable (false) |

---

## Section: [tools]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `disabled_tools` | list[str] | `[]` | `FOUNDRY_MCP_DISABLED_TOOLS` | Tools to disable (comma-separated in env) |

Available tools: `health`, `plan`, `pr`, `error`, `journal`, `authoring`, `review`, `spec`, `task`, `provider`, `environment`, `lifecycle`, `verification`, `server`, `test`, `research`.

---

## Section: [git]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `enabled` | bool | `false` | `FOUNDRY_MCP_GIT_ENABLED` | Enable git-aware workflows |
| `auto_commit` | bool | `false` | `FOUNDRY_MCP_GIT_AUTO_COMMIT` | Auto-commit after task/phase |
| `auto_push` | bool | `false` | `FOUNDRY_MCP_GIT_AUTO_PUSH` | Auto-push after commit |
| `auto_pr` | bool | `false` | `FOUNDRY_MCP_GIT_AUTO_PR` | Auto-create PR after push |
| `commit_cadence` | str | `"manual"` | `FOUNDRY_MCP_GIT_COMMIT_CADENCE` | `"manual"` \| `"task"` \| `"phase"` |
| `show_before_commit` | bool | `true` | `FOUNDRY_MCP_GIT_SHOW_PREVIEW` | Show staged diff before commit |

---

## Section: [autonomy_posture]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `profile` | str? | `null` | `FOUNDRY_MCP_AUTONOMY_POSTURE` | `"unattended"` \| `"supervised"` \| `"debug"` |

### Posture Profile Defaults

| Setting | unattended | supervised | debug |
|---------|-----------|------------|-------|
| `autonomy_security.role` | `autonomy_runner` | `maintainer` | `maintainer` |
| `allow_lock_bypass` | false | true | true |
| `allow_gate_waiver` | false | true | true |
| `enforce_required_phase_gates` | true | true | false |
| `gate_policy` | `strict` | `strict` | `manual` |
| `stop_on_phase_completion` | true | true | false |
| `auto_retry_fidelity_gate` | true | true | false |
| `max_tasks_per_session` | 100 | 100 | 250 |
| `max_consecutive_errors` | 3 | 5 | 20 |
| `max_fidelity_review_cycles_per_phase` | 3 | 3 | 10 |

---

## Section: [autonomy_security]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `role` | str | `"maintainer"` | `FOUNDRY_MCP_ROLE` | `"maintainer"` \| `"autonomy_runner"` |
| `allow_lock_bypass` | bool | `false` | `FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_LOCK_BYPASS` | Allow bypassing task locks |
| `allow_gate_waiver` | bool | `false` | `FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_GATE_WAIVER` | Allow waiving phase gates |
| `enforce_required_phase_gates` | bool | `true` | `FOUNDRY_MCP_AUTONOMY_SECURITY_ENFORCE_REQUIRED_PHASE_GATES` | Enforce required gates |
| `rate_limit_max_consecutive_denials` | int | `10` | `FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_MAX_CONSECUTIVE_DENIALS` | Max denials before rate-limiting |
| `rate_limit_denial_window_seconds` | int | `60` | `FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_DENIAL_WINDOW_SECONDS` | Window for denial counting |
| `rate_limit_retry_after_seconds` | int | `5` | `FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_RETRY_AFTER_SECONDS` | Cooldown after rate limit |

---

## Section: [autonomy_session_defaults]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `gate_policy` | str | `"strict"` | `FOUNDRY_MCP_AUTONOMY_DEFAULT_GATE_POLICY` | `"strict"` \| `"lenient"` \| `"manual"` |
| `stop_on_phase_completion` | bool | `false` | `FOUNDRY_MCP_AUTONOMY_DEFAULT_STOP_ON_PHASE_COMPLETION` | Pause when phase completes |
| `auto_retry_fidelity_gate` | bool | `true` | `FOUNDRY_MCP_AUTONOMY_DEFAULT_AUTO_RETRY_FIDELITY_GATE` | Auto-retry after gate failure |
| `max_tasks_per_session` | int | `100` | `FOUNDRY_MCP_AUTONOMY_DEFAULT_MAX_TASKS_PER_SESSION` | Max tasks before session ends |
| `max_consecutive_errors` | int | `3` | `FOUNDRY_MCP_AUTONOMY_DEFAULT_MAX_CONSECUTIVE_ERRORS` | Max consecutive errors |
| `max_fidelity_review_cycles_per_phase` | int | `3` | `FOUNDRY_MCP_AUTONOMY_DEFAULT_MAX_FIDELITY_REVIEW_CYCLES_PER_PHASE` | Max gate retry cycles |

---

## Section: [consultation]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `priority` | list[str] | `[]` | `FOUNDRY_MCP_CONSULTATION_PRIORITY` | Provider specs in priority order |
| `default_timeout` | float | `300.0` | `FOUNDRY_MCP_CONSULTATION_TIMEOUT` | Seconds per provider call |
| `max_retries` | int | `2` | `FOUNDRY_MCP_CONSULTATION_MAX_RETRIES` | Retry attempts per provider |
| `retry_delay` | float | `5.0` | `FOUNDRY_MCP_CONSULTATION_RETRY_DELAY` | Seconds between retries |
| `fallback_enabled` | bool | `true` | `FOUNDRY_MCP_CONSULTATION_FALLBACK_ENABLED` | Fall back to next provider |
| `cache_ttl` | int | `3600` | `FOUNDRY_MCP_CONSULTATION_CACHE_TTL` | Cache TTL in seconds |

### Per-Provider Overrides

```toml
[consultation.overrides]
"[cli]opencode:openai/gpt-5.2-codex" = { timeout = 600 }
```

### Per-Workflow Overrides

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_models` | int | `1` | Minimum models for consensus |
| `timeout_override` | float? | `null` | Workflow-specific timeout |

```toml
[consultation.workflows.fidelity_review]
min_models = 2
timeout_override = 600.0
```

---

## Section: [research] — Core Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Master switch for research tools |
| `default_provider` | str | `"gemini"` | Default LLM provider spec |
| `consensus_providers` | list[str] | `["gemini", "claude"]` | Providers for consensus workflow |
| `default_timeout` | float | `360.0` | Default provider call timeout (seconds) |
| `ttl_hours` | int | `24` | Thread/state TTL before cleanup (hours) |
| `max_messages_per_thread` | int | `100` | Max messages per conversation thread |
| `thinkdeep_max_depth` | int | `5` | Max investigation depth |
| `ideate_perspectives` | list[str] | `["technical", "creative", "practical", "visionary"]` | Brainstorming perspectives |
| `search_rate_limit` | int | `60` | Global requests per minute |
| `max_concurrent_searches` | int | `3` | Max concurrent search requests |
| `audit_verbosity` | str | `"full"` | `"full"` \| `"minimal"` |

---

## Section: [research] — Deep Research

### Core Workflow

| Field | Type | Default | Bounds | Description |
|-------|------|---------|--------|-------------|
| `deep_research_max_iterations` | int | `3` | 1–20 | Max refinement iterations |
| `deep_research_max_sub_queries` | int | `5` | 1–50 | Max sub-queries per decomposition |
| `deep_research_max_sources` | int | `5` | 1–100 | Max sources per sub-query |
| `deep_research_max_concurrent` | int | `3` | 1–20 | Max parallel operations |
| `deep_research_timeout` | float | `2400.0` | ≤3600 | Whole workflow timeout (seconds) |
| `deep_research_follow_links` | bool | `true` | — | Follow and extract URLs |
| `deep_research_audit_artifacts` | bool | `true` | — | Write audit artifacts |
| `deep_research_mode` | str | `"general"` | — | `"general"` \| `"academic"` \| `"technical"` |
| `deep_research_providers` | list[str] | `["tavily", "google", "semantic_scholar"]` | — | Search providers |
| `deep_research_stale_task_seconds` | float | `300.0` | — | Task staleness threshold |
| `status_persistence_throttle_seconds` | int | `5` | — | Min seconds between status writes |

### Clarification Phase

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_research_allow_clarification` | bool | `true` | Enable query clarification |
| `deep_research_clarification_provider` | str? | `null` | Provider (→ default_provider) |
| `deep_research_clarification_model` | str? | `null` | Model override |

### Supervision (Coverage Assessment)

| Field | Type | Default | Bounds | Description |
|-------|------|---------|--------|-------------|
| `deep_research_enable_supervision` | bool | `true` | — | Enable supervision loop |
| `deep_research_max_supervision_rounds` | int | `6` | 1–20 | Max supervision rounds |
| `deep_research_supervision_min_sources_per_query` | int | `2` | — | Min sources per query |
| `deep_research_coverage_confidence_threshold` | float | `0.75` | 0.0–1.0 | Confidence target |
| `deep_research_supervision_wall_clock_timeout` | float | `1800.0` | — | Wall-clock limit (seconds) |
| `deep_research_supervision_provider` | str? | `null` | — | Provider override |
| `deep_research_supervision_model` | str? | `null` | — | Model override |
| `deep_research_max_concurrent_research_units` | int | `5` | 1–20 | Max parallel research units |
| `deep_research_delegation_provider` | str? | `null` | — | Delegation provider |
| `deep_research_delegation_model_name` | str? | `null` | — | Delegation model |

### Topic Agents (Gathering)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_research_topic_max_tool_calls` | int | `10` | Max tool calls per topic |
| `deep_research_enable_extract` | bool | `true` | Allow URL content extraction |
| `deep_research_extract_max_per_iteration` | int | `2` | Max URLs per iteration |
| `deep_research_enable_content_dedup` | bool | `true` | Deduplicate across topics |
| `deep_research_content_dedup_threshold` | float | `0.8` | Similarity threshold (0.0–1.0) |
| `deep_research_topic_reflection_provider` | str? | `null` | Per-topic reflection provider |
| `deep_research_topic_reflection_model` | str? | `null` | Per-topic reflection model |

> **Backward-compat alias:** `deep_research_topic_max_searches` → `deep_research_topic_max_tool_calls`

### Fetch-Time Summarization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_research_summarization_provider` | str? | `null` | Provider override |
| `deep_research_summarization_model` | str? | `null` | Model override |
| `deep_research_max_content_length` | int | `50000` | Max chars before summarizing |
| `deep_research_summarization_timeout` | int | `60` | Timeout per request (seconds) |
| `deep_research_summarization_min_content_length` | int | `300` | Min chars to summarize |
| `deep_research_inline_compression` | bool | `true` | Compress inline during gathering |

### Per-Topic Compression

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_research_compression_provider` | str? | `null` | Provider override |
| `deep_research_compression_model` | str? | `null` | Model override |
| `deep_research_compression_max_content_length` | int | `50000` | Max chars |

### Document Digest

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_research_digest_min_chars` | int | `10000` | Min chars before digesting |
| `deep_research_digest_max_sources` | int | `8` | Max sources per batch |
| `deep_research_digest_timeout` | float | `120.0` | Timeout per digest (seconds) |
| `deep_research_digest_max_concurrent` | int | `3` | Max parallel digests |
| `deep_research_digest_include_evidence` | bool | `true` | Include direct quotes |
| `deep_research_digest_evidence_max_chars` | int | `400` | Max chars per snippet |
| `deep_research_digest_max_evidence_snippets` | int | `5` | Max snippets per digest |
| `deep_research_digest_fetch_pdfs` | bool | `false` | Fetch and extract PDFs |
| `deep_research_digest_provider` | str? | `null` | Provider override |
| `deep_research_digest_providers` | list[str] | `[]` | Fallback providers |

### Content Archive

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_research_archive_content` | bool | `false` | Archive content before processing |
| `deep_research_archive_retention_days` | int | `30` | Retention (days) |

### Synthesis & Evaluation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_research_synthesis_provider` | str? | `null` | Synthesis provider |
| `deep_research_synthesis_providers` | list[str] | `[]` | Fallback providers |
| `deep_research_evaluation_provider` | str? | `null` | Evaluation provider |
| `deep_research_evaluation_model` | str? | `null` | Evaluation model |
| `deep_research_evaluation_timeout` | float | `360.0` | Evaluation timeout (seconds) |
| `deep_research_reflection_provider` | str? | `null` | Reflection provider |
| `deep_research_reflection_model` | str? | `null` | Reflection model |
| `deep_research_reflection_timeout` | float | `60.0` | Reflection timeout (seconds) |

### Per-Phase Timeouts

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_research_planning_timeout` | float | `360.0` | Query decomposition |
| `deep_research_synthesis_timeout` | float | `600.0` | Report generation |

### Per-Phase Providers

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_research_planning_provider` | str? | `null` | Planning provider |
| `deep_research_planning_providers` | list[str] | `[]` | Planning fallback providers |

### Retry Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deep_research_max_retries` | int | `2` | Retries per provider |
| `deep_research_retry_delay` | float | `5.0` | Seconds between retries |

---

## Timeout Presets

Configure in `[research.timeouts]` with a `preset` key. Individual timeout overrides still win.

| Preset | Multiplier | Use Case |
|--------|-----------|----------|
| `fast` | 0.5x | Fast local models, low-latency APIs |
| `default` | 1.0x | Baseline (no change) |
| `relaxed` | 1.5x | CLI providers with variable latency |
| `patient` | 3.0x | Slow/overloaded providers |

### Timeouts Affected by Preset

| Timeout Field | Default (seconds) |
|---------------|-------------------|
| `default_timeout` | 360.0 |
| `deep_research_timeout` | 2400.0 |
| `deep_research_planning_timeout` | 360.0 |
| `deep_research_synthesis_timeout` | 600.0 |
| `deep_research_reflection_timeout` | 60.0 |
| `deep_research_evaluation_timeout` | 360.0 |
| `deep_research_supervision_wall_clock_timeout` | 1800.0 |
| `deep_research_summarization_timeout` | 60.0 |
| `deep_research_digest_timeout` | 120.0 |
| `summarization_timeout` | 60.0 |

---

## Fallback Provider Chains

Define named provider lists and assign them to phases.

```toml
[research.fallback_chains]
strong = ["[cli]gemini:pro", "[cli]codex:gpt-5.2-codex", "[cli]claude:opus"]
fast   = ["[cli]gemini:flash", "[cli]claude:sonnet"]

[research.phase_fallbacks]
planning  = "fast"       # Use the "fast" chain for planning
synthesis = "strong"     # Use the "strong" chain for synthesis
```

Explicit per-phase lists (`deep_research_planning_providers = [...]`) take priority over chain assignments. Chain values are expanded through `[providers]` aliases.

**Valid phase names:** `planning`, `synthesis`

---

## Model Tiers

Configure all 11 deep-research model roles via 3 named tiers.

### Tier Names

| Tier | Intended Use |
|------|-------------|
| `frontier` | Highest-quality reasoning |
| `standard` | Good balance of quality and cost |
| `efficient` | High-volume, low-complexity tasks |

### Default Role-to-Tier Mapping

| Role | Default Tier | Description |
|------|-------------|-------------|
| `research` | frontier | Main reasoning (analysis, planning) |
| `report` | frontier | Final synthesis / report generation |
| `evaluation` | frontier | LLM-as-judge quality assessment |
| `supervision` | frontier | Coverage gap assessment |
| `reflection` | standard | Think-tool pauses |
| `topic_reflection` | standard | Per-topic ReAct reflection |
| `clarification` | standard | Query disambiguation |
| `brief` | standard | Query enrichment before planning |
| `delegation` | standard | Supervisor delegation to units |
| `summarization` | efficient | Fetch-time content compression |
| `compression` | efficient | Per-topic aggregate compression |

### Resolution Precedence

When resolving a model for a role:

1. **Role-specific field** — `deep_research_{role}_provider` / `deep_research_{role}_model`
2. **Role resolution chain** — e.g., `delegation` → `supervision` → `reflection`
3. **Tier-based lookup** — when `[research.model_tiers]` is configured
4. **Cost-tier default** — `summarization`/`compression` → `gemini-2.5-flash`
5. **Global default** — `default_provider`

### Configuration Examples

```toml
# Simple form:
[research.model_tiers]
frontier  = "[cli]gemini:pro"
standard  = "[cli]gemini:flash"
efficient = "[cli]gemini:flash"

# Reassign a role to a different tier:
[research.model_tiers.role_assignments]
summarization = "standard"   # Promote from efficient → standard

# Table form with explicit model override:
[research.model_tiers.frontier]
provider = "[cli]gemini:pro"
model = "gemini-2.5-pro"
```

---

## Section: [research] — Search Providers

### Tavily

Env: `TAVILY_API_KEY`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tavily_api_key` | str? | `null` | API key (prefer env var) |
| `tavily_search_depth` | str | `"basic"` | `"basic"` \| `"advanced"` \| `"fast"` \| `"ultra_fast"` |
| `tavily_topic` | str | `"general"` | `"general"` \| `"news"` |
| `tavily_news_days` | int? | `null` | Days limit for news (1–365, topic=news only) |
| `tavily_include_images` | bool | `false` | Include image results |
| `tavily_country` | str? | `null` | ISO 3166-1 alpha-2 code |
| `tavily_chunks_per_source` | int | `3` | Chunks per source (1–5) |
| `tavily_auto_parameters` | bool | `false` | Auto-configure based on query |
| `tavily_extract_depth` | str | `"basic"` | `"basic"` \| `"advanced"` |
| `tavily_extract_include_images` | bool | `false` | Images in extractions |

### Perplexity

Env: `PERPLEXITY_API_KEY`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `perplexity_api_key` | str? | `null` | API key (prefer env var) |
| `perplexity_search_context_size` | str | `"medium"` | `"low"` \| `"medium"` \| `"high"` |
| `perplexity_max_tokens` | int | `50000` | Max response tokens |
| `perplexity_max_tokens_per_page` | int | `2048` | Max tokens per page |
| `perplexity_recency_filter` | str? | `null` | `"day"` \| `"week"` \| `"month"` \| `"year"` |
| `perplexity_country` | str? | `null` | ISO 3166-1 alpha-2 code |

### Semantic Scholar

Env: `SEMANTIC_SCHOLAR_API_KEY`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `semantic_scholar_api_key` | str? | `null` | API key (prefer env var) |
| `semantic_scholar_publication_types` | list[str]? | `null` | Filter by type |
| `semantic_scholar_sort_by` | str? | `null` | `"citationCount"` \| `"publicationDate"` \| `"paperId"` |
| `semantic_scholar_sort_order` | str | `"desc"` | `"asc"` \| `"desc"` |
| `semantic_scholar_use_extended_fields` | bool | `true` | Include TLDR and metadata |

Valid publication types: `Review`, `JournalArticle`, `Conference`, `CaseReport`, `ClinicalTrial`, `Dataset`, `Editorial`, `LettersAndComments`, `MetaAnalysis`, `News`, `Study`, `Book`, `BookSection`.

### Google

Env: `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `google_api_key` | str? | `null` | Google API key |
| `google_cse_id` | str? | `null` | Custom Search Engine ID |

### Per-Provider Rate Limits

```toml
[research.per_provider_rate_limits]
tavily = 60              # requests per minute
perplexity = 60
google = 100
semantic_scholar = 20
```

---

## Section: [research] — Token Management

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `token_management_enabled` | bool | `true` | Master switch |
| `token_safety_margin` | float | `0.15` | Buffer fraction (0.0–1.0) |
| `runtime_overhead` | int | `60000` | Tokens reserved for runtime context |

### Model Context Overrides

```toml
[research.model_context_overrides."claude:opus"]
context_window = 180000
max_output_tokens = 16000
budgeting_mode = "input_only"    # "input_only" | "combined"
output_reserved = 8000           # Only for "combined" mode
```

### Summarization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `summarization_provider` | str? | `null` | Primary provider |
| `summarization_providers` | list[str] | `[]` | Fallback providers |
| `summarization_timeout` | float | `60.0` | Timeout per request (seconds) |
| `summarization_cache_enabled` | bool | `true` | Cache results |

### Content Dropping & Archive

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `allow_content_dropping` | bool | `false` | Drop low-priority content |
| `content_archive_enabled` | bool | `false` | Archive to disk |
| `content_archive_ttl_hours` | int | `168` | Archive retention (hours, 7 days) |
| `research_archive_dir` | str? | `null` | Archive directory |

---

## Section: [observability]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `enabled` | bool | `false` | `FOUNDRY_MCP_OBSERVABILITY_ENABLED` | Master switch |
| `otel_enabled` | bool | `false` | `FOUNDRY_MCP_OTEL_ENABLED` | OpenTelemetry tracing |
| `otel_endpoint` | str | `"localhost:4317"` | `FOUNDRY_MCP_OTEL_ENDPOINT` | OTLP gRPC endpoint |
| `otel_service_name` | str | `"foundry-mcp"` | `FOUNDRY_MCP_OTEL_SERVICE_NAME` | Service name |
| `otel_sample_rate` | float | `1.0` | `FOUNDRY_MCP_OTEL_SAMPLE_RATE` | Sampling rate (0.0–1.0) |
| `prometheus_enabled` | bool | `false` | `FOUNDRY_MCP_PROMETHEUS_ENABLED` | Prometheus metrics |
| `prometheus_port` | int | `0` | `FOUNDRY_MCP_PROMETHEUS_PORT` | HTTP port (0 = disabled) |
| `prometheus_host` | str | `"0.0.0.0"` | `FOUNDRY_MCP_PROMETHEUS_HOST` | Bind address |
| `prometheus_namespace` | str | `"foundry_mcp"` | `FOUNDRY_MCP_PROMETHEUS_NAMESPACE` | Metric prefix |

Requires optional dependencies: `pip install foundry-mcp[tracing]`, `foundry-mcp[metrics]`, or `foundry-mcp[observability]`.

---

## Section: [health]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `enabled` | bool | `true` | `FOUNDRY_MCP_HEALTH_ENABLED` | Enable health probes |
| `liveness_timeout` | float | `1.0` | `FOUNDRY_MCP_HEALTH_LIVENESS_TIMEOUT` | Liveness check timeout (s) |
| `readiness_timeout` | float | `5.0` | `FOUNDRY_MCP_HEALTH_READINESS_TIMEOUT` | Readiness check timeout (s) |
| `health_timeout` | float | `10.0` | `FOUNDRY_MCP_HEALTH_TIMEOUT` | Full health check timeout (s) |
| `disk_space_threshold_mb` | int | `100` | `FOUNDRY_MCP_DISK_SPACE_THRESHOLD_MB` | Critical threshold (MB) |
| `disk_space_warning_mb` | int | `500` | `FOUNDRY_MCP_DISK_SPACE_WARNING_MB` | Warning threshold (MB) |

---

## Section: [error_collection]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `enabled` | bool | `true` | `FOUNDRY_MCP_ERROR_COLLECTION_ENABLED` | Enable error collection |
| `storage_path` | str | `""` | `FOUNDRY_MCP_ERROR_STORAGE_PATH` | Path (→ ~/.foundry-mcp/errors) |
| `retention_days` | int | `30` | `FOUNDRY_MCP_ERROR_RETENTION_DAYS` | Retention (days) |
| `max_errors` | int | `10000` | `FOUNDRY_MCP_ERROR_MAX_ERRORS` | Max error records |
| `include_stack_traces` | bool | `true` | `FOUNDRY_MCP_ERROR_INCLUDE_STACK_TRACES` | Include stack traces |
| `redact_inputs` | bool | `true` | `FOUNDRY_MCP_ERROR_REDACT_INPUTS` | Redact sensitive data |

---

## Section: [metrics_persistence]

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `enabled` | bool | `false` | `FOUNDRY_MCP_METRICS_PERSISTENCE_ENABLED` | Enable persistence |
| `storage_path` | str | `""` | `FOUNDRY_MCP_METRICS_STORAGE_PATH` | Path (→ ~/.foundry-mcp/metrics) |
| `retention_days` | int | `7` | `FOUNDRY_MCP_METRICS_RETENTION_DAYS` | Retention (days) |
| `max_records` | int | `100000` | `FOUNDRY_MCP_METRICS_MAX_RECORDS` | Max metric records |
| `bucket_interval_seconds` | int | `60` | `FOUNDRY_MCP_METRICS_BUCKET_INTERVAL` | Aggregation interval |
| `flush_interval_seconds` | int | `30` | `FOUNDRY_MCP_METRICS_FLUSH_INTERVAL` | Flush to disk interval |
| `persist_metrics` | list[str] | `["tool_invocations_total", ...]` | `FOUNDRY_MCP_METRICS_PERSIST_METRICS` | Metrics to persist (empty = all) |

---

## Section: [test]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_runner` | str | `"pytest"` | Default test runner |
| `runners` | dict | `{}` | Custom runner definitions |

### Custom Runner Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `command` | list[str] | `[]` | Command to execute |
| `run_args` | list[str] | `[]` | Additional arguments |
| `discover_args` | list[str] | `[]` | Discovery arguments |
| `pattern` | str | `"*"` | File pattern |
| `timeout` | int | `300` | Timeout (seconds) |

Built-in runners: `pytest`, `go`, `npm`, `jest`, `make`.

---

## Environment Variables

### Infrastructure

| Variable | Maps To | Type |
|----------|---------|------|
| `FOUNDRY_MCP_CONFIG_FILE` | Config file path override | str |
| `FOUNDRY_MCP_SPECS_DIR` | `workspace.specs_dir` | str |
| `FOUNDRY_MCP_RESEARCH_DIR` | `workspace.research_dir` | str |
| `FOUNDRY_MCP_LOG_LEVEL` | `logging.level` | str |
| `FOUNDRY_MCP_DISABLED_TOOLS` | `tools.disabled_tools` | comma-separated |
| `FOUNDRY_MCP_API_KEYS` | Auth API keys | comma-separated |
| `FOUNDRY_MCP_REQUIRE_AUTH` | Require authentication | bool |

### Git

| Variable | Maps To |
|----------|---------|
| `FOUNDRY_MCP_GIT_ENABLED` | `git.enabled` |
| `FOUNDRY_MCP_GIT_AUTO_COMMIT` | `git.auto_commit` |
| `FOUNDRY_MCP_GIT_AUTO_PUSH` | `git.auto_push` |
| `FOUNDRY_MCP_GIT_AUTO_PR` | `git.auto_pr` |
| `FOUNDRY_MCP_GIT_COMMIT_CADENCE` | `git.commit_cadence` |
| `FOUNDRY_MCP_GIT_SHOW_PREVIEW` | `git.show_before_commit` |

### Observability

| Variable | Maps To |
|----------|---------|
| `FOUNDRY_MCP_OBSERVABILITY_ENABLED` | `observability.enabled` |
| `FOUNDRY_MCP_OTEL_ENABLED` | `observability.otel_enabled` |
| `FOUNDRY_MCP_OTEL_ENDPOINT` | `observability.otel_endpoint` |
| `FOUNDRY_MCP_OTEL_SERVICE_NAME` | `observability.otel_service_name` |
| `FOUNDRY_MCP_OTEL_SAMPLE_RATE` | `observability.otel_sample_rate` |
| `FOUNDRY_MCP_PROMETHEUS_ENABLED` | `observability.prometheus_enabled` |
| `FOUNDRY_MCP_PROMETHEUS_PORT` | `observability.prometheus_port` |
| `FOUNDRY_MCP_PROMETHEUS_HOST` | `observability.prometheus_host` |
| `FOUNDRY_MCP_PROMETHEUS_NAMESPACE` | `observability.prometheus_namespace` |

### Health

| Variable | Maps To |
|----------|---------|
| `FOUNDRY_MCP_HEALTH_ENABLED` | `health.enabled` |
| `FOUNDRY_MCP_HEALTH_LIVENESS_TIMEOUT` | `health.liveness_timeout` |
| `FOUNDRY_MCP_HEALTH_READINESS_TIMEOUT` | `health.readiness_timeout` |
| `FOUNDRY_MCP_HEALTH_TIMEOUT` | `health.health_timeout` |
| `FOUNDRY_MCP_DISK_SPACE_THRESHOLD_MB` | `health.disk_space_threshold_mb` |
| `FOUNDRY_MCP_DISK_SPACE_WARNING_MB` | `health.disk_space_warning_mb` |

### Error Collection

| Variable | Maps To |
|----------|---------|
| `FOUNDRY_MCP_ERROR_COLLECTION_ENABLED` | `error_collection.enabled` |
| `FOUNDRY_MCP_ERROR_STORAGE_PATH` | `error_collection.storage_path` |
| `FOUNDRY_MCP_ERROR_RETENTION_DAYS` | `error_collection.retention_days` |
| `FOUNDRY_MCP_ERROR_MAX_ERRORS` | `error_collection.max_errors` |
| `FOUNDRY_MCP_ERROR_INCLUDE_STACK_TRACES` | `error_collection.include_stack_traces` |
| `FOUNDRY_MCP_ERROR_REDACT_INPUTS` | `error_collection.redact_inputs` |

### Metrics

| Variable | Maps To |
|----------|---------|
| `FOUNDRY_MCP_METRICS_PERSISTENCE_ENABLED` | `metrics_persistence.enabled` |
| `FOUNDRY_MCP_METRICS_STORAGE_PATH` | `metrics_persistence.storage_path` |
| `FOUNDRY_MCP_METRICS_RETENTION_DAYS` | `metrics_persistence.retention_days` |
| `FOUNDRY_MCP_METRICS_MAX_RECORDS` | `metrics_persistence.max_records` |
| `FOUNDRY_MCP_METRICS_BUCKET_INTERVAL` | `metrics_persistence.bucket_interval_seconds` |
| `FOUNDRY_MCP_METRICS_FLUSH_INTERVAL` | `metrics_persistence.flush_interval_seconds` |
| `FOUNDRY_MCP_METRICS_PERSIST_METRICS` | `metrics_persistence.persist_metrics` |

### Autonomy

| Variable | Maps To |
|----------|---------|
| `FOUNDRY_MCP_AUTONOMY_POSTURE` | `autonomy_posture.profile` |
| `FOUNDRY_MCP_ROLE` | `autonomy_security.role` |
| `FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_LOCK_BYPASS` | `autonomy_security.allow_lock_bypass` |
| `FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_GATE_WAIVER` | `autonomy_security.allow_gate_waiver` |
| `FOUNDRY_MCP_AUTONOMY_SECURITY_ENFORCE_REQUIRED_PHASE_GATES` | `autonomy_security.enforce_required_phase_gates` |
| `FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_MAX_CONSECUTIVE_DENIALS` | `autonomy_security.rate_limit_max_consecutive_denials` |
| `FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_DENIAL_WINDOW_SECONDS` | `autonomy_security.rate_limit_denial_window_seconds` |
| `FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_RETRY_AFTER_SECONDS` | `autonomy_security.rate_limit_retry_after_seconds` |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_GATE_POLICY` | `autonomy_session_defaults.gate_policy` |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_STOP_ON_PHASE_COMPLETION` | `autonomy_session_defaults.stop_on_phase_completion` |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_AUTO_RETRY_FIDELITY_GATE` | `autonomy_session_defaults.auto_retry_fidelity_gate` |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_MAX_TASKS_PER_SESSION` | `autonomy_session_defaults.max_tasks_per_session` |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_MAX_CONSECUTIVE_ERRORS` | `autonomy_session_defaults.max_consecutive_errors` |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_MAX_FIDELITY_REVIEW_CYCLES_PER_PHASE` | `autonomy_session_defaults.max_fidelity_review_cycles_per_phase` |

### Consultation

| Variable | Maps To |
|----------|---------|
| `FOUNDRY_MCP_CONSULTATION_PRIORITY` | `consultation.priority` |
| `FOUNDRY_MCP_CONSULTATION_TIMEOUT` | `consultation.default_timeout` |
| `FOUNDRY_MCP_CONSULTATION_MAX_RETRIES` | `consultation.max_retries` |
| `FOUNDRY_MCP_CONSULTATION_RETRY_DELAY` | `consultation.retry_delay` |
| `FOUNDRY_MCP_CONSULTATION_FALLBACK_ENABLED` | `consultation.fallback_enabled` |
| `FOUNDRY_MCP_CONSULTATION_CACHE_TTL` | `consultation.cache_ttl` |

### Search Provider API Keys

| Variable | Maps To |
|----------|---------|
| `TAVILY_API_KEY` | `research.tavily_api_key` |
| `PERPLEXITY_API_KEY` | `research.perplexity_api_key` |
| `GOOGLE_API_KEY` | `research.google_api_key` |
| `GOOGLE_CSE_ID` | `research.google_cse_id` |
| `SEMANTIC_SCHOLAR_API_KEY` | `research.semantic_scholar_api_key` |

### Feature Flags

| Variable | Description |
|----------|-------------|
| `FOUNDRY_MCP_FEATURE_FLAG_<NAME>` | Per-flag override (e.g., `FOUNDRY_MCP_FEATURE_FLAG_AUTONOMY_SESSIONS=1`) |
| `FOUNDRY_MCP_FEATURE_FLAGS` | Bulk flag list (comma-separated names to enable) |

---

## Deprecated Fields

These fields are no longer recognized. Remove them from your config and use the replacements.

| Deprecated Field | Replacement | Reason |
|------------------|-------------|--------|
| `deep_research_enable_reflection` | *(always-on)* | Now always enabled in supervision loop |
| `deep_research_enable_contradiction_detection` | *(always-on)* | Folded into supervision |
| `deep_research_enable_topic_agents` | *(always-on)* | Per-topic ReAct agents always enabled |
| `deep_research_analysis_timeout` | `deep_research_planning_timeout` or `deep_research_synthesis_timeout` | Phases restructured |
| `deep_research_refinement_timeout` | `deep_research_planning_timeout` or `deep_research_synthesis_timeout` | Phases restructured |
| `deep_research_analysis_provider` | `deep_research_planning_provider` or `deep_research_synthesis_provider` | Phases restructured |
| `deep_research_refinement_provider` | `deep_research_planning_provider` or `deep_research_synthesis_provider` | Phases restructured |
| `deep_research_analysis_providers` | `deep_research_planning_providers` or `deep_research_synthesis_providers` | Phases restructured |
| `deep_research_refinement_providers` | `deep_research_planning_providers` or `deep_research_synthesis_providers` | Phases restructured |
| `tavily_extract_in_deep_research` | `deep_research_enable_extract` | URL extraction is now per-topic |
| `tavily_extract_max_urls` | `deep_research_extract_max_per_iteration` | URL extraction is now per-topic |
| `deep_research_digest_policy` | *(auto-handled)* | Digest policy removed; digestion automatic |
| `notes_dir` | *(phantom)* | Never existed in config dataclass |

### Orphaned Sections

| Section | Status | Notes |
|---------|--------|-------|
| `[implement]` | Never parsed | Use `[autonomy_posture]` and `[autonomy_session_defaults]` |
| `[workflow]` | Never parsed | Use `[autonomy_session_defaults]` |
| `[feature_flags]` | Never parsed | Features controlled by `enabled` flags on each config section |
