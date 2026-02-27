# Configuration

foundry-mcp supports configuration via TOML and environment variables. The CLI
and MCP server share the same settings.

## Configuration order

Configuration is loaded in layers, with each layer overriding the previous:

1. **Defaults** - Built-in default values
2. **User config** - `~/.foundry-mcp.toml` (optional, user-wide settings)
3. **Project config** - `./foundry-mcp.toml` (optional, project-specific)
4. **Environment variables** - Runtime overrides (highest priority)

### Config file locations

| Location | Purpose | Example use cases |
|----------|---------|-------------------|
| `~/.foundry-mcp.toml` | User defaults | Preferred CLI providers, logging preferences |
| `./foundry-mcp.toml` | Project settings | specs_dir, workspace roots, project-specific tool config |

### Legacy compatibility

For backwards compatibility, if `./foundry-mcp.toml` doesn't exist, the system
will fall back to `./.foundry-mcp.toml` (dot-prefixed) in the project directory.

## Minimal TOML example

```toml
[workspace]
specs_dir = "./specs"

[logging]
level = "INFO"

[consultation]
priority = ["[cli]claude:opus", "[cli]gemini:pro"]
```

## Common environment variables

| Variable | Purpose |
| --- | --- |
| `FOUNDRY_MCP_SPECS_DIR` | Override specs directory |
| `FOUNDRY_MCP_WORKSPACE_ROOTS` | Restrict allowed workspace roots |
| `FOUNDRY_MCP_LOG_LEVEL` | Set log level (INFO, DEBUG, etc.) |
| `FOUNDRY_MCP_API_KEYS` | Require API keys for tool access |
| `FOUNDRY_MCP_REQUIRE_AUTH` | Enforce auth on all tools |
| `FOUNDRY_MCP_FEATURE_FLAGS` | Bulk feature flag overrides (`flag` or `flag=true/false`, comma-separated) |
| `FOUNDRY_MCP_FEATURE_FLAG_<NAME>` | Per-flag override (highest precedence), e.g. `FOUNDRY_MCP_FEATURE_FLAG_AUTONOMY_SESSIONS=true` |
| `FOUNDRY_MCP_AUTONOMY_POSTURE` | Posture profile (`unattended`, `supervised`, `debug`) |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_GATE_POLICY` | Default `session-start` gate policy (`strict`, `lenient`, `manual`) |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_STOP_ON_PHASE_COMPLETION` | Default `stop_on_phase_completion` for session starts |
| `FOUNDRY_MCP_AUTONOMY_DEFAULT_AUTO_RETRY_FIDELITY_GATE` | Default `auto_retry_fidelity_gate` for session starts |

## Feature Flags

Feature flags can be set in TOML and overridden via environment variables.
Precedence (highest to lowest):

1. `FOUNDRY_MCP_FEATURE_FLAG_<NAME>` per-flag env vars
2. `FOUNDRY_MCP_FEATURE_FLAGS` bulk env var
3. `[feature_flags]` in TOML

Example TOML:

```toml
[feature_flags]
autonomy_sessions = true
autonomy_fidelity_gates = false
```

Example env overrides:

```bash
export FOUNDRY_MCP_FEATURE_FLAGS="autonomy_sessions=true,autonomy_fidelity_gates=true"
export FOUNDRY_MCP_FEATURE_FLAG_AUTONOMY_FIDELITY_GATES=false
```

Startup validation emits warnings for inconsistent combinations (for example,
`autonomy_fidelity_gates=true` while `autonomy_sessions=false`).

## Autonomy Posture Profiles

Posture profiles provide one-knob defaults for autonomy security and session behavior.
Profiles are a fixed enum:

- `unattended`: headless-safe defaults (`autonomy_runner`, escape hatches closed)
- `supervised`: human-in-the-loop defaults (guardrails on, escape hatches available with reason codes)
- `debug`: maximum flexibility for manual troubleshooting (no unattended loops)

Example TOML:

```toml
[autonomy_posture]
profile = "unattended"
```

Profile defaults can be overridden through direct configuration:

```toml
[autonomy_security]
role = "maintainer"
allow_lock_bypass = true
allow_gate_waiver = true

[autonomy_session_defaults]
gate_policy = "manual"
stop_on_phase_completion = false
```

When a profile and direct overrides conflict, startup warnings call out unsafe combinations (for example, unattended posture with maintainer role or bypass enabled).

## LLM providers

foundry-mcp uses CLI-based providers (claude, gemini, codex, cursor-agent, opencode) configured via the `[consultation]` section. See the [LLM Configuration Guide](guides/llm-configuration.md) for full details.

Common consultation environment variables:

| Variable | Purpose |
| --- | --- |
| `FOUNDRY_MCP_CONSULTATION_PRIORITY` | Comma-separated provider priority list |
| `FOUNDRY_MCP_CONSULTATION_TIMEOUT` | Default timeout in seconds |
| `FOUNDRY_MCP_CONSULTATION_MAX_RETRIES` | Max retry attempts |
| `FOUNDRY_MCP_CONSULTATION_FALLBACK_ENABLED` | Enable provider fallback |

## Research Configuration

The `[research]` section controls deep research workflows including search provider
settings. For full configuration options, see `samples/foundry-mcp.toml`.

### Tavily Search Provider

Tavily is a web search provider optimized for AI applications. Configure via
environment variable or TOML:

```bash
export TAVILY_API_KEY="tvly-..."
```

#### Search Parameters

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tavily_search_depth` | string | `"basic"` | Search mode: `"basic"`, `"advanced"` (2x credits), `"fast"`, `"ultra_fast"` |
| `tavily_topic` | string | `"general"` | Search topic: `"general"`, `"news"` |
| `tavily_news_days` | int | `null` | Days limit for news (1-365, only when `topic="news"`) |
| `tavily_include_images` | bool | `false` | Include image results |
| `tavily_country` | string | `null` | ISO 3166-1 alpha-2 code to boost results (e.g., `"US"`) |
| `tavily_chunks_per_source` | int | `3` | Chunks per source for advanced search (1-5) |
| `tavily_auto_parameters` | bool | `false` | Let Tavily auto-configure based on query |

#### Extract Parameters

Tavily Extract enables URL content extraction for deeper analysis.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `tavily_extract_depth` | string | `"basic"` | Extract mode: `"basic"`, `"advanced"` |
| `tavily_extract_include_images` | bool | `false` | Include images in extraction |
| `tavily_extract_in_deep_research` | bool | `false` | Enable extract as follow-up step |
| `tavily_extract_max_urls` | int | `5` | Max URLs to extract per deep research run |

#### Research Mode Smart Defaults

When using deep research, parameters are adjusted based on `deep_research_mode`:

| Mode | Search Depth | Source Prioritization |
|------|-------------|----------------------|
| `"general"` | `basic` | No preference |
| `"academic"` | `advanced` | Journals, publishers, preprints |
| `"technical"` | `advanced` | Official docs, arxiv, Stack Overflow |

#### Example Configuration

```toml
[research]
# Search provider credentials (prefer env vars in production)
# tavily_api_key = "tvly-..."

# Search parameters
tavily_search_depth = "basic"      # "basic", "advanced" (2x credits), "fast", "ultra_fast"
tavily_topic = "general"           # "general", "news"
tavily_news_days = 7               # only when topic = "news"
tavily_include_images = false
tavily_country = "US"              # boost results from country
tavily_chunks_per_source = 3       # 1-5, for advanced search
tavily_auto_parameters = false     # let Tavily auto-configure

# Extract parameters
tavily_extract_depth = "basic"           # "basic", "advanced"
tavily_extract_include_images = false
tavily_extract_in_deep_research = false  # enable extract follow-up
tavily_extract_max_urls = 5              # max URLs per deep research run

# Deep research mode affects Tavily parameter selection
deep_research_mode = "technical"   # "general", "academic", "technical"
```

#### Credit Cost Awareness

- `search_depth="basic"` - Standard credit cost
- `search_depth="advanced"` - 2x credit cost (use for deeper analysis)
- `search_depth="fast"` / `"ultra_fast"` - Reduced latency, standard cost

### Deep Research Resilience

The following settings control timeout, cancellation, and resilience behavior for deep research workflows.

#### Timeout Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `deep_research_timeout` | float | `600.0` | Overall workflow timeout in seconds (10 minutes) |
| `deep_research_planning_timeout` | float | `360.0` | Planning phase timeout |
| `deep_research_analysis_timeout` | float | `360.0` | Analysis phase timeout |
| `deep_research_synthesis_timeout` | float | `600.0` | Synthesis phase timeout (longer for complex reports) |
| `deep_research_refinement_timeout` | float | `360.0` | Refinement phase timeout |

**Timeout Precedence:**
1. Explicit `task_timeout` parameter in API call (highest priority)
2. `deep_research_timeout` from configuration
3. Hardcoded fallback of 600 seconds

#### Status Response Metadata

When polling `deep-research-status`, the response includes resilience metadata:

| Field | Type | Description |
|-------|------|-------------|
| `last_heartbeat_at` | string (ISO 8601) | Last activity timestamp, updated before provider calls |
| `is_timed_out` | bool | True if task exceeded timeout |
| `is_stale` | bool | True if no activity for 5+ minutes |
| `effective_timeout` | float | The actual timeout applied to the task |

#### Example Configuration

```toml
[research]
# Workflow-level timeout (overall limit)
deep_research_timeout = 600.0  # 10 minutes

# Per-phase timeouts (optional overrides)
deep_research_planning_timeout = 360.0
deep_research_analysis_timeout = 360.0
deep_research_synthesis_timeout = 600.0
deep_research_refinement_timeout = 360.0

# Retry behavior
deep_research_max_retries = 2
deep_research_retry_delay = 5.0
```

### Document Digest

The document digest feature compresses source content into structured summaries
with evidence snippets for citation traceability. This reduces context usage
while preserving key information.

#### Digest Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `deep_research_digest_policy` | string | `"auto"` | When to digest: `"off"`, `"auto"`, `"always"`, `"proactive"` |
| `deep_research_digest_min_chars` | int | `500` | Minimum content length to trigger digest |
| `deep_research_digest_max_sources` | int | `50` | Maximum sources to digest per iteration |
| `deep_research_digest_timeout` | float | `120.0` | Timeout per digest operation (seconds) |
| `deep_research_digest_max_concurrent` | int | `3` | Maximum concurrent digest operations |
| `deep_research_digest_include_evidence` | bool | `true` | Include evidence snippets in output |
| `deep_research_digest_evidence_max_chars` | int | `400` | Maximum characters per evidence snippet (1-500) |
| `deep_research_digest_max_evidence_snippets` | int | `5` | Maximum evidence snippets per digest (1-10) |
| `deep_research_digest_fetch_pdfs` | bool | `false` | Fetch and extract PDF content |
| `deep_research_digest_provider` | string | `null` | Primary LLM provider for digest (uses analysis provider if not set) |
| `deep_research_digest_providers` | list | `[]` | Fallback providers for digest (tried in order if primary fails) |

**Digest Policies:**

| Policy | Behavior |
|--------|----------|
| `off` | Never digest - all sources pass through unchanged |
| `auto` | Digest HIGH/MEDIUM quality sources above size threshold |
| `always` | Always digest sources with content |
| `proactive` | Digest every source immediately at retrieval time in the gathering phase, ensuring uniform content for downstream analysis |

#### Content Archival

When archival is enabled, canonical (normalized) text is stored before compression,
enabling verification of evidence snippet locators against original content.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `deep_research_archive_content` | bool | `false` | Archive canonical text before digest |
| `deep_research_archive_retention_days` | int | `30` | Days to retain archived content (0 = keep indefinitely) |

**Archive Location:**

Archives are stored at:
```
~/.foundry-mcp/research_archives/{source_id}/{content_hash}.txt
```

- Files are created with owner-only permissions (0600)
- Directories are created with owner-only permissions (0700)
- Old archives are automatically cleaned based on `retention_days`

#### Example Configuration

```toml
[research]
# Digest settings
deep_research_digest_policy = "auto"
deep_research_digest_min_chars = 500
deep_research_digest_fetch_pdfs = true

# Archival (for citation verification)
deep_research_archive_content = true
deep_research_archive_retention_days = 60
```

### Query Clarification

Before research begins, an optional clarification phase analyzes the query
for completeness and infers scope, timeframe, or domain constraints. Since
the workflow runs non-interactively, the LLM infers reasonable constraints
rather than blocking on user input. Inferred constraints are fed into the
planning phase for more focused sub-query generation.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `deep_research_allow_clarification` | bool | `true` | Enable the clarification phase before planning |
| `deep_research_clarification_provider` | string | `null` | LLM provider for clarification (uses `default_provider` if not set) |

When enabled, the clarification step sends the query to a fast model that returns
structured JSON indicating whether the query needs clarification and what
constraints can be inferred. Constraints are fed into the planning phase for
more focused sub-query generation.

```toml
[research]
deep_research_allow_clarification = true
# Use a fast/cheap model for the single clarification call
deep_research_clarification_provider = "[cli]gemini:flash"
```

### LLM-Driven Supervisor Reflection

After each phase completes, an optional LLM reflection step evaluates phase
results and decides whether quality is sufficient to proceed. This coexists
with (does not replace) the existing heuristic quality gates.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `deep_research_enable_reflection` | bool | `true` | Enable LLM reflection at phase boundaries |
| `deep_research_reflection_provider` | string | `null` | LLM provider for reflection (uses `default_provider` if not set) |
| `deep_research_reflection_timeout` | float | `60.0` | Timeout per reflection call in seconds |

The reflection LLM returns a structured assessment: quality rating, whether to
proceed, suggested adjustments, and rationale. Reflection decisions are recorded
in the audit trail.

```toml
[research]
deep_research_enable_reflection = true
deep_research_reflection_provider = "[cli]gemini:flash"
deep_research_reflection_timeout = 60.0
```

### Parallel Topic Researcher Agents

When enabled, each sub-query in the gathering phase runs its own mini ReAct
loop instead of a single flat search. Each topic researcher independently
searches, reflects on coverage gaps, refines its query, and searches again
until it has sufficient information or reaches the iteration limit. Topic
researchers run in parallel, bounded by `deep_research_max_concurrent`.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `deep_research_enable_topic_agents` | bool | `true` | Enable per-topic ReAct loops in the gathering phase |
| `deep_research_topic_max_tool_calls` | int | `10` | Maximum tool calls (search + extract) per topic |
| `deep_research_topic_reflection_provider` | string | `null` | LLM provider for per-topic reflection (uses `default_provider` if not set) |
| `deep_research_enable_extract` | bool | `true` | Allow researchers to extract full content from promising URLs |
| `deep_research_extract_max_per_iteration` | int | `2` | Maximum URLs to extract per ReAct iteration |

> **Note:** `deep_research_topic_max_searches` is accepted as a backward-compatible alias for `deep_research_topic_max_tool_calls`.

Per-topic summaries are compiled and fed into the analysis phase, providing
more coherent per-topic coverage than flat parallel search. Sources are
deduplicated across topic researchers. When extraction is enabled, the
reflection step can recommend URLs for full content extraction, which is
particularly valuable for technical documentation, academic papers, and
comparison tables.

```toml
[research]
deep_research_enable_topic_agents = true
deep_research_topic_max_tool_calls = 10
deep_research_enable_extract = true
deep_research_extract_max_per_iteration = 2
deep_research_topic_reflection_provider = "[cli]gemini:flash"
```

### Contradiction Detection

After the analysis phase extracts findings, a contradiction detection step
identifies conflicting claims between sources. Detected contradictions are
stored in research state and surfaced in the synthesis prompt so the final
report can address them explicitly.

Contradiction detection runs automatically when findings are extracted —
there is no separate config toggle. Each contradiction includes the conflicting
finding IDs, a description, a resolution suggestion, the preferred source,
and a severity rating (major/minor).

### Citation Tracking

Deep research assigns each source a stable citation number (1-indexed) when
it enters the research state. The synthesis phase presents findings with
`[N]` citation markers, and the LLM is instructed to use inline citations
in the report. A `## Sources` section is auto-generated from the state
(not from LLM output) and appended to the report.

Post-processing verifies citation consistency:
- All referenced `[N]` numbers exist in sources
- Dangling citations (referencing non-existent sources) are removed
- Unreferenced sources are logged as warnings

Citation numbers survive refinement iterations — re-synthesis preserves
the same numbering scheme.

### Audit Verbosity

The `audit_verbosity` setting controls the size of JSONL audit payloads written during deep research workflows. This can reduce CPU spent on large audit writes while maintaining schema stability for downstream analysis tools.

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `audit_verbosity` | string | `"full"` | Audit payload detail level: `"full"` or `"minimal"` |

**Verbosity Modes:**

| Mode | Behavior |
|------|----------|
| `full` | Original audit events unchanged - complete audit trail with all prompts, responses, and content |
| `minimal` | Large text fields set to `null` while preserving schema shape and metrics |

**Fields Nulled in Minimal Mode:**

Top-level fields:
- `system_prompt` - LLM system prompt text
- `user_prompt` - LLM user prompt text
- `raw_response` - Raw LLM response text
- `report` - Generated report content
- `error` - Error message text
- `traceback` - Error traceback text

Nested fields:
- `findings[*].content` - Finding content text
- `gaps[*].description` - Gap description text

**Fields Preserved (Always Included):**

Metrics and identifiers are always preserved for analysis compatibility:
- `provider_id` - LLM provider identifier
- `model_used` - Model identifier
- `tokens_used` - Token consumption
- `duration_ms` - Operation duration
- `sources_added` - Source count
- `report_length` - Report character count
- `parse_success` - Parse status boolean

**Schema Stability Guarantee:**

Both modes produce the same JSON schema shape - all keys are present in both modes. The difference is that `minimal` mode sets large text values to `null`. This ensures downstream analysis tools can process audit files from either mode without schema changes.

**Example Configuration:**

```toml
[research]
# Reduce audit file size while maintaining schema
audit_verbosity = "minimal"
```

## Secret Management

The autonomy subsystem uses a server secret for HMAC-based integrity protection
of gate evidence. The secret ensures that gate verdicts cannot be tampered with
between the review step and the evidence-submission step.

### How the secret works

- **Auto-generated** on first use: a 32-byte random key is created automatically
- **Stored** at `$FOUNDRY_DATA_DIR/.server_secret` (default: `~/.foundry-mcp/.server_secret`) with mode `0600` (owner read/write only)
- **Cached** in memory after first load for performance
- **Override** via `FOUNDRY_MCP_GATE_SECRET` environment variable (for deterministic testing)

### Secret rotation / recovery procedure

If the server secret is compromised or needs rotation:

1. **Stop the server** — ensure no autonomous sessions are actively running
2. **Delete the secret file**:
   ```bash
   rm ~/.foundry-mcp/.server_secret
   # or if using custom data dir:
   rm $FOUNDRY_DATA_DIR/.server_secret
   ```
3. **Restart the server** — a new secret is auto-generated on first use

### Impact of rotation

- All **in-flight gate evidence** becomes invalid because HMAC checksums were
  computed with the old secret. The orchestrator will reject stale evidence and
  re-request gate steps.
- Active sessions with **pending manual gate acknowledgments** will need the
  gate step re-issued.
- Sessions should be **rebased** (`session-rebase`) after rotation to
  re-establish a clean state, or ended and restarted.
- Completed sessions and historical audit data are **not affected** — they are
  already persisted and do not require the secret for reads.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `FOUNDRY_DATA_DIR` | Override the data directory (default: `~/.foundry-mcp`) |
| `FOUNDRY_MCP_GATE_SECRET` | Provide a deterministic secret (for testing only — do not use in production) |

### Security notes

- Never commit the secret file to version control
- The secret file permissions are set to `0600` (owner-only) automatically
- In production, use filesystem-level access controls to protect `$FOUNDRY_DATA_DIR`
- Gate evidence is short-lived by design, limiting the window of exposure if
  a secret is compromised

## Specs directory resolution

If you do not set `FOUNDRY_MCP_SPECS_DIR`, the CLI and server will attempt to
auto-detect a `specs/` directory in the workspace.
