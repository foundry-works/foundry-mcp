# Plan: foundry-mcp.toml Full Overhaul

## Problem

The sample config (`samples/foundry-mcp.toml`) has accumulated significant drift:

- **12 deprecated fields** still shown as active
- **2 wrong default values** misleading users
- **1 orphaned section** (`[implement]`) never parsed
- **1 phantom field** (`notes_dir`) that doesn't exist
- **23+ fields** undocumented (supervision, delegation, compression, evaluation, etc.)
- **65 environment variable overrides** undocumented
- **Model tier system** poorly explained
- **960 lines** — too long for a "sample" config; mixes quick-start guidance with exhaustive reference material

## Solution: Three-File Split

| File | Purpose | Audience |
|------|---------|----------|
| `samples/foundry-mcp.toml` | **Quick-start.** Minimal, opinionated, good defaults. ~150 lines. | New users, copy-paste setup |
| `samples/foundry-mcp-reference.toml` | **Full reference.** Every field commented out with annotations. Copy sections as needed. | Power users tuning knobs |
| `dev_docs/guides/config-reference.md` | **Searchable docs.** Tables of every field, default, env var, and description. | Lookup & search |

---

## Phase 1: Minimal Quick-Start Sample

**Impact: High | Effort: Medium | File: `samples/foundry-mcp.toml`**

Rewrite the sample as a short, opinionated config that works out of the box. Only show fields a user would actually want to change. No deprecated fields, no phantom sections, no commented-out walls of text.

### Structure

```toml
# foundry-mcp.toml — Quick-start configuration
#
# Place in your project root. For all available options, see:
#   samples/foundry-mcp-reference.toml  (copy-pasteable TOML)
#   dev_docs/guides/config-reference.md (searchable reference)
#
# Config priority: env vars > project config > user config > defaults
# User config locations: ~/.foundry-mcp.toml or ~/.config/foundry-mcp/config.toml

[workspace]
specs_dir = "./specs"

[logging]
level = "INFO"

[tools]
disabled_tools = ["error", "health"]

[git]
enabled = false
commit_cadence = "phase"

# --- Provider Aliases (optional) ---
# Define short names for provider specs, used everywhere below.
# [providers]
# pro  = "[cli]gemini:pro"
# fast = "[cli]gemini:flash"
# opus = "[cli]claude:opus"

[research]
enabled = true
default_provider = "[cli]gemini:pro"
consensus_providers = [
    "[cli]gemini:pro",
    "[cli]codex:gpt-5.2-codex",
    "[cli]claude:opus",
]
deep_research_mode = "technical"

[consultation]
priority = [
    "[cli]codex:gpt-5.2-codex",
    "[cli]gemini:pro",
    "[cli]claude:opus",
]

[test]
default_runner = "pytest"
```

### What gets removed from the sample

- All deprecated fields (12 fields)
- Wrong default values (fixed in reference)
- `[implement]` section (orphaned)
- `notes_dir` (phantom)
- All 70+ deep_research_* commented-out fields
- All search provider config details (tavily, perplexity, semantic_scholar)
- All observability/health/error_collection/metrics_persistence details
- All autonomy config details
- All token management details
- Digest, compression, summarization, evaluation details
- Model tier configuration
- Fallback chains / timeout presets

### What stays

- Workspace basics
- Logging level
- Tool disabling
- Git basics
- Provider aliases (commented example)
- Research: enabled, default_provider, consensus_providers, mode
- Consultation: priority list
- Test runner
- Pointer to reference files

---

## Phase 2: Comprehensive Reference TOML

**Impact: High | Effort: Large | File: `samples/foundry-mcp-reference.toml`**

Every field in every config section, commented out, annotated with defaults and types. Organized by section with clear headers. Users copy what they need.

### Organization

Sections in order:

1. **Header** — file purpose, config priority, pointers
2. **`[providers]`** — alias definitions
3. **`[workspace]`** — specs_dir, research_dir
4. **`[logging]`** — level, structured
5. **`[tools]`** — disabled_tools
6. **`[feature_flags]`** — autonomy_sessions, autonomy_fidelity_gates
7. **`[git]`** — all git workflow fields
8. **`[workflow]`** — mode, auto_validate, batch_size, context_threshold
9. **`[autonomy_posture]`** — profile
10. **`[autonomy_security]`** — role, lock bypass, gate waiver, rate limits
11. **`[autonomy_session_defaults]`** — gate_policy, stop_on_phase, max_tasks, etc.
12. **`[consultation]`** — priority, timeout, retries, per-workflow overrides
13. **`[research]`** — core fields, then sub-sections:
    - Core (enabled, provider, timeout, ttl)
    - Sub-table syntax guide
    - Timeout presets (`[research.timeouts]`)
    - Fallback chains (`[research.fallback_chains]`, `[research.phase_fallbacks]`)
    - Deep research: core workflow settings
    - Deep research: supervision & delegation
    - Deep research: query phases (clarification, brief, planning critique)
    - Deep research: gathering (topic agents, extraction, dedup)
    - Deep research: content processing (summarization, compression, digestion)
    - Deep research: synthesis & evaluation
    - Deep research: per-phase timeouts & providers
    - Model tiers (`[research.model_tiers]`)
    - Per-role model overrides (advanced)
    - Search provider config (`[research.tavily]`, `[research.perplexity]`, `[research.semantic_scholar]`)
    - Token management
    - Content archive
    - Rate limiting
    - Operational tuning
14. **`[observability]`** — OTel, Prometheus
15. **`[health]`** — probes, thresholds
16. **`[error_collection]`** — storage, retention
17. **`[metrics_persistence]`** — storage, retention, bucketing
18. **`[test]`** — runner config

### Accuracy rules

- Every field must exist in the corresponding config dataclass
- Every default value must match the code default
- No deprecated field names appear (even commented)
- Deprecated features are noted with "Removed:" comments pointing to replacements
- Env var override noted inline where applicable

---

## Phase 3: Markdown Config Reference

**Impact: Medium | Effort: Medium | File: `dev_docs/guides/config-reference.md`**

Searchable documentation with tables for every config section.

### Structure

```markdown
# Configuration Reference

## Config File Locations
## Environment Variables
## Section: [providers]
## Section: [workspace]
...
## Section: [research]
### Core Fields
### Deep Research Fields
### Tavily Fields
...
```

### Table format per section

| Field | Type | Default | Env Var | Description |
|-------|------|---------|---------|-------------|
| `enabled` | bool | `true` | — | Master switch for research tools |
| `default_provider` | str | `"gemini"` | — | Default LLM provider spec |
| ... | ... | ... | ... | ... |

### Content

- **Config priority** explanation with examples
- **Environment variable reference** — all 65+ vars with mappings
- **Provider spec format** — explanation of `[cli]provider:model` syntax
- **Sub-table syntax** — flat vs nested, priority rules
- **Timeout presets** — preset names and multipliers table
- **Fallback chains** — how chains and phase assignments work
- **Model tiers** — the 11-role, 4-level precedence chain explained
- **Deprecated fields** — table of all 12 deprecated fields with migration paths

---

## Phase 4: Validation Tests

**Impact: Low (safety net) | Effort: Small | File: `tests/unit/test_sample_toml.py`**

### Tests

1. **Sample parses**: `samples/foundry-mcp.toml` is valid TOML
2. **Reference parses**: `samples/foundry-mcp-reference.toml` is valid TOML
3. **Sample fields known**: Every uncommented key in `[research]` is a known ResearchConfig field
4. **Reference fields known**: Every commented field in `[research]` section is a known field
5. **No deprecated fields**: Neither file contains deprecated field names (uncommented)
6. **Sample loads**: `ResearchConfig.from_toml_dict()` accepts the sample's `[research]` section without warnings

---

## Implementation Order

1. **Phase 2** first — write the reference TOML (most work, establishes ground truth)
2. **Phase 1** second — strip the sample down to minimal (easy once reference exists)
3. **Phase 3** third — generate markdown from the reference TOML fields
4. **Phase 4** last — add validation tests

## Files Changed

| File | Action |
|------|--------|
| `samples/foundry-mcp.toml` | Rewrite (minimal quick-start) |
| `samples/foundry-mcp-reference.toml` | New (comprehensive reference) |
| `dev_docs/guides/config-reference.md` | New (searchable docs) |
| `tests/unit/test_sample_toml.py` | New (validation tests) |
