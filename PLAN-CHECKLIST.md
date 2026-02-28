# foundry-mcp.toml Full Overhaul — Checklist

## Phase 2: Comprehensive Reference TOML (`samples/foundry-mcp-reference.toml`)

### Header & aliases
- [x] Write file header with purpose, config priority, pointers
- [x] `[providers]` section with example aliases (commented)

### Infrastructure sections
- [x] `[workspace]` — specs_dir, research_dir (no phantom notes_dir)
- [x] `[logging]` — level, structured
- [x] `[tools]` — disabled_tools with full tool list
- [x] ~~`[feature_flags]`~~ — orphaned, never parsed; omitted intentionally

### Git & workflow
- [x] `[git]` — all 6 fields
- [x] ~~`[workflow]`~~ — orphaned, never parsed; omitted intentionally
- [x] NO `[implement]` section (orphaned, never parsed)

### Autonomy
- [x] `[autonomy_posture]` — profile
- [x] `[autonomy_security]` — role, lock bypass, gate waiver, rate limits
- [x] `[autonomy_session_defaults]` — all 6 fields

### Consultation
- [x] `[consultation]` section with boundary explanation comment
- [x] priority, timeout, retries, fallback, cache_ttl
- [x] `[consultation.overrides]` example
- [x] `[consultation.workflows.*]` examples

### Research: core
- [x] enabled, default_provider, consensus_providers, default_timeout
- [x] ttl_hours, max_messages_per_thread, thinkdeep_max_depth, ideate_perspectives
- [x] Sub-table syntax guide comment
- [x] Timeout presets (`[research.timeouts]`) with multiplier table
- [x] Fallback chains (`[research.fallback_chains]`, `[research.phase_fallbacks]`) with alias example

### Research: deep research — core workflow
- [x] max_iterations, max_sub_queries, max_sources (default=5), follow_links
- [x] timeout (default=2400.0), max_concurrent, mode, audit_artifacts
- [x] providers list (search providers)

### Research: deep research — supervision & delegation
- [x] enable_supervision, max_supervision_rounds
- [x] supervision_min_sources_per_query, coverage_confidence_threshold
- [x] supervision_wall_clock_timeout, supervision_provider/model
- [x] max_concurrent_research_units, delegation_provider/model_name

### Research: deep research — query phases
- [x] allow_clarification, clarification_provider/model
- [x] brief_provider/model
- [x] enable_planning_critique

### Research: deep research — gathering
- [x] topic_max_tool_calls (note backward-compat alias topic_max_searches)
- [x] enable_extract, extract_max_per_iteration
- [x] enable_content_dedup, content_dedup_threshold

### Research: deep research — content processing
- [x] Summarization: provider/model, timeout, max_content_length, min_content_length
- [x] Compression: inline_compression, provider/model, max_content_length
- [x] Digestion: all 11 digest_* fields
- [x] Archive: archive_content, archive_retention_days

### Research: deep research — synthesis & evaluation
- [x] evaluation_provider/model/timeout
- [x] reflection_timeout, reflection_provider/model

### Research: deep research — per-phase config
- [x] Per-phase timeouts: planning_timeout, synthesis_timeout (only 2 phases now)
- [x] Per-phase providers: planning_provider, synthesis_provider
- [x] Per-phase fallback lists: planning_providers, synthesis_providers
- [x] Retry settings: max_retries, retry_delay
- [x] NO deprecated analysis/refinement phase fields

### Research: model tiers
- [x] Precedence chain documentation (5 levels)
- [x] Default role → tier mapping (11 roles)
- [x] `[research.model_tiers]` with tier examples
- [x] `[research.model_tiers.role_assignments]` example
- [x] Per-role overrides section with cross-reference to tiers

### Research: search providers
- [x] `[research.tavily]` / flat tavily_* fields (no deprecated extract fields)
- [x] `[research.perplexity]` / flat perplexity_* fields
- [x] `[research.semantic_scholar]` / flat semantic_scholar_* fields
- [x] Google config: google_api_key, google_cse_id
- [x] API key env var references
- [x] `[research.per_provider_rate_limits]`

### Research: token management & archive
- [x] token_management_enabled, token_safety_margin, runtime_overhead
- [x] model_context_overrides example
- [x] Summarization: provider, providers, timeout, cache_enabled
- [x] Content dropping: allow_content_dropping
- [x] Content archive: enabled, ttl_hours, research_archive_dir

### Research: operational tuning
- [x] search_rate_limit, max_concurrent_searches
- [x] deep_research_stale_task_seconds
- [x] status_persistence_throttle_seconds
- [x] audit_verbosity

### Observability & monitoring
- [x] `[observability]` — all OTel + Prometheus fields
- [x] `[health]` — all probe + threshold fields
- [x] `[error_collection]` — all fields
- [x] `[metrics_persistence]` — all fields

### Test runner
- [x] `[test]` — default_runner, custom runner example

### Accuracy verification
- [x] No deprecated field names appear (even commented)
- [x] All defaults match code
- [x] `python -c "import tomllib; tomllib.load(open('samples/foundry-mcp-reference.toml', 'rb'))"` ✓

---

## Phase 1: Minimal Quick-Start Sample (`samples/foundry-mcp.toml`)

- [x] Rewrite as ~51-line minimal config
- [x] Include pointer to reference files in header
- [x] `[workspace]` — specs_dir only
- [x] `[logging]` — level only
- [x] `[tools]` — disabled_tools
- [x] `[git]` — enabled, commit_cadence
- [x] `[providers]` — commented alias example
- [x] `[research]` — enabled, default_provider, consensus_providers, mode
- [x] `[consultation]` — priority only
- [x] `[test]` — default_runner
- [x] NO deprecated fields, NO phantom fields, NO orphaned sections
- [x] `python -c "import tomllib; tomllib.load(open('samples/foundry-mcp.toml', 'rb'))"` ✓

---

## Phase 3: Markdown Config Reference (`dev_docs/guides/config-reference.md`)

### Structure
- [x] Config file locations & priority
- [x] Provider spec format explanation
- [x] Sub-table syntax (flat vs nested, priority rules)

### Section tables
- [x] `[providers]` table
- [x] `[workspace]` table
- [x] `[logging]` table
- [x] `[tools]` table
- [x] ~~`[feature_flags]` table~~ — orphaned, documented in Deprecated section
- [x] `[git]` table
- [x] ~~`[workflow]` table~~ — orphaned, documented in Deprecated section
- [x] `[autonomy_posture]` table
- [x] `[autonomy_security]` table
- [x] `[autonomy_session_defaults]` table
- [x] `[consultation]` table
- [x] `[research]` core table
- [x] `[research]` deep research table (grouped by feature)
- [x] `[research]` search provider tables
- [x] `[observability]` table
- [x] `[health]` table
- [x] `[error_collection]` table
- [x] `[metrics_persistence]` table
- [x] `[test]` table

### Special topics
- [x] Environment variable reference (all 65+ vars)
- [x] Timeout presets table
- [x] Fallback chains explanation
- [x] Model tier precedence chain
- [x] Deprecated fields migration table (all 12)

---

## Phase 4: Validation Tests (`tests/unit/test_sample_toml.py`)

- [x] Test: sample TOML parses as valid TOML
- [x] Test: reference TOML parses as valid TOML
- [x] Test: uncommented sample [research] fields are known to ResearchConfig
- [x] Test: no deprecated field names appear uncommented in either file
- [x] Test: ResearchConfig.from_toml_dict() accepts sample [research] without warnings
- [x] `pytest tests/unit/test_sample_toml.py -v` — 10/10 passed ✓
- [x] `pytest tests/ -x` — 1960 passed, 1 pre-existing failure (unrelated), 0 new regressions ✓
