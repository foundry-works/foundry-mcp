# Deep Research CPU Optimization

## Mission

Reduce CPU usage in the deep research workflow without changing research outputs, preserving behavior and content fidelity.

## Objective

Optimize the deep research workflow to reduce unnecessary CPU cycles caused by:
1. Excessive status persistence on every poll
2. Token estimation recalculation after process restarts
3. Repeated tiktoken encoding lookups
4. Large audit payload serialization

All optimizations must preserve identical research outputs and maintain backward compatibility with existing APIs.

## Scope

### In Scope
- Throttling status persistence with configurable intervals
- Adding cross-session token count caching with content hashing
- Caching tiktoken encodings at module level
- Adding optional audit verbosity configuration
- Updating relevant configuration options
- Unit tests for new caching and throttling logic
- Fidelity reviews at each phase
- Benchmark instrumentation for measuring improvements

### Out of Scope
- Changes to model prompts, source selection, or result formatting
- Changes to concurrency limits, provider choices, or query decomposition
- Changes to the research algorithm or quality of results
- Breaking API changes

## Assumptions

- Content fidelity must remain identical for a given run
- Existing APIs and defaults are preserved unless new config flags are explicitly set
- tiktoken library availability is optional (graceful fallback exists)
- Process-scoped caches remain alongside new persistent caches
- "Byte-identical outputs" refers to: final research reports, findings, gaps, and source attributions (NOT internal audit artifacts which are opt-in)

## Benchmark Targets

- **Workload**: 3 deep research queries with 5 status polls each
- **Target**: Status persistence writes reduced to ≤20% of baseline (5x reduction)
- **Target**: Token estimation calls on resume should be 0 (full cache hit)
- **Measurement**: DEBUG log lines for persistence and estimation calls, compare before/after

## Output Equivalence Definition

**What counts as "outputs" that must remain identical:**
- Final research report text
- Findings and gaps content
- Source citations and attributions
- Status response structure and content (excluding internal timing metadata)

**What is NOT considered "outputs" (allowed to change):**
- Internal audit artifacts (when verbosity changes)
- Disk write frequency (optimization target)
- Internal cache metadata (stored in source.metadata but not exposed in API responses)

## Phases

### Phase 1: Throttle Status Persistence

**Purpose**: Reduce disk I/O by adding a minimum interval between status-save operations. Currently `_get_status` persists the full `DeepResearchState` on every poll, serializing large `source.content` payloads.

**Tasks**:
1. Add `status_persistence_throttle_seconds: int = 5` to `ResearchConfig` in `src/foundry_mcp/config.py`
   - Add validation: must be >= 0, where 0 means "always persist" (current behavior)
2. Add `_last_persisted_at: datetime | None` tracking field to `DeepResearchRunner` in `src/foundry_mcp/core/research/workflows/deep_research.py`
3. Modify `_get_status` to only persist when:
   - Phase or iteration changes (always persist)
   - Terminal state is reached - completed/failed (always persist)
   - Throttle interval has elapsed since last persist
   - Priority order: terminal state > phase change > throttle interval
4. Keep `status_check_count` and `last_status_check_at` updates in memory, persist at interval
5. Add explicit state flush on workflow completion (success or error paths) to ensure token cache is persisted
6. Add DEBUG logging for status persistence to enable benchmark measurement
7. Add unit tests for throttle behavior including edge cases:
   - `throttle_seconds=0` (always persist)
   - Terminal state during throttle interval
   - Phase change during throttle interval
8. Fidelity review: verify implementation matches spec requirements

**Files**:
- `src/foundry_mcp/config.py`
- `src/foundry_mcp/core/research/workflows/deep_research.py`
- `tests/unit/research/test_deep_research.py`

**Verification**:
- Status responses unchanged in content
- Disk writes reduced during frequent polling (measurable via log counts)
- State still correctly persisted on phase/iteration changes
- Final state always persisted on workflow completion
- Fidelity review passes

### Phase 2: Token Count Caching Across Sessions

**Purpose**: Avoid expensive token re-estimation after process restarts by persisting per-source token counts keyed by provider/model and content hash.

**Storage Design**:
- Token cache stored in `ResearchSource.metadata["_token_cache"]` (underscore prefix = internal)
- Schema: `{"v": 1, "counts": {"{content_hash_32}:{len}:{provider}:{model}": token_count}}`
- Key includes 32-char hash + content length for collision protection
- Version field enables future migrations
- Internal fields excluded from API response serialization via existing `_internal_fields` pattern

**Cache Bounds**:
- Max 50 cached counts per source (FIFO eviction - oldest removed when full)
- Content hash: first 32 chars of SHA-256 (128-bit collision resistance)
- Dict single-key operations are thread-safe, no complex locking needed

**Tasks**:
1. Add `_content_hash()` helper method to `ResearchSource` in `src/foundry_mcp/core/research/models.py`
   - Returns first 32 chars of SHA-256 of content (128-bit collision resistance)
2. Define token cache schema with version field in source metadata
   - Key format: `{hash_32}:{len}:{provider}:{model}`
3. Modify `ContextBudgetManager` in `src/foundry_mcp/core/research/context_budget.py` to:
   - Check for cached token count in `ResearchSource.metadata["_token_cache"]` before estimation
   - Store computed token counts back to metadata with FIFO bounds (max 50, oldest evicted)
4. Add `_internal_fields` exclusion to prevent `_token_cache` from appearing in API responses
5. Keep existing in-memory cache in `token_management.py` as-is (complementary layer)
6. Add DEBUG logging for token estimation (cache hit vs miss) for benchmark measurement
7. Add unit tests for cache hit/miss scenarios, FIFO eviction, and persistence across sessions
8. Add test that loads old state files without token cache (backward compatibility)
9. Fidelity review: verify implementation matches spec requirements

**Files**:
- `src/foundry_mcp/core/research/models.py`
- `src/foundry_mcp/core/research/context_budget.py`
- `src/foundry_mcp/core/research/workflows/deep_research.py`
- `tests/unit/research/test_context_budget.py`

**Verification**:
- Token estimates identical for unchanged content
- Reduced CPU on resumed runs (measurable via log counts)
- Cache invalidation works when content changes
- Old state files load without errors
- `_token_cache` not present in API responses
- Fidelity review passes

### Phase 3: Cache Tokenizer Encodings

**Purpose**: Reduce overhead from repeated `tiktoken.encoding_for_model` and `tiktoken.get_encoding` calls during heavy estimation loops.

**Tasks**:
1. Add module-level cache using `@functools.lru_cache(maxsize=32)` in `src/foundry_mcp/core/research/token_management.py`
   - Thread-safe by design (lru_cache is thread-safe)
   - Bounded to 32 encodings max
2. Create `_get_cached_encoding(model_name: str)` helper function decorated with lru_cache
3. Modify `_estimate_with_tiktoken` to use cached encodings
4. Preserve graceful fallback when tiktoken is unavailable
5. Add unit tests for encoding cache behavior
6. Fidelity review: verify implementation matches spec requirements

**Files**:
- `src/foundry_mcp/core/research/token_management.py`
- `tests/unit/research/test_token_management.py`

**Verification**:
- Token counts unchanged
- Lower CPU in heavy estimation loops (measurable)
- Cache reuses encoding objects across calls
- Fidelity review passes

### Phase 4: Optional Audit Payload Reduction

**Purpose**: Reduce CPU spent on large JSONL audit writes by adding configurable audit verbosity levels.

**Schema Stability Guarantee**:
- Audit JSONL schema remains stable across verbosity modes
- In `minimal` mode: fields set to `null` (NOT removed)
- This ensures downstream consumers don't break on missing fields

**Fields Nulled in Minimal Mode**:
- `data.system_prompt`
- `data.user_prompt`
- `data.raw_response`
- `data.report`
- `data.error`
- `data.traceback`
- `data.findings[*].content`
- `data.gaps[*].description`

**Fields Preserved (Metrics)**:
- `provider_id`, `model_used`, `tokens_used`, `duration_ms`
- `sources_added`, `report_length`, `parse_success`

**Tasks**:
1. Add `audit_verbosity: str = "full"` to `ResearchConfig` with validation:
   - Allowed values: `"full"`, `"minimal"`
   - Invalid values raise clear error message
2. Modify audit event writing in `DeepResearchRunner` to respect verbosity setting
3. Add helper method `_prepare_audit_payload(event: dict, verbosity: str)` that:
   - In `full` mode: returns event unchanged
   - In `minimal` mode: sets documented fields to `null` (preserves schema)
   - Handles nested fields (findings[*].content, gaps[*].description)
4. Update `docs/deep-research.md` to document:
   - New configuration flag
   - Schema stability guarantee
   - Exact list of fields nulled in minimal mode
   - Fields preserved (metrics)
5. Add unit tests for both verbosity modes asserting:
   - Schema shape identical (same keys present)
   - Content differs only in documented redacted fields
   - Nested field handling works correctly
6. Fidelity review: verify implementation matches spec requirements

**Files**:
- `src/foundry_mcp/config.py`
- `src/foundry_mcp/core/research/workflows/deep_research.py`
- `docs/deep-research.md`
- `tests/unit/research/test_deep_research.py`

**Verification**:
- Research outputs unchanged
- Audit artifacts smaller when `minimal` is enabled
- JSONL schema identical in both modes (fields present, values differ)
- Fidelity review passes

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Status loss on crash during throttle interval | Medium | Always persist on phase/iteration changes, terminal states, and workflow completion |
| Token cache key collisions | Low | Use 32-char SHA-256 + content length in key (128-bit + length disambiguation) |
| Stale token counts after model changes | Medium | Include model identifier in cache key, invalidate on mismatch |
| Audit schema changes breaking consumers | Medium | Keep fields present but null in minimal mode; document exact field list |
| tiktoken API changes | Low | Cache layer is thin; easy to update if API changes |
| Unbounded cache growth | Medium | FIFO bounds on token cache (50 per source), LRU on encodings (32 max) |
| Token cache not persisted due to throttling | Medium | Explicit flush on workflow completion guarantees persistence |

## Dependencies

- Phases are independent and can proceed in parallel
- Phase 1 config patterns can be reused in Phases 2 and 4 but not blocking

## Success Criteria

- [ ] Status persistence reduced to ≤20% of baseline (5x reduction)
- [ ] Token estimation calls on resume = 0 (full cache hit)
- [ ] All existing tests pass without modification
- [ ] New unit tests cover throttle, cache hit/miss, FIFO eviction, and verbosity modes
- [ ] Research outputs identical before and after changes (report, findings, gaps)
- [ ] No breaking changes to public APIs
- [ ] Old state files load without errors (backward compatibility)
- [ ] Audit JSONL schema stable across verbosity modes
- [ ] Documentation updated for new configuration options
- [ ] Fidelity review passes for each phase
