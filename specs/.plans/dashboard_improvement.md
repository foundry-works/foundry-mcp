# Dashboard Overhaul Plan

## Context
The Streamlit dashboard at http://127.0.0.1:8501/ shows stale data, feels sluggish, and the UI hierarchy/presentation is not ideal. The goal is to refresh data reliably, improve responsiveness, and redesign layout/aggregation for better utility.

## Goals
- Keep data fresh and visibly up-to-date (10s cadence), without blocking the UI.
- Improve perceived and actual performance with caching and smarter queries.
- Redesign layout and information hierarchy for clarity and faster insight.
- Preserve current Streamlit-based architecture (no new UI layer).

## Non-Goals
- Replace Streamlit with a new frontend (React/Next/etc.).
- Rebuild the metrics/error persistence backends from scratch.
- Add multi-user auth or remote access.

## Current Issues (Observed)
- Data staleness: dashboard process does not consistently see new metrics/errors.
- Sluggish UX: blocking refresh loop and heavy re-reads on every interaction.
- "Vanilla" UI: weak hierarchy, inconsistent aggregation choices, limited context.

## Root Causes (Confirmed)
- In-memory indexes in `FileMetricsStore` / `FileErrorStore` are built once at singleton creation and never refreshed when JSONL files update (stale read across processes).
- Auto-refresh uses `time.sleep(...)` in the main Streamlit script, blocking interactivity.
- Data layer lacks consistent caching: only 2 of ~12 data functions use `st.cache_data`.

## Technical Architecture

### Key Files
- `src/foundry_mcp/core/error_store.py` - FileErrorStore singleton, JSONL + index.json
- `src/foundry_mcp/core/metrics_store.py` - FileMetricsStore singleton, JSONL + index.json
- `src/foundry_mcp/dashboard/data/stores.py` - Adapter layer (stores -> pandas)
- `src/foundry_mcp/dashboard/app.py` - Main Streamlit app
- `src/foundry_mcp/dashboard/views/` - Page views (overview, errors, metrics, tool_usage, providers)

### Data Flow
```
MCP Server -> FileErrorStore/FileMetricsStore (JSONL files)
                     |
                     v
Dashboard Process -> stores.py adapter -> pandas DataFrames -> views
```

**Problem:** Dashboard holds its own singleton with stale in-memory index.

---

## Planned Workstreams

### Phase 0: Baseline Measurement (NEW)
Before implementing changes, capture current performance metrics:

- [ ] Measure full page load time for each view
- [ ] Measure query latency for `get_errors()`, `get_metrics_timeseries()`
- [ ] Count queries-per-page-load to identify redundant fetches
- [ ] Document current behavior for "time from write to dashboard display"

**Success Metrics:**
| Metric | Current (Measure) | Target |
|--------|-------------------|--------|
| Page load time | TBD | <2s |
| Data freshness lag | TBD | <10s |
| Query latency (100 records) | TBD | <100ms |

---

### Phase 1: Data Freshness + Caching

#### 1.1 Index Refresh Strategy
Add mtime-based staleness detection inside store classes (Option B):

```python
# In FileErrorStore / FileMetricsStore
class FileErrorStore(ErrorStore):
    def __init__(self, ...):
        self._last_data_mtime: float = 0.0
        self._load_index()

    def _check_and_refresh(self) -> None:
        """Check if data file changed; reload index if stale."""
        current_mtime = self._data_file.stat().st_mtime if self._data_file.exists() else 0.0
        if current_mtime > self._last_data_mtime:
            self._load_index()
            self._last_data_mtime = current_mtime

    def query(self, ...) -> list[ErrorRecord]:
        self._check_and_refresh()  # <-- Add this call
        # ... existing query logic
```

**Files to modify:**
- `src/foundry_mcp/core/error_store.py` - Add `_check_and_refresh()` to `FileErrorStore`
- `src/foundry_mcp/core/metrics_store.py` - Add `_check_and_refresh()` to `FileMetricsStore`

#### 1.2 Streamlit Caching Strategy
Add `st.cache_data` with TTL to adapter functions. Cache keys automatically include all function parameters.

| Function | TTL | Rationale |
|----------|-----|-----------|
| `get_errors()` | 10s | Frequent updates expected |
| `get_error_patterns()` | 30s | Aggregated, less volatile |
| `get_error_stats()` | 10s | KPI display |
| `get_metrics_list()` | 60s | Catalog rarely changes |
| `get_metrics_timeseries()` | 10s | Time-series data |
| `get_metrics_summary()` | 10s | KPI display |
| `get_tool_action_breakdown()` | 10s | Main chart data |
| `get_top_tool_actions()` | 10s | Top-N list |
| `get_overview_summary()` | 10s | KPI display |

**Cache Invalidation:**
- "Refresh Now" button calls `st.cache_data.clear()` then `st.rerun()`
- This clears ALL caches globally (acceptable for manual refresh)

#### 1.3 Timestamp Display
- Add "Last updated: X seconds ago" to page headers
- Store refresh timestamp in `st.session_state['last_refresh']`

**Files to modify:**
- `src/foundry_mcp/dashboard/data/stores.py` - Add `@st.cache_data` decorators
- `src/foundry_mcp/dashboard/app.py` - Track and display refresh timestamp

---

### Phase 2: Auto-Refresh (Revised)

#### 2.1 Implementation Approach
Use `st.rerun()` with `st.session_state` for state-preserving refresh (avoid browser refresh hack):

```python
# In app.py
import time

if 'last_auto_refresh' not in st.session_state:
    st.session_state.last_auto_refresh = time.time()

# Sidebar toggle
auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Interval (seconds)", 5, 60, 10)

# Check if refresh needed (non-blocking)
if auto_refresh:
    elapsed = time.time() - st.session_state.last_auto_refresh
    if elapsed >= refresh_interval:
        st.session_state.last_auto_refresh = time.time()
        st.cache_data.clear()
        st.rerun()
```

**Benefits over browser refresh:**
- Preserves filter selections, scroll position, expanded state
- No page flicker
- Integrates with Streamlit state management

#### 2.2 Status Indicator
- Show "Auto-refresh: ON (next in Xs)" in sidebar
- Use `st.empty()` placeholder for dynamic countdown

**Files to modify:**
- `src/foundry_mcp/dashboard/app.py` - Implement auto-refresh logic

---

### Phase 3: UX/UI Restructure

#### 3.1 Global Controls
Add persistent filter bar at top of each page:
- Time range selector: 1h, 6h, 24h (default), 7d, 30d
- Tool filter: multi-select dropdown
- Status filter: success/error toggle

Store selections in `st.session_state` to persist across page navigation.

#### 3.2 KPI Hierarchy (Defined)
**Overview page - Top row (priority order):**
1. **Success Rate** - % of tool invocations without errors (24h)
2. **Total Invocations** - Count (24h)
3. **Error Count** - Count (24h) with delta vs previous period
4. **Unique Patterns** - Distinct error fingerprints

**Second row:**
- Trend sparklines for invocations and errors (hourly buckets)

**Third row:**
- Recent Errors panel (last 10)
- Top Tool Actions panel (top 5)

#### 3.3 Errors Page
- Summary bar: total errors, unique patterns, most common error_code
- Filterable table with clickable rows (expand to show full details)
- Remove manual ID copy/paste - add "View Details" button per row
- Quick filters: tool dropdown, error_code dropdown, time range

#### 3.4 Tool Usage Page
- Per-tool cards showing: invocation count, success rate, avg latency
- Action drilldown: click tool to see action breakdown
- Status distribution pie chart

#### 3.5 Metrics Page
- Default view: curated list (tool_invocation_*, error_*)
- "Show all metrics" expander for full list
- Simplified time-series display with summary stats

#### 3.6 Providers Page
- Consistent with global time range filter
- Provider availability status grid

**Files to modify:**
- `src/foundry_mcp/dashboard/views/overview.py`
- `src/foundry_mcp/dashboard/views/errors.py`
- `src/foundry_mcp/dashboard/views/tool_usage.py`
- `src/foundry_mcp/dashboard/views/metrics.py`
- `src/foundry_mcp/dashboard/views/providers.py`
- `src/foundry_mcp/dashboard/components/filters.py` - Global filter component

---

### Phase 4: Robustness & Diagnostics

#### 4.1 Data Status Panel
Add collapsible "System Status" section to sidebar with:

| Item | Source |
|------|--------|
| Errors collection | config `enabled` flag |
| Metrics collection | config `enabled` flag |
| Errors storage path | `~/.foundry-mcp/errors/` |
| Metrics storage path | `~/.foundry-mcp/metrics/` |
| Errors file size | `errors.jsonl` stat |
| Metrics file size | `metrics.jsonl` stat |
| Error count | index record count |
| Metric count | index record count |
| Last data write | max mtime of JSONL files |
| Index status | valid / needs rebuild |

#### 4.2 Logging
- Add structured logging for refresh events: `{"event": "dashboard_refresh", "page": "...", "duration_ms": ...}`
- Log cache hit/miss rates for debugging

#### 4.3 Launcher Improvements
- Capture stderr from Streamlit subprocess
- Surface startup errors in CLI output
- Add health check endpoint query on launch

**Files to modify:**
- `src/foundry_mcp/dashboard/app.py` - Data status panel
- `src/foundry_mcp/dashboard/launcher.py` - Error capture improvements

---

### Phase 5: Testing & Rollout (NEW)

#### 5.1 Integration Test
Add test to verify end-to-end freshness:

```python
def test_dashboard_sees_new_data_within_10s():
    # 1. Write a metric via store
    store = get_metrics_store()
    store.append(MetricDataPoint(...))

    # 2. Query via dashboard adapter
    time.sleep(1)  # Allow mtime to update
    df = get_metrics_timeseries(...)

    # 3. Assert new data visible
    assert len(df) > 0
    assert df['timestamp'].max() > (datetime.now() - timedelta(seconds=10))
```

#### 5.2 Feature Flag (Optional)
If risk is high, add config flag to toggle new refresh behavior:

```toml
[dashboard]
use_mtime_refresh = true  # false to revert to old behavior
```

#### 5.3 Rollback Procedure
1. Set `use_mtime_refresh = false` in config
2. Restart dashboard
3. Old singleton behavior restored

**Files to modify:**
- `tests/integration/test_dashboard_freshness.py` (new)
- `src/foundry_mcp/core/config.py` - Optional feature flag

---

## Deliverables
- Updated store classes with mtime-based refresh
- Cached dashboard data layer
- Auto-refresh without UI blocking
- Refreshed UI layout with defined KPI hierarchy
- Data Status panel for diagnostics
- Integration test for freshness guarantee
- Documentation updates (dashboard usage notes)

## Acceptance Criteria
- [ ] Data updates appear within 10 seconds of new writes (with refresh enabled)
- [ ] No UI blocking during refresh; user interactions remain responsive
- [ ] Clear visual hierarchy: Success Rate first, then volume, then details
- [ ] Errors and tool usage are actionable without manual ID copy/paste
- [ ] Dashboard continues to run with optional deps missing (graceful messages)
- [ ] Integration test passes for freshness guarantee
- [ ] Page load time < 2s (measured)

## Risks / Mitigations
| Risk | Mitigation |
|------|------------|
| Frequent mtime checks degrade performance | Check only at query time, not continuously; mtime is a fast syscall |
| Cache causes stale data display | Short TTL (10s) + manual "Refresh Now" button |
| Auto-refresh causes excessive reruns | User toggle + configurable interval (default 10s) |
| Index reload during active queries | File locking already in place; reload is atomic |

## Decisions Made (Previously Open Questions)
| Question | Decision | Rationale |
|----------|----------|-----------|
| Default time range | 24h | Provides broader context; 1h available via filter |
| Top KPI | Success Rate | Most actionable operational metric |
| Advanced metrics | Behind expander | Reduce cognitive load for common use cases |

## Best-Practices Consulted
- docs/mcp_best_practices/README.md#L24-L77
- docs/mcp_best_practices/05-observability-telemetry.md
- docs/mcp_best_practices/12-timeout-resilience.md
- docs/mcp_best_practices/10-testing-fixtures.md (for integration test)
