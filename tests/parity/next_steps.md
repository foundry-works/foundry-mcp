# Parity Testing Next Steps

## Overview

With 12 new fixtures created, the next phase is to write the actual parity test files that exercise SDD CLI commands vs foundry-mcp tools side-by-side.

---

## Phase 1: Authoring Operations Tests

### File: `test_authoring_ops.py`

**Fixtures to use:**
- `authoring_adapters` - isolated adapters for mutation tests
- `authoring_subtasks_dir` - for cascade delete tests
- `authoring_assumptions_dir` - for assumption list/filter tests

**Tests to implement:**

#### Task Add/Remove Operations
```python
class TestTaskAddParity:
    def test_add_task_to_phase_parity(self, authoring_adapters):
        """Add a new task to phase-1, verify both systems create identical task."""
        # Call: task-add with parent=phase-1, title="New task"
        # Assert: task_id returned, task appears in hierarchy

    def test_add_subtask_parity(self, authoring_adapters):
        """Add subtask to existing task."""
        # Call: task-add with parent=task-1-1, type=subtask

    def test_add_task_with_position_parity(self, authoring_adapters):
        """Add task at specific position in parent's children list."""
        # Call: task-add with position=0

    def test_add_task_dry_run_parity(self, authoring_adapters):
        """Dry run should return task_id without modifying spec."""


class TestTaskRemoveParity:
    def test_remove_leaf_task_parity(self, authoring_adapters):
        """Remove task with no children."""
        # Call: task-remove task-1-2

    def test_remove_task_cascade_parity(self, test_dir):
        """Remove task with subtasks using cascade=True."""
        # Setup: authoring_with_subtasks fixture
        # Call: task-remove task-1-1 cascade=True
        # Assert: task-1-1, subtask-1-1-1, subtask-1-1-2, deep-1-1-1-1 all removed

    def test_remove_task_no_cascade_error_parity(self, test_dir):
        """Remove task with children without cascade should error."""
        # Setup: authoring_with_subtasks fixture
        # Call: task-remove task-1-1 cascade=False
        # Assert: both systems return error

    def test_remove_nonexistent_task_parity(self, authoring_adapters):
        """Removing nonexistent task should error consistently."""
```

#### Assumption Operations
```python
class TestAssumptionsParity:
    def test_add_assumption_constraint_parity(self, authoring_adapters):
        """Add constraint-type assumption."""
        # Call: assumption-add type=constraint text="Must use Python 3.11+"

    def test_add_assumption_requirement_parity(self, authoring_adapters):
        """Add requirement-type assumption."""

    def test_list_all_assumptions_parity(self, test_dir):
        """List all assumptions from pre-populated spec."""
        # Setup: authoring_with_assumptions fixture (has 4 assumptions)
        # Call: assumption-list
        # Assert: 4 assumptions returned

    def test_list_assumptions_filter_type_parity(self, test_dir):
        """Filter assumptions by type."""
        # Call: assumption-list type=constraint
        # Assert: 2 constraints returned
```

#### Revision Operations
```python
class TestRevisionsParity:
    def test_add_revision_parity(self, authoring_adapters):
        """Add revision history entry."""
        # Call: revision-add version="1.1.0" changes="Added new feature" author="dev"

    def test_revision_appears_in_history_parity(self, authoring_adapters):
        """Verify revision appears in spec's revision_history."""
```

#### Estimate/Metadata Operations
```python
class TestEstimatesParity:
    def test_update_estimate_hours_parity(self, authoring_adapters):
        """Update task estimated hours."""
        # Call: task-update-estimate task-1-1 hours=4.5

    def test_update_estimate_complexity_parity(self, authoring_adapters):
        """Update task complexity."""
        # Call: task-update-estimate task-1-1 complexity=high

    def test_update_task_metadata_parity(self, authoring_adapters):
        """Update arbitrary task metadata."""
        # Call: task-update-metadata task-1-1 file_path="src/new.py"
```

---

## Phase 2: Analysis Operations Tests

### File: `test_analysis_ops.py`

**Fixtures to use:**
- `bottleneck_spec_dir` / `analysis_adapters` - bottleneck detection
- `circular_deps_dir` - cycle detection
- `patterns_spec_dir` - pattern finding

**Tests to implement:**

#### Dependency Analysis
```python
class TestAnalyzeDepsParity:
    def test_analyze_deps_basic_parity(self, analysis_adapters):
        """Basic dependency analysis."""
        # Call: spec-analyze-deps parity-test-bottleneck
        # Assert: dependency_count, has bottlenecks

    def test_bottleneck_detection_parity(self, analysis_adapters):
        """Detect task-1-1 as bottleneck (blocks 4 tasks)."""
        # Call: spec-analyze-deps bottleneck_threshold=3
        # Assert: task-1-1 in bottlenecks list

    def test_critical_path_parity(self, analysis_adapters):
        """Identify critical path: task-1-1 -> task-1-2 -> task-1-6."""
        # Assert: critical_path contains these tasks in order
```

#### Cycle Detection
```python
class TestDetectCyclesParity:
    def test_detect_cycles_with_cycle_parity(self, circular_deps_dir):
        """Detect circular dependency in spec."""
        # Setup: analysis_circular_deps fixture
        # Call: spec-detect-cycles parity-test-circular
        # Assert: has_cycles=True, cycle contains [task-1-1, task-1-2, task-1-3]

    def test_detect_cycles_no_cycle_parity(self, bottleneck_spec_dir):
        """Spec without cycles returns has_cycles=False."""
        # Call: spec-detect-cycles parity-test-bottleneck
        # Assert: has_cycles=False, cycles=[]

    def test_affected_tasks_in_cycle_parity(self, circular_deps_dir):
        """List all tasks involved in cycles."""
        # Assert: affected_tasks contains task-1-1, task-1-2, task-1-3
        # Assert: task-1-4 NOT in affected_tasks (independent)
```

#### Pattern Finding
```python
class TestFindPatternsParity:
    def test_find_python_files_parity(self, patterns_spec_dir):
        """Find tasks with .py file paths."""
        # Call: spec-find-patterns pattern="*.py"
        # Assert: matches include src/core/module.py, src/utils/helpers.py, tests/*.py

    def test_find_test_files_parity(self, patterns_spec_dir):
        """Find tasks with test file paths."""
        # Call: spec-find-patterns pattern="tests/*"
        # Assert: matches include tests/test_module.py, tests/test_helpers.py

    def test_find_related_files_parity(self, patterns_spec_dir):
        """Find files related to a source file."""
        # Call: spec-find-related-files file_path="src/core/module.py"
```

---

## Phase 3: Edge Case Tests

### File: `test_edge_cases.py`

**Fixtures to use:**
- `deep_nesting_dir` - hierarchy traversal
- `large_spec_dir` / `edge_adapters` - pagination
- `empty_spec_dir` - empty handling
- `all_blocked_dir` - blocker scenarios

**Tests to implement:**

#### Deep Nesting
```python
class TestDeepNestingParity:
    def test_task_list_deep_hierarchy_parity(self, deep_nesting_dir):
        """List all tasks in deeply nested spec."""
        # Call: task-list parity-test-deep
        # Assert: 8 tasks returned including deeper-1-1-1-1-1, deeper-1-1-1-1-2

    def test_progress_deep_hierarchy_parity(self, deep_nesting_dir):
        """Progress calculation with deep nesting."""
        # Call: task-progress parity-test-deep

    def test_remove_deep_cascade_parity(self, test_dir):
        """Cascade delete from deep hierarchy."""
        # Setup: edge_deep_nesting fixture
        # Call: task-remove task-1-1 cascade=True
        # Assert: All nested children removed (6 tasks total)

    def test_render_deep_hierarchy_parity(self, deep_nesting_dir):
        """Render spec with deep nesting."""
        # Call: spec-render parity-test-deep max_depth=0
        # Assert: All levels rendered

    def test_render_max_depth_parity(self, deep_nesting_dir):
        """Render with max_depth limit."""
        # Call: spec-render parity-test-deep max_depth=2
        # Assert: Only 2 levels rendered
```

#### Large Spec / Pagination
```python
class TestLargeSpecParity:
    def test_task_list_large_spec_parity(self, edge_adapters):
        """List all 100 tasks."""
        # Call: task-list parity-test-large
        # Assert: 100 tasks returned

    def test_task_query_pagination_parity(self, edge_adapters):
        """Paginate through tasks with limit."""
        # Call: task-query parity-test-large limit=25
        # Assert: 25 tasks, cursor returned
        # Call: task-query cursor=<cursor>
        # Assert: next 25 tasks

    def test_spec_stats_large_spec_parity(self, edge_adapters):
        """Stats calculation on large spec."""
        # Call: spec-stats parity-test-large
        # Assert: total_tasks=100, completed=40, phases=5

    def test_progress_large_spec_parity(self, edge_adapters):
        """Progress on large spec."""
        # Assert: percentage=40 (40/100 completed)

    def test_journal_pagination_parity(self, edge_adapters):
        """Paginate through journal entries."""
        # Call: journal-list limit=2
        # Assert: 2 entries, cursor for more
```

#### Empty Spec
```python
class TestEmptySpecParity:
    def test_task_list_empty_parity(self, empty_spec_dir):
        """List tasks on empty spec returns empty list."""
        # Call: task-list parity-test-empty
        # Assert: tasks=[], count=0

    def test_task_next_empty_parity(self, empty_spec_dir):
        """Next task on empty spec returns not found."""
        # Call: task-next parity-test-empty
        # Assert: found=False or appropriate error

    def test_progress_empty_parity(self, empty_spec_dir):
        """Progress on empty spec."""
        # Assert: total_tasks=0, percentage=0 or 100

    def test_spec_stats_empty_parity(self, empty_spec_dir):
        """Stats on empty spec."""
        # Assert: handles gracefully
```

#### All Blocked
```python
class TestAllBlockedParity:
    def test_list_blocked_all_parity(self, all_blocked_dir):
        """List all blocked tasks."""
        # Call: task-list-blocked parity-test-all-blocked
        # Assert: 3 blocked tasks

    def test_next_task_all_blocked_parity(self, all_blocked_dir):
        """Next task when all are blocked."""
        # Call: task-next parity-test-all-blocked
        # Assert: No actionable task found

    def test_blocker_types_parity(self, all_blocked_dir):
        """Verify blocker types are preserved."""
        # Assert: dependency, technical, resource types present
```

---

## Phase 4: Review/Verification Tests

### File: `test_review_ops.py`

**Fixtures to use:**
- `review_spec_dir` / `review_adapters` - verify nodes
- `verification_results_dir` - pre-populated results

**Tests to implement:**

#### Verification Operations
```python
class TestVerificationParity:
    def test_verification_add_parity(self, review_adapters):
        """Add verification result to verify node."""
        # Call: verification-add verify-2-1 result=PASSED command="pytest" output="10 passed"

    def test_verification_execute_parity(self, review_adapters):
        """Execute verification command."""
        # Call: verification-execute verify-2-1
        # Note: May need mocking for actual command execution

    def test_format_verification_summary_parity(self, verification_results_dir):
        """Format summary from pre-populated results."""
        # Setup: review_with_results fixture (2 PASSED, 1 FAILED, 1 PARTIAL)
        # Call: verification-format-summary
        # Assert: summary shows 2 passed, 1 failed, 1 partial
```

#### Spec Review
```python
class TestSpecReviewParity:
    def test_spec_review_basic_parity(self, review_adapters):
        """Basic spec review."""
        # Call: spec-review parity-test-review review_type=quick
        # Note: LLM-powered - may need mocking or skip in CI
```

---

## Implementation Notes

### Adapter Method Mapping

Ensure the parity test harness adapters implement these methods:

| Operation | Foundry Method | SDD CLI Command |
|-----------|----------------|-----------------|
| task-add | `add_task()` | `sdd add-task` |
| task-remove | `remove_task()` | `sdd remove-task` |
| assumption-add | `add_assumption()` | `sdd add-assumption` |
| assumption-list | `list_assumptions()` | `sdd list-assumptions` |
| revision-add | `add_revision()` | `sdd add-revision` |
| task-update-estimate | `update_estimate()` | `sdd update-estimate` |
| task-update-metadata | `update_task_metadata()` | `sdd update-task-metadata` |
| spec-analyze-deps | `analyze_deps()` | `sdd analyze-deps` |
| spec-detect-cycles | `detect_cycles()` | `sdd find-circular-deps` |
| spec-find-patterns | `find_patterns()` | `sdd find-pattern` |
| verification-add | `add_verification()` | `sdd add-verification` |
| verification-execute | `execute_verification()` | `sdd execute-verify` |
| verification-format-summary | `format_verification_summary()` | `sdd format-verification-summary` |

### Test Markers

Add appropriate pytest markers:

```python
@pytest.mark.parity  # For side-by-side comparison tests
@pytest.mark.authoring  # Category marker
@pytest.mark.analysis
@pytest.mark.edge_case
@pytest.mark.review
@pytest.mark.slow  # For large spec tests
```

### Normalizer Updates

May need to update `harness/normalizers.py` for new fields:
- `assumptions` array normalization
- `revision_history` array normalization
- `verification` result fields
- `blocker_type` field mapping

### Comparator Assertions

Use existing comparator patterns:
```python
from .harness.comparators import ResultComparator

comparator = ResultComparator()
comparator.assert_parity(foundry_result, sdd_result)
comparator.assert_key_match(foundry_result, sdd_result, "task_id")
comparator.assert_success(foundry_result, sdd_result)
comparator.assert_both_error(foundry_result, sdd_result)
```

---

## Estimated Effort

| Phase | Tests | Complexity | Estimate |
|-------|-------|------------|----------|
| Phase 1: Authoring | ~15 tests | Medium | Requires adapter method additions |
| Phase 2: Analysis | ~10 tests | Medium | Cycle detection may have edge cases |
| Phase 3: Edge Cases | ~15 tests | Low-Medium | Mostly straightforward |
| Phase 4: Review | ~5 tests | High | May need mocking for LLM/execution |

**Total:** ~45 new parity tests

---

## Priority Order

1. **Phase 1: Authoring** - Most commonly used operations, high value
2. **Phase 3: Edge Cases** - Catches bugs early, good coverage
3. **Phase 2: Analysis** - Important for spec health
4. **Phase 4: Review** - Lower priority, may need mocking infrastructure

---

## Prerequisites Before Implementation

1. **Verify adapter methods exist** - Check `foundry_adapter.py` and `sdd_adapter.py` have methods for all operations
2. **Add missing adapter methods** - Implement any gaps in the adapter interface
3. **Update normalizers** - Add normalization for new field types
4. **Run existing tests** - Ensure baseline is green before adding new tests
