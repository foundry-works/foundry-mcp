"""Contract test for the public API surface of foundry_mcp.core.task.

Asserts that all public symbols remain importable after the module is split
into a package.  Any addition or removal of public symbols will cause a test
failure, making the change explicit and reviewable.
"""

from __future__ import annotations

import importlib
import inspect
import types

import pytest


def _is_defined_in_module(mod: types.ModuleType, name: str) -> bool:
    """Return True if *name* is defined in *mod* (not just imported).

    Filters out re-imported typing symbols, helper functions from other
    modules, and class/type objects not originally defined here.
    """
    obj = getattr(mod, name, None)
    if obj is None:
        return False
    # Functions: check __module__
    if inspect.isfunction(obj):
        return getattr(obj, "__module__", None) == mod.__name__
    # Classes: check __module__
    if inspect.isclass(obj):
        return getattr(obj, "__module__", None) == mod.__name__
    # For non-callable constants (tuples, sets, ints, strings, etc.),
    # we can't reliably determine origin — include them (they're the
    # constants we want to track).
    if not callable(obj):
        return True
    # Default: exclude (likely an imported callable like asdict)
    return False


# ---------------------------------------------------------------------------
# Phase 1 baseline: every public symbol in foundry_mcp.core.task
# ---------------------------------------------------------------------------

# Functions (17)
BASELINE_FUNCTIONS = sorted(
    [
        "is_unblocked",
        "is_in_current_phase",
        "get_next_task",
        "check_dependencies",
        "get_previous_sibling",
        "get_parent_context",
        "get_phase_context",
        "get_task_journal_summary",
        "prepare_task",
        "add_task",
        "remove_task",
        "update_estimate",
        "manage_task_dependency",
        "update_task_metadata",
        "move_task",
        "update_task_requirements",
        "batch_update_tasks",
    ]
)

# Constants (11 — CATEGORIES is the raw import aliased to TASK_CATEGORIES)
BASELINE_CONSTANTS = sorted(
    [
        "CATEGORIES",
        "TASK_TYPES",
        "COMPLEXITY_LEVELS",
        "VERIFICATION_TYPES",
        "TASK_CATEGORIES",
        "DEPENDENCY_TYPES",
        "REQUIREMENT_TYPES",
        "MAX_REQUIREMENTS_PER_TASK",
        "BATCH_ALLOWED_STATUSES",
        "MAX_PATTERN_LENGTH",
        "DEFAULT_MAX_MATCHES",
    ]
)

# All public symbols
BASELINE_ALL = sorted(BASELINE_FUNCTIONS + BASELINE_CONSTANTS)

# Symbols re-exported via foundry_mcp.core.__init__
CORE_REEXPORTS = sorted(
    [
        "is_unblocked",
        "is_in_current_phase",
        "get_next_task",
        "check_dependencies",
        "get_previous_sibling",
        "get_parent_context",
        "get_phase_context",
        "get_task_journal_summary",
        "prepare_task",
    ]
)

# Consumer import map: file -> symbols used
CONSUMER_IMPORTS = {
    "foundry_mcp.core.batch_operations": [
        "is_unblocked",
        "check_dependencies",
        "get_parent_context",
        "get_phase_context",
    ],
    "foundry_mcp.cli.commands.modify": [
        "add_task",
        "remove_task",
    ],
    "foundry_mcp.cli.commands.tasks": [
        "check_dependencies",
        "get_next_task",
        "get_parent_context",
        "get_phase_context",
        "get_previous_sibling",
        "get_task_journal_summary",
        "prepare_task",
    ],
    "foundry_mcp.tools.unified.authoring": [
        "TASK_TYPES",
    ],
    "foundry_mcp.tools.unified.task": [
        "add_task",
        "batch_update_tasks",
        "check_dependencies",
        "get_next_task",
        "manage_task_dependency",
        "move_task",
        "prepare_task",
        "remove_task",
        "REQUIREMENT_TYPES",
        "update_estimate",
        "update_task_metadata",
        "update_task_requirements",
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTaskModulePublicAPI:
    """Assert all baseline symbols are importable from foundry_mcp.core.task."""

    def _get_module(self):
        return importlib.import_module("foundry_mcp.core.task")

    @pytest.mark.parametrize("symbol", BASELINE_ALL, ids=BASELINE_ALL)
    def test_symbol_importable(self, symbol):
        """Each baseline symbol should be importable."""
        mod = self._get_module()
        assert hasattr(mod, symbol), f"foundry_mcp.core.task.{symbol} is missing"

    @pytest.mark.parametrize("symbol", BASELINE_FUNCTIONS, ids=BASELINE_FUNCTIONS)
    def test_function_is_callable(self, symbol):
        """Each baseline function should be callable."""
        mod = self._get_module()
        obj = getattr(mod, symbol)
        assert callable(obj), f"foundry_mcp.core.task.{symbol} should be callable"

    @pytest.mark.parametrize("symbol", BASELINE_CONSTANTS, ids=BASELINE_CONSTANTS)
    def test_constant_is_not_callable(self, symbol):
        """Each baseline constant should not be a function."""
        mod = self._get_module()
        obj = getattr(mod, symbol)
        assert not inspect.isfunction(obj), f"foundry_mcp.core.task.{symbol} should be a constant, not a function"

    def test_no_unexpected_public_symbols(self):
        """No new public functions/constants defined in task.py should appear
        without updating the baseline.

        Filters out re-imported names (typing, helper modules) by checking
        that the symbol is actually defined in this module.
        """
        mod = self._get_module()
        actual = sorted(
            name
            for name in dir(mod)
            if not name.startswith("_")
            and not inspect.ismodule(getattr(mod, name))
            and _is_defined_in_module(mod, name)
        )
        unexpected = sorted(set(actual) - set(BASELINE_ALL))
        assert not unexpected, f"Unexpected public symbols found — update BASELINE if intentional: {unexpected}"

    def test_baseline_counts(self):
        """Baseline has expected counts."""
        assert len(BASELINE_FUNCTIONS) == 17
        assert len(BASELINE_CONSTANTS) == 11
        assert len(BASELINE_ALL) == 28


class TestCoreReexports:
    """Assert symbols re-exported via foundry_mcp.core remain available."""

    @pytest.mark.parametrize("symbol", CORE_REEXPORTS, ids=CORE_REEXPORTS)
    def test_reexport_available(self, symbol):
        """Each re-exported symbol should be importable from foundry_mcp.core."""
        core = importlib.import_module("foundry_mcp.core")
        assert hasattr(core, symbol), f"foundry_mcp.core.{symbol} re-export is missing"


class TestConsumerImports:
    """Assert all known consumer imports still work."""

    @pytest.mark.parametrize(
        "consumer, symbols",
        list(CONSUMER_IMPORTS.items()),
        ids=list(CONSUMER_IMPORTS.keys()),
    )
    def test_consumer_can_import(self, consumer, symbols):
        """Each consumer's imports should resolve."""
        mod = importlib.import_module("foundry_mcp.core.task")
        for sym in symbols:
            assert hasattr(mod, sym), f"{consumer} imports {sym} from foundry_mcp.core.task but it's missing"
