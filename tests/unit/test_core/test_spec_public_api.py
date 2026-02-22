"""Contract test for the public API surface of foundry_mcp.core.spec.

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
# Phase 1 baseline: every public symbol in foundry_mcp.core.spec
# ---------------------------------------------------------------------------

# Functions (33)
BASELINE_FUNCTIONS = sorted(
    [
        "add_assumption",
        "add_phase",
        "add_phase_bulk",
        "add_revision",
        "apply_phase_template",
        "backup_spec",
        "check_spec_completeness",
        "create_spec",
        "detect_duplicate_tasks",
        "diff_specs",
        "find_git_root",
        "find_replace_in_spec",
        "find_spec_file",
        "find_specs_directory",
        "generate_spec_data",
        "generate_spec_id",
        "get_node",
        "get_phase_template_structure",
        "get_template_structure",
        "list_assumptions",
        "list_spec_backups",
        "list_specs",
        "load_spec",
        "move_phase",
        "recalculate_actual_hours",
        "remove_phase",
        "resolve_spec_file",
        "rollback_spec",
        "save_spec",
        "update_frontmatter",
        "update_node",
        "update_phase_metadata",
    ]
)

# Constants (10)
BASELINE_CONSTANTS = sorted(
    [
        "CATEGORIES",
        "DEFAULT_BACKUP_PAGE_SIZE",
        "DEFAULT_DIFF_MAX_RESULTS",
        "DEFAULT_MAX_BACKUPS",
        "FRONTMATTER_KEYS",
        "MAX_BACKUP_PAGE_SIZE",
        "PHASE_TEMPLATES",
        "TEMPLATES",
        "TEMPLATE_DESCRIPTIONS",
        "VERIFICATION_TYPES",
    ]
)

# All public symbols
BASELINE_ALL = sorted(BASELINE_FUNCTIONS + BASELINE_CONSTANTS)

# Symbols re-exported via foundry_mcp.core.__init__
CORE_REEXPORTS = sorted(
    [
        "find_specs_directory",
        "find_spec_file",
        "resolve_spec_file",
        "load_spec",
        "save_spec",
        "backup_spec",
        "list_specs",
        "get_node",
        "update_node",
    ]
)

# Consumer import map: file -> symbols used
CONSUMER_IMPORTS = {
    "foundry_mcp.core": [
        "find_specs_directory",
        "find_spec_file",
        "resolve_spec_file",
        "load_spec",
        "save_spec",
        "backup_spec",
        "list_specs",
        "get_node",
        "update_node",
    ],
    "foundry_mcp.core.batch_operations": [
        "load_spec",
        "find_specs_directory",
        "save_spec",
    ],
    "foundry_mcp.tools.unified.spec": [
        "TEMPLATES",
        "TEMPLATE_DESCRIPTIONS",
        "check_spec_completeness",
        "detect_duplicate_tasks",
        "diff_specs",
        "find_spec_file",
        "find_specs_directory",
        "list_spec_backups",
        "list_specs",
        "load_spec",
        "recalculate_actual_hours",
    ],
    "foundry_mcp.tools.unified.authoring": [
        "CATEGORIES",
        "PHASE_TEMPLATES",
        "TEMPLATES",
        "add_assumption",
        "add_phase",
        "add_phase_bulk",
        "add_revision",
        "apply_phase_template",
        "create_spec",
        "find_replace_in_spec",
        "find_specs_directory",
        "generate_spec_data",
        "get_phase_template_structure",
        "list_assumptions",
        "load_spec",
        "move_phase",
        "remove_phase",
        "rollback_spec",
        "update_frontmatter",
        "update_phase_metadata",
    ],
    "foundry_mcp.tools.unified.task": [
        "find_specs_directory",
        "load_spec",
        "save_spec",
    ],
    "foundry_mcp.tools.unified.journal": [
        "find_specs_directory",
        "load_spec",
        "save_spec",
    ],
    "foundry_mcp.tools.unified.lifecycle": [
        "find_specs_directory",
    ],
    "foundry_mcp.tools.unified.plan": [
        "find_specs_directory",
    ],
    "foundry_mcp.tools.unified.review": [
        "find_spec_file",
        "find_specs_directory",
        "load_spec",
    ],
    "foundry_mcp.tools.unified.verification": [
        "find_specs_directory",
        "load_spec",
        "save_spec",
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpecModulePublicAPI:
    """Assert all baseline symbols are importable from foundry_mcp.core.spec."""

    def _get_module(self):
        return importlib.import_module("foundry_mcp.core.spec")

    @pytest.mark.parametrize("symbol", BASELINE_ALL, ids=BASELINE_ALL)
    def test_symbol_importable(self, symbol):
        """Each baseline symbol should be importable."""
        mod = self._get_module()
        assert hasattr(mod, symbol), f"foundry_mcp.core.spec.{symbol} is missing"

    @pytest.mark.parametrize("symbol", BASELINE_FUNCTIONS, ids=BASELINE_FUNCTIONS)
    def test_function_is_callable(self, symbol):
        """Each baseline function should be callable."""
        mod = self._get_module()
        obj = getattr(mod, symbol)
        assert callable(obj), f"foundry_mcp.core.spec.{symbol} should be callable"

    @pytest.mark.parametrize("symbol", BASELINE_CONSTANTS, ids=BASELINE_CONSTANTS)
    def test_constant_is_not_callable(self, symbol):
        """Each baseline constant should not be a function."""
        mod = self._get_module()
        obj = getattr(mod, symbol)
        assert not inspect.isfunction(obj), f"foundry_mcp.core.spec.{symbol} should be a constant, not a function"

    def test_no_unexpected_public_symbols(self):
        """No new public functions/constants defined in spec.py should appear
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
        assert len(BASELINE_FUNCTIONS) == 32
        assert len(BASELINE_CONSTANTS) == 10
        assert len(BASELINE_ALL) == 42


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
        mod = importlib.import_module("foundry_mcp.core.spec")
        for sym in symbols:
            assert hasattr(mod, sym), f"{consumer} imports {sym} from foundry_mcp.core.spec but it's missing"
