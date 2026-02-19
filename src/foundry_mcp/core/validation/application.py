"""Fix action application for spec files."""

import copy
import json
from pathlib import Path
from typing import List

from foundry_mcp.core.validation.models import FixAction, FixReport
from foundry_mcp.core.validation.stats import _recalculate_counts


def apply_fixes(
    actions: List[FixAction],
    spec_path: str,
    *,
    dry_run: bool = False,
    create_backup: bool = True,
    capture_diff: bool = False,
) -> FixReport:
    """
    Apply fix actions to a spec file.

    Args:
        actions: List of FixAction objects to apply
        spec_path: Path to spec file
        dry_run: If True, don't actually save changes
        create_backup: If True, create backup before modifying
        capture_diff: If True, capture before/after state

    Returns:
        FixReport with results
    """
    report = FixReport(spec_path=spec_path)

    if dry_run:
        report.skipped_actions.extend(actions)
        return report

    try:
        with open(spec_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return report

    # Capture before state
    if capture_diff:
        report.before_state = copy.deepcopy(data)

    # Create backup
    if create_backup:
        backup_path = Path(spec_path).with_suffix(".json.backup")
        try:
            with open(backup_path, "w") as f:
                json.dump(data, f, indent=2)
            report.backup_path = str(backup_path)
        except OSError:
            pass

    # Apply each action
    for action in actions:
        try:
            action.apply(data)
            report.applied_actions.append(action)
        except Exception:
            report.skipped_actions.append(action)

    # Recalculate counts after all fixes
    if report.applied_actions:
        _recalculate_counts(data)

    # Capture after state
    if capture_diff:
        report.after_state = copy.deepcopy(data)

    # Save changes
    try:
        with open(spec_path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass

    return report
