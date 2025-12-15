"""
Workflow prompts for foundry-mcp.

Provides MCP prompts for common SDD workflows like starting features,
debugging tests, and completing phases.
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.spec import (
    load_spec,
    list_specs,
    find_specs_directory,
)
from foundry_mcp.core.progress import get_progress_summary, list_phases

logger = logging.getLogger(__name__)


# Schema version for prompt responses
SCHEMA_VERSION = "1.0.0"


def register_workflow_prompts(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register workflow prompts with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    def _get_specs_dir() -> Optional[str]:
        """Get the specs directory path."""
        specs_dir = config.specs_dir or find_specs_directory()
        return str(specs_dir) if specs_dir else None

    @mcp.prompt(
        description='{"desc": "Start a new feature spec with phases and tasks", "args": {"feature_name": "Name of feature", "description": "Optional description", "template": "basic|feature|bugfix"}, "returns": "Formatted prompt"}'
    )
    def start_feature(
        feature_name: str, description: Optional[str] = None, template: str = "feature"
    ) -> str:
        """
        Start a new feature implementation.

        Creates a structured prompt to guide the AI through setting up
        a new feature spec with proper phases and tasks.

        Args:
            feature_name: Name of the feature to implement
            description: Optional description of the feature
            template: Template to use (basic, feature, bugfix)

        Returns:
            Formatted prompt for starting a new feature
        """
        specs_dir = _get_specs_dir()

        # Check for existing specs
        existing_specs = []
        if specs_dir:
            from pathlib import Path

            specs = list_specs(specs_dir=Path(specs_dir), status="active")
            existing_specs = [s["spec_id"] for s in specs[:5]]

        prompt_parts = [
            f"# Start New Feature: {feature_name}",
            "",
            "## Overview",
            f"Feature Name: {feature_name}",
        ]

        if description:
            prompt_parts.append(f"Description: {description}")

        prompt_parts.extend(
            [
                f"Template: {template}",
                "",
                "## Instructions",
                "",
                "Please help me set up a new SDD specification for this feature.",
                "",
                "### Step 1: Create the Spec",
                f"Use the `{template}` template to create a new spec file.",
                "The spec should be placed in `specs/pending/` initially.",
                "",
                "### Step 2: Define Phases",
                "Based on the feature requirements, define appropriate phases:",
            ]
        )

        if template == "feature":
            prompt_parts.extend(
                [
                    "1. **Design Phase**: Architecture decisions, API design, data models",
                    "2. **Implementation Phase**: Core functionality, tests, documentation",
                    "3. **Verification Phase**: Integration tests, manual QA, sign-off",
                ]
            )
        elif template == "bugfix":
            prompt_parts.extend(
                [
                    "1. **Investigation Phase**: Reproduce bug, identify root cause",
                    "2. **Fix & Verify Phase**: Implement fix, verify resolution",
                ]
            )
        else:
            prompt_parts.extend(
                [
                    "1. **Implementation Phase**: Core tasks for the feature",
                ]
            )

        prompt_parts.extend(
            [
                "",
                "### Step 3: Break Down Tasks",
                "For each phase, create specific, actionable tasks with:",
                "- Clear acceptance criteria",
                "- Estimated effort (in hours)",
                "- Dependencies between tasks",
                "",
                "### Step 4: Activate When Ready",
                "Once the spec is reviewed and approved, move it to `specs/active/`.",
                "",
            ]
        )

        if existing_specs:
            prompt_parts.extend(
                [
                    "## Active Specs",
                    "Note: The following specs are currently active:",
                    *[f"- {spec}" for spec in existing_specs],
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "## Available Tools",
                "Use these MCP tools to manage the spec:",
                '- `spec(action="validate")`: Validate spec structure',
                '- `lifecycle(action="activate")`: Move spec to active',
                '- `task(action="next")`: Find next task to work on',
                "",
                "Ready to begin? Please provide more details about the feature requirements.",
            ]
        )

        return "\n".join(prompt_parts)

    @mcp.prompt(
        description='{"desc": "Debug a failing test systematically", "args": {"test_name": "Failing test name (optional)", "error_message": "Error output (optional)", "spec_id": "Related spec ID (optional)"}, "returns": "Debugging workflow prompt"}'
    )
    def debug_test(
        test_name: Optional[str] = None,
        error_message: Optional[str] = None,
        spec_id: Optional[str] = None,
    ) -> str:
        """
        Debug a failing test.

        Creates a structured prompt to guide the AI through debugging
        a test failure systematically.

        Args:
            test_name: Name of the failing test (optional)
            error_message: Error message from the test (optional)
            spec_id: Related spec ID if working on a specific task

        Returns:
            Formatted prompt for debugging a test failure
        """
        prompt_parts = [
            "# Debug Test Failure",
            "",
            "## Problem Details",
        ]

        if test_name:
            prompt_parts.append(f"**Test Name:** `{test_name}`")
        else:
            prompt_parts.append("**Test Name:** Not specified")

        if error_message:
            prompt_parts.extend(
                [
                    "",
                    "**Error Message:**",
                    "```",
                    error_message[:500] if len(error_message) > 500 else error_message,
                    "```" if error_message else "",
                ]
            )

        if spec_id:
            prompt_parts.append(f"**Related Spec:** {spec_id}")

        prompt_parts.extend(
            [
                "",
                "## Debugging Workflow",
                "",
                "### Step 1: Understand the Failure",
                "- What is the test trying to verify?",
                "- What was the expected behavior?",
                "- What actually happened?",
                "",
                "### Step 2: Reproduce Locally",
                "Run the test in isolation:",
                "```bash",
                f"pytest {test_name or 'path/to/test.py'} -v --tb=long",
                "```",
                "",
                "### Step 3: Identify Root Cause",
                "Common causes to check:",
                "- [ ] Missing or incorrect test fixtures",
                "- [ ] State from previous tests",
                "- [ ] Environment differences",
                "- [ ] Race conditions or timing issues",
                "- [ ] Incorrect assertions",
                "- [ ] Changed API or implementation",
                "",
                "### Step 4: Implement Fix",
                "Based on the root cause:",
                "- If test is wrong: Update the test",
                "- If code is wrong: Fix the implementation",
                "- If both: Fix both and add regression tests",
                "",
                "### Step 5: Verify",
                "- Run the fixed test",
                "- Run related tests",
                "- Run full test suite if changes are significant",
                "",
                "## Available Tools",
                "Use these MCP tools to help debug:",
                '- `test(action="run", preset="quick")`: Run tests',
                '- `test(action="discover")`: Discover available tests',
                "",
            ]
        )

        if spec_id:
            prompt_parts.extend(
                [
                    "## Spec Context",
                    f"This test is related to spec `{spec_id}`.",
                    "After fixing, remember to update the task status.",
                    "",
                ]
            )

        prompt_parts.append(
            "Please provide the test output or more details about the failure."
        )

        return "\n".join(prompt_parts)

    @mcp.prompt(
        description='{"desc": "Complete all tasks in a phase and move to next", "args": {"spec_id": "Specification ID", "phase_id": "Phase ID (optional, uses current)"}, "returns": "Phase completion checklist"}'
    )
    def complete_phase(spec_id: str, phase_id: Optional[str] = None) -> str:
        """
        Complete a phase in a specification.

        Creates a structured prompt to guide the AI through completing
        all tasks in a phase and moving to the next one.

        Args:
            spec_id: Specification ID
            phase_id: Phase ID to complete (optional, uses current if not specified)

        Returns:
            Formatted prompt for completing a phase
        """
        specs_dir = _get_specs_dir()

        prompt_parts = [
            f"# Complete Phase for Spec: {spec_id}",
            "",
        ]

        # Try to load spec and get phase info
        spec_data = None
        phase_info = None
        progress_info = None

        if specs_dir:
            from pathlib import Path

            spec_data = load_spec(spec_id, Path(specs_dir))

            if spec_data:
                progress_info = get_progress_summary(spec_data)
                phases = list_phases(spec_data)

                if phase_id:
                    phase_info = next((p for p in phases if p["id"] == phase_id), None)
                else:
                    # Find current in-progress phase
                    phase_info = next(
                        (p for p in phases if p["status"] == "in_progress"), None
                    )
                    if not phase_info:
                        # Find first pending phase
                        phase_info = next(
                            (p for p in phases if p["status"] == "pending"), None
                        )

        if spec_data and progress_info:
            prompt_parts.extend(
                [
                    "## Current Progress",
                    f"- **Overall:** {progress_info['percentage']}% complete",
                    f"- **Tasks:** {progress_info['completed_tasks']}/{progress_info['total_tasks']} done",
                    "",
                ]
            )

        if phase_info:
            prompt_parts.extend(
                [
                    "## Phase Details",
                    f"- **Phase:** {phase_info.get('title', phase_info.get('id', 'Unknown'))}",
                    f"- **Status:** {phase_info.get('status', 'unknown')}",
                    f"- **Progress:** {phase_info.get('completed_tasks', 0)}/{phase_info.get('total_tasks', 0)} tasks",
                    "",
                ]
            )
        elif phase_id:
            prompt_parts.extend(
                [
                    f"## Phase: {phase_id}",
                    "(Phase details not available)",
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "## Completion Checklist",
                "",
                "### Step 1: Review Remaining Tasks",
                "List all pending/in-progress tasks in this phase:",
                "```bash",
                '# Use task(action="progress") to check status',
                "```",
                "",
                "### Step 2: Complete Each Task",
                "For each remaining task:",
                '1. Start the task (`task(action="start")`)',
                "2. Implement the required changes",
                "3. Verify the implementation",
                '4. Complete the task with journal entry (`task(action="complete")`)',
                "",
                "### Step 3: Run Verification",
                "Before marking the phase complete:",
                "- [ ] All tasks show status: completed",
                "- [ ] All verification tasks pass",
                "- [ ] No blockers remain",
                "- [ ] Tests pass for this phase's changes",
                "",
                "### Step 4: Phase Wrap-up",
                "Once all tasks are done:",
                "1. Review the phase journal entries",
                "2. Update any documentation",
                "3. The phase will auto-complete when all children are done",
                "",
                "### Step 5: Prepare for Next Phase",
                "After this phase completes:",
                "1. Review the next phase requirements",
                '2. Use `task(action="next")` to get the first task',
                "3. Continue the workflow",
                "",
                "## Available Tools",
                '- `task(action="progress")`: Check spec/phase progress',
                '- `task(action="next")`: Find next task to work on',
                '- `task(action="prepare")`: Get task context',
                '- `task(action="complete")`: Mark task done with journal',
                '- `test(action="run", preset="quick")`: Run verification tests',
                "",
                "Ready to proceed? Let's review the remaining tasks.",
            ]
        )

        return "\n".join(prompt_parts)

    @mcp.prompt(
        description='{"desc": "Review spec status, progress, and journal", "args": {"spec_id": "Specification ID to review"}, "returns": "Spec overview with actions"}'
    )
    def review_spec(spec_id: str) -> str:
        """
        Review a specification's status and progress.

        Creates a comprehensive overview of a spec for review purposes.

        Args:
            spec_id: Specification ID to review

        Returns:
            Formatted prompt with spec review information
        """
        specs_dir = _get_specs_dir()

        prompt_parts = [
            f"# Spec Review: {spec_id}",
            "",
        ]

        # Try to load spec
        spec_data = None
        if specs_dir:
            from pathlib import Path

            spec_data = load_spec(spec_id, Path(specs_dir))

        if not spec_data:
            prompt_parts.extend(
                [
                    "**Error:** Spec not found or could not be loaded.",
                    "",
                    f"Please verify the spec ID `{spec_id}` is correct.",
                    'Use `spec(action="list")` to see available specs.',
                ]
            )
            return "\n".join(prompt_parts)

        # Get metadata
        metadata = spec_data.get("metadata", {})
        title = metadata.get("title", spec_data.get("title", "Untitled"))

        prompt_parts.extend(
            [
                f"## {title}",
                "",
            ]
        )

        if metadata.get("description"):
            prompt_parts.extend(
                [
                    "### Description",
                    metadata["description"],
                    "",
                ]
            )

        # Get progress
        progress_info = get_progress_summary(spec_data)
        prompt_parts.extend(
            [
                "### Progress Overview",
                f"- **Completion:** {progress_info['percentage']}%",
                f"- **Tasks:** {progress_info['completed_tasks']}/{progress_info['total_tasks']}",
                f"- **Remaining:** {progress_info.get('remaining_tasks', 0)}",
                "",
            ]
        )

        # Get phases
        phases = list_phases(spec_data)
        if phases:
            prompt_parts.extend(
                [
                    "### Phases",
                    "",
                ]
            )
            for phase in phases:
                status_icon = {
                    "completed": "✅",
                    "in_progress": "🔄",
                    "pending": "⏳",
                    "blocked": "🚫",
                }.get(phase.get("status", "pending"), "❓")

                pct = phase.get("percentage", 0)
                prompt_parts.append(
                    f"- {status_icon} **{phase.get('title', phase['id'])}**: "
                    f"{phase.get('completed_tasks', 0)}/{phase.get('total_tasks', 0)} ({pct}%)"
                )
            prompt_parts.append("")

        # Get journal summary
        journal = spec_data.get("journal", [])
        if journal:
            recent = journal[-5:]  # Last 5 entries
            prompt_parts.extend(
                [
                    "### Recent Journal Entries",
                    "",
                ]
            )
            for entry in reversed(recent):
                entry_type = entry.get("entry_type", "note")
                title = entry.get("title", "Untitled")
                prompt_parts.append(f"- [{entry_type}] {title}")
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "### Actions",
                "",
                "What would you like to do?",
                '1. Continue with next task (`task(action="next")`)',
                "2. View specific phase details",
                "3. Check blocked tasks",
                "4. Review journal entries",
                "5. Run verification tests",
            ]
        )

        return "\n".join(prompt_parts)

    logger.debug(
        "Registered workflow prompts: start_feature, debug_test, complete_phase, review_spec"
    )
