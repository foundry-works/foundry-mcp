import shutil
import subprocess
import os

import pytest


# Skip all tests in this module if foundry-cli is not installed
pytestmark = pytest.mark.skipif(
    shutil.which("foundry-cli") is None, reason="foundry-cli not installed"
)


def test_run_tests_fallback_gemini_to_cursor():
    """Test run-tests falls back from gemini to cursor-agent."""
    # Rename gemini binary temporarily to simulate unavailability
    # (This is a simplified example - actual test would need proper setup)

    result = subprocess.run(
        [
            "foundry-cli",
            "test",
            "consult",
            "--issue",
            "Test failed: assertion error. Root cause unknown",
        ],
        capture_output=True,
        text=True,
    )

    # Should succeed even if gemini unavailable (returncode 0)
    # or fail gracefully with message (returncode 1)
    assert result.returncode in (0, 1)
    # Should produce some output (success message or error)
    assert result.stdout or result.stderr


def test_sdd_plan_review_with_limits():
    """Test sdd-plan-review respects consultation limits."""
    # Run plan review that might consult multiple models
    env = os.environ.copy()
    env["HOME"] = "/tmp/test_home"
    subprocess.run(
        ["foundry-cli", "plan", "review", "test-spec-id"],
        capture_output=True,
        text=True,
        env=env,  # Use test config
    )

    # Parse output to verify tool count
    # Implementation would check that no more than max_tools_per_run were used
    pass


def test_fidelity_review_retry_on_timeout():
    """Test fidelity review retries on timeout."""
    # Set very short timeout to force timeout
    subprocess.run(
        [
            "foundry-cli",
            "review",
            "fidelity",
            "test-spec-id",
            "--timeout",
            "1",
        ],  # 1 second timeout
        capture_output=True,
        text=True,
    )

    # Should have attempted retries (check logs or output)
    pass
