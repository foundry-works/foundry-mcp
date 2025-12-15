"""
Testing operations for foundry-mcp.
Provides functions for running tests and test discovery.
"""

import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Schema version for compatibility tracking
SCHEMA_VERSION = "1.0.0"


# Presets for common test configurations
TEST_PRESETS = {
    "quick": {
        "timeout": 60,
        "verbose": False,
        "fail_fast": True,
        "markers": "not slow",
    },
    "full": {
        "timeout": 300,
        "verbose": True,
        "fail_fast": False,
        "markers": None,
    },
    "unit": {
        "timeout": 120,
        "verbose": True,
        "fail_fast": False,
        "markers": "unit",
        "pattern": "test_*.py",
    },
    "integration": {
        "timeout": 300,
        "verbose": True,
        "fail_fast": False,
        "markers": "integration",
    },
    "smoke": {
        "timeout": 30,
        "verbose": False,
        "fail_fast": True,
        "markers": "smoke",
    },
}


# Data structures

@dataclass
class TestResult:
    """
    Result of a single test.
    """
    name: str
    outcome: str  # passed, failed, skipped, error
    duration: float = 0.0
    message: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass
class TestRunResult:
    """
    Result of a test run.
    """
    success: bool
    execution_id: str = ""
    schema_version: str = SCHEMA_VERSION
    timestamp: str = ""
    duration: float = 0.0
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    tests: List[TestResult] = field(default_factory=list)
    command: str = ""
    cwd: str = ""
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class DiscoveredTest:
    """
    A discovered test.
    """
    name: str
    file_path: str
    line_number: Optional[int] = None
    markers: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class TestDiscoveryResult:
    """
    Result of test discovery.
    """
    success: bool
    schema_version: str = SCHEMA_VERSION
    timestamp: str = ""
    total: int = 0
    tests: List[DiscoveredTest] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.total = len(self.tests)


# Main test runner

class TestRunner:
    """
    Test runner for pytest-based projects.
    """

    def __init__(self, workspace: Optional[Path] = None):
        """
        Initialize test runner.

        Args:
            workspace: Repository root (defaults to current directory)
        """
        self.workspace = workspace or Path.cwd()

    def run_tests(
        self,
        target: Optional[str] = None,
        preset: Optional[str] = None,
        timeout: int = 300,
        verbose: bool = True,
        fail_fast: bool = False,
        markers: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ) -> TestRunResult:
        """
        Run tests using pytest.

        Args:
            target: Test target (file, directory, or test name)
            preset: Use a preset configuration (quick, full, unit, integration, smoke)
            timeout: Timeout in seconds
            verbose: Enable verbose output
            fail_fast: Stop on first failure
            markers: Pytest markers expression
            extra_args: Additional pytest arguments

        Returns:
            TestRunResult with test outcomes
        """
        # Apply preset if specified
        if preset and preset in TEST_PRESETS:
            preset_config = TEST_PRESETS[preset]
            timeout = preset_config.get("timeout", timeout)
            verbose = preset_config.get("verbose", verbose)
            fail_fast = preset_config.get("fail_fast", fail_fast)
            markers = preset_config.get("markers", markers)

        # Build command
        cmd = ["python", "-m", "pytest"]

        if target:
            cmd.append(target)

        if verbose:
            cmd.append("-v")

        if fail_fast:
            cmd.append("-x")

        if markers:
            cmd.extend(["-m", markers])

        # Add JSON output for parsing
        cmd.append("--tb=short")

        if extra_args:
            cmd.extend(extra_args)

        command_str = " ".join(cmd)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            # Parse output
            tests, passed, failed, skipped, errors = self._parse_pytest_output(result.stdout)

            return TestRunResult(
                success=result.returncode == 0,
                duration=0.0,  # Would need timing wrapper
                total=len(tests),
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                tests=tests,
                command=command_str,
                cwd=str(self.workspace),
                stdout=result.stdout,
                stderr=result.stderr,
                metadata={
                    "return_code": result.returncode,
                    "preset": preset,
                    "target": target,
                },
            )

        except subprocess.TimeoutExpired:
            return TestRunResult(
                success=False,
                command=command_str,
                cwd=str(self.workspace),
                error=f"Test run timed out after {timeout} seconds",
                metadata={"timeout": timeout},
            )

        except FileNotFoundError:
            return TestRunResult(
                success=False,
                command=command_str,
                cwd=str(self.workspace),
                error="pytest not found. Install with: pip install pytest",
            )

        except Exception as e:
            return TestRunResult(
                success=False,
                command=command_str,
                cwd=str(self.workspace),
                error=str(e),
            )

    def _parse_pytest_output(self, output: str) -> tuple:
        """
        Parse pytest output to extract test results.

        Returns:
            Tuple of (tests, passed, failed, skipped, errors)
        """
        tests = []
        passed = 0
        failed = 0
        skipped = 0
        errors = 0

        lines = output.split("\n")

        for line in lines:
            line = line.strip()

            # Parse individual test results
            if "::" in line:
                if " PASSED" in line:
                    name = line.split(" PASSED")[0].strip()
                    tests.append(TestResult(name=name, outcome="passed"))
                    passed += 1
                elif " FAILED" in line:
                    name = line.split(" FAILED")[0].strip()
                    tests.append(TestResult(name=name, outcome="failed"))
                    failed += 1
                elif " SKIPPED" in line:
                    name = line.split(" SKIPPED")[0].strip()
                    tests.append(TestResult(name=name, outcome="skipped"))
                    skipped += 1
                elif " ERROR" in line:
                    name = line.split(" ERROR")[0].strip()
                    tests.append(TestResult(name=name, outcome="error"))
                    errors += 1

            # Parse summary line
            if "passed" in line.lower() and ("failed" in line.lower() or "error" in line.lower() or "skipped" in line.lower()):
                # Try to extract counts from summary like "5 passed, 2 failed"
                import re
                passed_match = re.search(r"(\d+) passed", line)
                failed_match = re.search(r"(\d+) failed", line)
                skipped_match = re.search(r"(\d+) skipped", line)
                error_match = re.search(r"(\d+) error", line)

                if passed_match:
                    passed = int(passed_match.group(1))
                if failed_match:
                    failed = int(failed_match.group(1))
                if skipped_match:
                    skipped = int(skipped_match.group(1))
                if error_match:
                    errors = int(error_match.group(1))

        return tests, passed, failed, skipped, errors

    def discover_tests(
        self,
        target: Optional[str] = None,
        pattern: str = "test_*.py",
    ) -> TestDiscoveryResult:
        """
        Discover tests without running them.

        Args:
            target: Directory or file to search
            pattern: File pattern for test files

        Returns:
            TestDiscoveryResult with discovered tests
        """
        cmd = ["python", "-m", "pytest", "--collect-only", "-q"]

        if target:
            cmd.append(target)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=60,
            )

            tests, test_files = self._parse_collect_output(result.stdout)

            return TestDiscoveryResult(
                success=result.returncode == 0,
                tests=tests,
                test_files=test_files,
                metadata={
                    "target": target,
                    "pattern": pattern,
                },
            )

        except subprocess.TimeoutExpired:
            return TestDiscoveryResult(
                success=False,
                error="Test discovery timed out",
            )

        except Exception as e:
            return TestDiscoveryResult(
                success=False,
                error=str(e),
            )

    def _parse_collect_output(self, output: str) -> tuple:
        """
        Parse pytest --collect-only output.

        Returns:
            Tuple of (tests, test_files)
        """
        tests = []
        test_files = set()

        for line in output.split("\n"):
            line = line.strip()
            if "::" in line and not line.startswith("="):
                # Parse test path like "tests/test_foo.py::TestClass::test_method"
                parts = line.split("::")
                if parts:
                    file_path = parts[0]
                    test_files.add(file_path)

                    tests.append(DiscoveredTest(
                        name=line,
                        file_path=file_path,
                    ))

        return tests, list(test_files)


# Convenience functions

def run_tests(
    target: Optional[str] = None,
    preset: Optional[str] = None,
    workspace: Optional[Path] = None,
    **kwargs,
) -> TestRunResult:
    """
    Run tests using pytest.

    Args:
        target: Test target
        preset: Preset configuration
        workspace: Repository root
        **kwargs: Additional arguments for TestRunner.run_tests

    Returns:
        TestRunResult with test outcomes
    """
    runner = TestRunner(workspace)
    return runner.run_tests(target, preset, **kwargs)


def discover_tests(
    target: Optional[str] = None,
    workspace: Optional[Path] = None,
    pattern: str = "test_*.py",
) -> TestDiscoveryResult:
    """
    Discover tests without running them.

    Args:
        target: Directory or file to search
        workspace: Repository root
        pattern: File pattern

    Returns:
        TestDiscoveryResult with discovered tests
    """
    runner = TestRunner(workspace)
    return runner.discover_tests(target, pattern)


def get_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get available test presets.

    Returns:
        Dict of preset names to configurations
    """
    return TEST_PRESETS.copy()
