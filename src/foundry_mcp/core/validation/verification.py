"""Verification management for SDD spec files."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from foundry_mcp.core.security import MAX_STRING_LENGTH
from foundry_mcp.core.validation.constants import VERIFICATION_RESULTS


def add_verification(
    spec_data: Dict[str, Any],
    verify_id: str,
    result: str,
    command: Optional[str] = None,
    output: Optional[str] = None,
    issues: Optional[str] = None,
    notes: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Add verification result to a verify node.

    Records verification results including test outcomes, command output,
    and issues found during verification.

    Args:
        spec_data: The loaded spec data dict (modified in place).
        verify_id: Verification node ID (e.g., verify-1-1).
        result: Verification result (PASSED, FAILED, PARTIAL).
        command: Optional command that was run for verification.
        output: Optional command output or test results.
        issues: Optional issues found during verification.
        notes: Optional additional notes about the verification.

    Returns:
        Tuple of (success, error_message).
        On success: (True, None)
        On failure: (False, "error message")
    """
    # Validate result
    result_upper = result.upper().strip()
    if result_upper not in VERIFICATION_RESULTS:
        return (
            False,
            f"Invalid result '{result}'. Must be one of: {', '.join(VERIFICATION_RESULTS)}",
        )

    # Get hierarchy
    hierarchy = spec_data.get("hierarchy")
    if not hierarchy or not isinstance(hierarchy, dict):
        return False, "Invalid spec data: missing or invalid hierarchy"

    # Find the verify node
    node = hierarchy.get(verify_id)
    if node is None:
        return False, f"Verification node '{verify_id}' not found"

    # Validate node type
    node_type = node.get("type")
    if node_type != "verify":
        return False, f"Node '{verify_id}' is type '{node_type}', expected 'verify'"

    # Get or create metadata
    metadata = node.get("metadata")
    if metadata is None:
        metadata = {}
        node["metadata"] = metadata

    # Build verification result entry
    verification_entry: Dict[str, Any] = {
        "result": result_upper,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    if command:
        verification_entry["command"] = command.strip()

    if output:
        # Truncate output if very long
        max_output_len = MAX_STRING_LENGTH
        output_text = output.strip()
        if len(output_text) > max_output_len:
            output_text = output_text[:max_output_len] + "\n... (truncated)"
        verification_entry["output"] = output_text

    if issues:
        verification_entry["issues"] = issues.strip()

    if notes:
        verification_entry["notes"] = notes.strip()

    # Add to verification history (keep last N entries)
    verification_history = metadata.get("verification_history", [])
    if not isinstance(verification_history, list):
        verification_history = []

    verification_history.append(verification_entry)

    # Keep only last 10 entries
    if len(verification_history) > 10:
        verification_history = verification_history[-10:]

    metadata["verification_history"] = verification_history

    # Update latest result fields for quick access
    metadata["last_result"] = result_upper
    metadata["last_verified_at"] = verification_entry["timestamp"]

    return True, None


def execute_verification(
    spec_data: Dict[str, Any],
    verify_id: str,
    record: bool = False,
    timeout: int = 300,
    cwd: Optional[str] = None,
    command_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute verification command and capture results.

    Runs the verification command defined in a verify node's metadata
    and captures output, exit code, and result status.

    Args:
        spec_data: The loaded spec data dict.
        verify_id: Verification node ID (e.g., verify-1-1).
        record: If True, automatically record result to spec using add_verification().
        timeout: Command timeout in seconds (default: 300).
        cwd: Working directory for command execution (default: current directory).
        command_override: Optional command to use as fallback when the verify node
            has no embedded command in its metadata.

    Returns:
        Dict with execution results:
        - success: Whether execution completed (not result status)
        - spec_id: The specification ID
        - verify_id: The verification ID
        - result: Execution result (PASSED, FAILED, PARTIAL)
        - command: Command that was executed
        - output: Combined stdout/stderr output
        - exit_code: Command exit code
        - recorded: Whether result was recorded to spec
        - error: Error message if execution failed

    Example:
        >>> result = execute_verification(spec_data, "verify-1-1", record=True)
        >>> if result["success"]:
        ...     print(f"Verification {result['result']}: {result['exit_code']}")
    """
    import subprocess

    response: Dict[str, Any] = {
        "success": False,
        "spec_id": spec_data.get("spec_id", "unknown"),
        "verify_id": verify_id,
        "result": None,
        "command": None,
        "output": None,
        "exit_code": None,
        "recorded": False,
        "error": None,
    }

    # Get hierarchy
    hierarchy = spec_data.get("hierarchy")
    if not hierarchy or not isinstance(hierarchy, dict):
        response["error"] = "Invalid spec data: missing or invalid hierarchy"
        return response

    # Find the verify node
    node = hierarchy.get(verify_id)
    if node is None:
        response["error"] = f"Verification node '{verify_id}' not found"
        return response

    # Validate node type
    node_type = node.get("type")
    if node_type != "verify":
        response["error"] = (
            f"Node '{verify_id}' is type '{node_type}', expected 'verify'"
        )
        return response

    # Get command from metadata, falling back to caller-provided override
    metadata = node.get("metadata", {})
    command = metadata.get("command") or command_override

    if not command:
        response["error"] = (
            f"No command defined in verify node '{verify_id}' metadata "
            f"and no command parameter provided as fallback"
        )
        return response

    response["command"] = command

    # Execute the command
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )

        exit_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        # Combine output
        output_parts = []
        if stdout.strip():
            output_parts.append(stdout.strip())
        if stderr.strip():
            output_parts.append(f"[stderr]\n{stderr.strip()}")
        output = "\n".join(output_parts) if output_parts else "(no output)"

        # Truncate if too long
        if len(output) > MAX_STRING_LENGTH:
            output = output[:MAX_STRING_LENGTH] + "\n... (truncated)"

        response["exit_code"] = exit_code
        response["output"] = output

        # Determine result based on exit code
        if exit_code == 0:
            result = "PASSED"
        else:
            result = "FAILED"

        response["result"] = result
        response["success"] = True

        # Optionally record result to spec
        if record:
            record_success, record_error = add_verification(
                spec_data=spec_data,
                verify_id=verify_id,
                result=result,
                command=command,
                output=output,
            )
            if record_success:
                response["recorded"] = True
            else:
                response["recorded"] = False
                # Don't fail the whole operation, just note the recording failed
                if response.get("error"):
                    response["error"] += f"; Recording failed: {record_error}"
                else:
                    response["error"] = f"Recording failed: {record_error}"

    except subprocess.TimeoutExpired:
        response["error"] = f"Command timed out after {timeout} seconds"
        response["result"] = "FAILED"
        response["exit_code"] = -1
        response["output"] = f"Command timed out after {timeout} seconds"

    except subprocess.SubprocessError as e:
        response["error"] = f"Command execution failed: {e}"
        response["result"] = "FAILED"

    except Exception as e:
        response["error"] = f"Unexpected error: {e}"
        response["result"] = "FAILED"

    return response


def format_verification_summary(
    verification_data: Dict[str, Any] | List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Format verification results into a human-readable summary.

    Processes verification results (from execute_verification or JSON input)
    and produces a structured summary with counts and formatted text.

    Args:
        verification_data: Either:
            - A single verification result dict (from execute_verification)
            - A list of verification result dicts
            - A dict with "verifications" key containing a list

    Returns:
        Dict with formatted summary:
        - summary: Human-readable summary text
        - total_verifications: Total number of verifications
        - passed: Number of passed verifications
        - failed: Number of failed verifications
        - partial: Number of partial verifications
        - results: List of individual result summaries

    Example:
        >>> results = [
        ...     execute_verification(spec_data, "verify-1"),
        ...     execute_verification(spec_data, "verify-2"),
        ... ]
        >>> summary = format_verification_summary(results)
        >>> print(summary["summary"])
    """
    # Normalize input to a list of verification results
    verifications: List[Dict[str, Any]] = []

    if isinstance(verification_data, list):
        verifications = verification_data
    elif isinstance(verification_data, dict):
        if "verifications" in verification_data:
            verifications = verification_data.get("verifications", [])
        else:
            # Single verification result
            verifications = [verification_data]

    # Count results by type
    passed = 0
    failed = 0
    partial = 0
    results: List[Dict[str, Any]] = []

    for v in verifications:
        if not isinstance(v, dict):
            continue

        result = (v.get("result") or "").upper()
        verify_id = v.get("verify_id", "unknown")
        command = v.get("command", "")
        output = v.get("output", "")
        error = v.get("error")

        # Count by result type
        if result == "PASSED":
            passed += 1
            status_icon = "\u2713"
        elif result == "FAILED":
            failed += 1
            status_icon = "\u2717"
        elif result == "PARTIAL":
            partial += 1
            status_icon = "\u25d0"
        else:
            status_icon = "?"

        # Build individual result summary
        result_entry: Dict[str, Any] = {
            "verify_id": verify_id,
            "result": result or "UNKNOWN",
            "status_icon": status_icon,
            "command": command,
        }

        if error:
            result_entry["error"] = error

        # Truncate output for summary
        if output:
            output_preview = output[:200].strip()
            if len(output) > 200:
                output_preview += "..."
            result_entry["output_preview"] = output_preview

        results.append(result_entry)

    # Calculate totals
    total = len(results)

    # Build summary text
    summary_lines = []
    summary_lines.append(f"Verification Summary: {total} total")
    summary_lines.append(f"  \u2713 Passed:  {passed}")
    summary_lines.append(f"  \u2717 Failed:  {failed}")
    if partial > 0:
        summary_lines.append(f"  \u25d0 Partial: {partial}")
    summary_lines.append("")

    # Add individual results
    if results:
        summary_lines.append("Results:")
        for r in results:
            icon = r["status_icon"]
            vid = r["verify_id"]
            res = r["result"]
            cmd = r.get("command", "")

            line = f"  {icon} {vid}: {res}"
            if cmd:
                # Truncate command for display
                cmd_display = cmd[:50]
                if len(cmd) > 50:
                    cmd_display += "..."
                line += f" ({cmd_display})"

            summary_lines.append(line)

            if r.get("error"):
                summary_lines.append(f"      Error: {r['error']}")

    summary_text = "\n".join(summary_lines)

    return {
        "summary": summary_text,
        "total_verifications": total,
        "passed": passed,
        "failed": failed,
        "partial": partial,
        "results": results,
    }
