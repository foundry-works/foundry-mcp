"""LLM documentation generation commands for SDD CLI.

Provides commands for AI-powered documentation generation including:
- Generating comprehensive documentation from code
- Managing LLM generation cache
- Checking LLM status
"""

import json
import subprocess
import time
from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    SLOW_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)

logger = get_cli_logger()

# Default timeout for LLM doc generation (can be long)
DOCGEN_TIMEOUT = 600  # 10 minutes


@click.group("llm-doc")
def llm_doc_group() -> None:
    """LLM-powered documentation generation commands."""
    pass


@llm_doc_group.command("generate")
@click.argument("directory")
@click.option(
    "--output-dir",
    help="Output directory for documentation (default: ./docs).",
)
@click.option(
    "--name",
    help="Project name (default: directory name).",
)
@click.option(
    "--description",
    help="Project description for documentation context.",
)
@click.option(
    "--batch-size",
    type=int,
    default=3,
    help="Number of shards to process per batch.",
)
@click.option(
    "--use-cache/--no-cache",
    default=True,
    help="Enable persistent caching of parse results.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from previous interrupted generation.",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear the cache before generating documentation.",
)
@click.pass_context
@cli_command("llm-doc-generate")
@handle_keyboard_interrupt()
@with_sync_timeout(DOCGEN_TIMEOUT, "Documentation generation timed out")
def llm_doc_generate_cmd(
    ctx: click.Context,
    directory: str,
    output_dir: Optional[str],
    name: Optional[str],
    description: Optional[str],
    batch_size: int,
    use_cache: bool,
    resume: bool,
    clear_cache: bool,
) -> None:
    """Generate LLM-powered documentation for a project.

    DIRECTORY is the project directory to document.

    Uses AI to analyze code and generate rich, context-aware
    documentation with explanations and architectural insights.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    # Check LLM status first
    llm_status = _get_llm_status()
    if not llm_status.get("configured"):
        emit_error(
            "LLM not configured",
            code="LLM_NOT_CONFIGURED",
            error_type="validation",
            remediation="Set ANTHROPIC_API_KEY or configure LLM provider",
            details={
                "hint": "Set ANTHROPIC_API_KEY or configure LLM provider",
                "llm_status": llm_status,
            },
        )

    # Build command
    cmd = ["sdd", "llm-doc-gen", directory, "--json"]

    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    if name:
        cmd.extend(["--name", name])
    if description:
        cmd.extend(["--description", description])
    cmd.extend(["--batch-size", str(batch_size)])

    if not use_cache:
        cmd.append("--no-cache")
    if resume:
        cmd.append("--resume")
    if clear_cache:
        cmd.append("--clear-cache")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DOCGEN_TIMEOUT,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Generation failed"
            emit_error(
                f"Documentation generation failed: {error_msg}",
                code="GENERATION_FAILED",
                error_type="internal",
                remediation="Check LLM configuration and ensure the directory contains valid source files",
                details={
                    "directory": directory,
                    "exit_code": result.returncode,
                    "llm_status": llm_status,
                },
            )

        # Parse output
        try:
            gen_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            gen_data = {"raw_output": result.stdout}

        emit_success({
            "directory": directory,
            "llm_status": llm_status,
            **gen_data,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except subprocess.TimeoutExpired:
        emit_error(
            f"Documentation generation timed out after {DOCGEN_TIMEOUT}s",
            code="TIMEOUT",
            error_type="internal",
            remediation="Try using --batch-size to process smaller batches or use --resume to continue later",
            details={"directory": directory, "timeout_seconds": DOCGEN_TIMEOUT},
        )
    except FileNotFoundError:
        emit_error(
            "SDD CLI not found",
            code="CLI_NOT_FOUND",
            error_type="internal",
            remediation="Ensure 'sdd' is installed and in PATH",
            details={"hint": "Ensure 'sdd' is installed and in PATH"},
        )


@llm_doc_group.command("status")
@click.pass_context
@cli_command("llm-doc-status")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "LLM status check timed out")
def llm_doc_status_cmd(ctx: click.Context) -> None:
    """Check LLM configuration status."""
    start_time = time.perf_counter()

    llm_status = _get_llm_status()
    duration_ms = (time.perf_counter() - start_time) * 1000

    recommendations = []
    if not llm_status.get("configured"):
        recommendations.append("Set ANTHROPIC_API_KEY environment variable")
        recommendations.append("Or configure alternative LLM provider")

    emit_success({
        **llm_status,
        "recommendations": recommendations,
        "telemetry": {"duration_ms": round(duration_ms, 2)},
    })


@llm_doc_group.command("cache")
@click.option(
    "--action",
    type=click.Choice(["info", "clear"]),
    default="info",
    help="Cache operation to perform.",
)
@click.option(
    "--spec-id",
    help="Optional spec ID filter for clear operation.",
)
@click.pass_context
@cli_command("llm-doc-cache")
@handle_keyboard_interrupt()
@with_sync_timeout(30, "Cache operation timed out")
def llm_doc_cache_cmd(
    ctx: click.Context,
    action: str,
    spec_id: Optional[str],
) -> None:
    """Manage LLM documentation cache.

    Actions:
      info  - Show cache statistics
      clear - Remove cached entries
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    # Build command
    cmd = ["sdd", "cache", action, "--json"]

    if spec_id:
        cmd.extend(["--spec-id", spec_id])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        if result.returncode != 0:
            # Fall back to basic cache info
            emit_success({
                "action": action,
                "status": "cache_unavailable",
                "message": "Cache management not available",
                "telemetry": {"duration_ms": round(duration_ms, 2)},
            })
            return

        # Parse output
        try:
            cache_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            cache_data = {"raw_output": result.stdout}

        emit_success({
            "action": action,
            **cache_data,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except subprocess.TimeoutExpired:
        emit_error(
            "Cache operation timed out",
            code="TIMEOUT",
            error_type="internal",
            remediation="Try again or check system resources",
            details={"action": action},
        )
    except FileNotFoundError:
        # Provide basic cache info without CLI
        emit_success({
            "action": action,
            "status": "cli_unavailable",
            "message": "SDD CLI not available for cache management",
            "telemetry": {"duration_ms": round((time.perf_counter() - start_time) * 1000, 2)},
        })


def _get_llm_status() -> dict:
    """Get LLM configuration status."""
    try:
        from foundry_mcp.core.llm_config import get_llm_config

        config = get_llm_config()
        return {
            "configured": config.get_api_key() is not None,
            "provider": config.provider.value,
            "model": config.get_model(),
        }
    except ImportError:
        return {"configured": False, "error": "LLM config not available"}
    except Exception as e:
        return {"configured": False, "error": str(e)}


# Top-level alias
@click.command("generate-docs")
@click.argument("directory")
@click.option("--output-dir", help="Output directory.")
@click.option("--name", help="Project name.")
@click.option("--resume", is_flag=True, help="Resume interrupted generation.")
@click.pass_context
@cli_command("generate-docs-alias")
@handle_keyboard_interrupt()
@with_sync_timeout(DOCGEN_TIMEOUT, "Documentation generation timed out")
def generate_docs_alias_cmd(
    ctx: click.Context,
    directory: str,
    output_dir: Optional[str],
    name: Optional[str],
    resume: bool,
) -> None:
    """Generate AI-powered documentation (alias for llm-doc generate)."""
    ctx.invoke(
        llm_doc_generate_cmd,
        directory=directory,
        output_dir=output_dir,
        name=name,
        resume=resume,
    )
