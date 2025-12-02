"""Doc-query commands for SDD CLI.

Provides commands for querying codebase documentation including:
- Finding classes and functions
- Tracing call graphs
- Analyzing code impact
- Documentation statistics
"""

import json
import time
from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    SLOW_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)

logger = get_cli_logger()


@click.group("doc")
def doc_group() -> None:
    """Documentation query commands."""
    pass


@doc_group.command("find-class")
@click.argument("name")
@click.option(
    "--exact/--fuzzy",
    default=True,
    help="Use exact match (default) or substring match.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of results.",
)
@click.pass_context
@cli_command("doc-find-class")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Class search timed out")
def doc_find_class_cmd(
    ctx: click.Context,
    name: str,
    exact: bool,
    limit: int,
) -> None:
    """Find a class by name in codebase documentation.

    NAME is the class name to search for.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.codebase_docs import get_codebase_docs

        docs = get_codebase_docs(cli_ctx.specs_dir)
        if docs is None:
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        # Search for classes
        matches = []
        classes = docs.get("classes", {})

        for class_name, class_info in classes.items():
            if exact:
                if class_name == name:
                    matches.append({
                        "name": class_name,
                        **class_info,
                    })
            else:
                if name.lower() in class_name.lower():
                    matches.append({
                        "name": class_name,
                        **class_info,
                    })

            if len(matches) >= limit:
                break

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success({
            "query": name,
            "exact": exact,
            "matches": matches,
            "total_count": len(matches),
            "truncated": len(matches) >= limit,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except ImportError:
        emit_error(
            "Codebase docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation and reinstall if needed",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Find class failed: {e}",
            code="FIND_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={"query": name},
        )


@doc_group.command("find-function")
@click.argument("name")
@click.option(
    "--exact/--fuzzy",
    default=True,
    help="Use exact match (default) or substring match.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of results.",
)
@click.pass_context
@cli_command("doc-find-function")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Function search timed out")
def doc_find_function_cmd(
    ctx: click.Context,
    name: str,
    exact: bool,
    limit: int,
) -> None:
    """Find a function by name in codebase documentation.

    NAME is the function name to search for.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.codebase_docs import get_codebase_docs

        docs = get_codebase_docs(cli_ctx.specs_dir)
        if docs is None:
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        # Search for functions
        matches = []
        functions = docs.get("functions", {})

        for func_name, func_info in functions.items():
            if exact:
                if func_name == name:
                    matches.append({
                        "name": func_name,
                        **func_info,
                    })
            else:
                if name.lower() in func_name.lower():
                    matches.append({
                        "name": func_name,
                        **func_info,
                    })

            if len(matches) >= limit:
                break

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success({
            "query": name,
            "exact": exact,
            "matches": matches,
            "total_count": len(matches),
            "truncated": len(matches) >= limit,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except ImportError:
        emit_error(
            "Codebase docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation and reinstall if needed",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Find function failed: {e}",
            code="FIND_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={"query": name},
        )


@doc_group.command("trace-calls")
@click.argument("function_name")
@click.option(
    "--direction",
    type=click.Choice(["callers", "callees", "both"]),
    default="both",
    help="Direction to trace (callers, callees, or both).",
)
@click.option(
    "--max-depth",
    type=int,
    default=3,
    help="Maximum traversal depth.",
)
@click.pass_context
@cli_command("doc-trace-calls")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Call trace timed out")
def doc_trace_calls_cmd(
    ctx: click.Context,
    function_name: str,
    direction: str,
    max_depth: int,
) -> None:
    """Trace function calls in the call graph.

    FUNCTION_NAME is the function to trace from.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.codebase_docs import get_codebase_docs

        docs = get_codebase_docs(cli_ctx.specs_dir)
        if docs is None:
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        call_graph = docs.get("call_graph", {})
        result = {
            "function": function_name,
            "direction": direction,
            "max_depth": max_depth,
        }

        # Trace callers (who calls this function)
        if direction in ("callers", "both"):
            callers = _trace_direction(call_graph, function_name, "callers", max_depth)
            result["callers"] = callers

        # Trace callees (what this function calls)
        if direction in ("callees", "both"):
            callees = _trace_direction(call_graph, function_name, "callees", max_depth)
            result["callees"] = callees

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success({
            **result,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except ImportError:
        emit_error(
            "Codebase docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation and reinstall if needed",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Trace calls failed: {e}",
            code="TRACE_FAILED",
            error_type="internal",
            remediation="Check that the function exists in codebase documentation",
            details={"function": function_name},
        )


def _trace_direction(call_graph: dict, start: str, direction: str, max_depth: int) -> list:
    """Trace call relationships in a given direction."""
    visited = set()
    result = []

    def _trace(name: str, depth: int):
        if depth > max_depth or name in visited:
            return
        visited.add(name)

        node = call_graph.get(name, {})
        relations = node.get(direction, [])

        for rel in relations:
            result.append({
                "from": name if direction == "callees" else rel,
                "to": rel if direction == "callees" else name,
                "depth": depth,
            })
            _trace(rel, depth + 1)

    _trace(start, 1)
    return result


@doc_group.command("impact")
@click.argument("target")
@click.option(
    "--type",
    "target_type",
    type=click.Choice(["class", "function", "auto"]),
    default="auto",
    help="Type of target (auto-detected by default).",
)
@click.option(
    "--max-depth",
    type=int,
    default=3,
    help="Maximum depth for impact propagation.",
)
@click.pass_context
@cli_command("doc-impact")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Impact analysis timed out")
def doc_impact_cmd(
    ctx: click.Context,
    target: str,
    target_type: str,
    max_depth: int,
) -> None:
    """Analyze impact of changing a class or function.

    TARGET is the name of the class or function to analyze.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.codebase_docs import get_codebase_docs

        docs = get_codebase_docs(cli_ctx.specs_dir)
        if docs is None:
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        # Auto-detect type if needed
        detected_type = target_type
        if target_type == "auto":
            if target in docs.get("classes", {}):
                detected_type = "class"
            elif target in docs.get("functions", {}):
                detected_type = "function"
            else:
                detected_type = "unknown"

        # Get direct impacts
        call_graph = docs.get("call_graph", {})
        callers = _trace_direction(call_graph, target, "callers", max_depth)

        # Collect affected files
        affected_files = set()
        for caller in callers:
            caller_name = caller.get("from", "")
            func_info = docs.get("functions", {}).get(caller_name, {})
            if "file" in func_info:
                affected_files.add(func_info["file"])

        # Calculate impact score (simple heuristic)
        direct_impacts = len([c for c in callers if c.get("depth") == 1])
        indirect_impacts = len([c for c in callers if c.get("depth", 0) > 1])
        impact_score = direct_impacts * 3 + indirect_impacts

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success({
            "target": target,
            "target_type": detected_type,
            "direct_impacts": direct_impacts,
            "indirect_impacts": indirect_impacts,
            "impact_score": impact_score,
            "affected_files": sorted(affected_files),
            "callers": callers,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except ImportError:
        emit_error(
            "Codebase docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation and reinstall if needed",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Impact analysis failed: {e}",
            code="IMPACT_FAILED",
            error_type="internal",
            remediation="Check that the target exists in codebase documentation",
            details={"target": target},
        )


@doc_group.command("stats")
@click.pass_context
@cli_command("doc-stats")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Documentation stats timed out")
def doc_stats_cmd(ctx: click.Context) -> None:
    """Get documentation statistics."""
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.codebase_docs import get_codebase_docs

        docs = get_codebase_docs(cli_ctx.specs_dir)
        if docs is None:
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        # Calculate stats
        classes = docs.get("classes", {})
        functions = docs.get("functions", {})
        files = docs.get("files", {})
        call_graph = docs.get("call_graph", {})

        # Count dependencies
        total_deps = 0
        for node in call_graph.values():
            total_deps += len(node.get("callers", []))
            total_deps += len(node.get("callees", []))

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success({
            "class_count": len(classes),
            "function_count": len(functions),
            "file_count": len(files),
            "call_graph_nodes": len(call_graph),
            "total_dependencies": total_deps // 2,  # Each dep counted twice
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except ImportError:
        emit_error(
            "Codebase docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation and reinstall if needed",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Get stats failed: {e}",
            code="STATS_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={},
        )


# Top-level aliases
@click.command("find-class")
@click.argument("name")
@click.option("--exact/--fuzzy", default=True)
@click.option("--limit", type=int, default=100)
@click.pass_context
@handle_keyboard_interrupt()
def find_class_alias_cmd(
    ctx: click.Context,
    name: str,
    exact: bool,
    limit: int,
) -> None:
    """Find a class by name (alias for doc find-class)."""
    ctx.invoke(doc_find_class_cmd, name=name, exact=exact, limit=limit)


@click.command("find-function")
@click.argument("name")
@click.option("--exact/--fuzzy", default=True)
@click.option("--limit", type=int, default=100)
@click.pass_context
@handle_keyboard_interrupt()
def find_function_alias_cmd(
    ctx: click.Context,
    name: str,
    exact: bool,
    limit: int,
) -> None:
    """Find a function by name (alias for doc find-function)."""
    ctx.invoke(doc_find_function_cmd, name=name, exact=exact, limit=limit)


@doc_group.command("scope")
@click.argument("target")
@click.option(
    "--view",
    type=click.Choice(["plan", "implement", "trace", "debug"]),
    default="plan",
    help="Context view mode: plan (lightweight), implement (detailed), trace (call graph), debug (all info).",
)
@click.pass_context
@cli_command("doc-scope")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Scope query timed out")
def doc_scope_cmd(
    ctx: click.Context,
    target: str,
    view: str,
) -> None:
    """Get scoped context for a file.

    TARGET is the file path to get scope for.

    View modes:
    - plan: Lightweight context for planning (class/function names, signatures)
    - implement: Detailed context (includes docstrings, parameters, calls)
    - trace: Call graph and dependencies
    - debug: All available information
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None)
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        # Map view to mode
        mode = "implement" if view in ("implement", "debug") else "plan"
        result = query.get_scope(target, mode=mode)

        if not result.success:
            emit_error(
                result.error or "Scope query failed",
                code="SCOPE_FAILED",
                error_type="internal",
                remediation="Check that the file exists in documentation",
                details={"target": target},
            )
            return

        scope_data = result.results[0] if result.results else {}

        # Add trace info if requested
        if view in ("trace", "debug"):
            # Add call graph info for functions
            trace_info = []
            for func in scope_data.get("functions", []):
                func_name = func.get("name", "")
                trace_result = query.trace_calls(func_name, direction="both", max_depth=2)
                if trace_result.success and trace_result.results:
                    trace_info.append({
                        "function": func_name,
                        "calls": [{"caller": e.caller, "callee": e.callee} for e in trace_result.results[:10]],
                    })
            scope_data["trace"] = trace_info

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success({
            "target": target,
            "view": view,
            "scope": scope_data,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Scope query failed: {e}",
            code="SCOPE_FAILED",
            error_type="internal",
            remediation="Check that the target file exists in documentation",
            details={"target": target},
        )


@doc_group.command("dependencies")
@click.argument("module")
@click.option(
    "--reverse",
    is_flag=True,
    default=False,
    help="Show modules that depend on this module instead.",
)
@click.pass_context
@cli_command("doc-dependencies")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Dependencies query timed out")
def doc_dependencies_cmd(
    ctx: click.Context,
    module: str,
    reverse: bool,
) -> None:
    """Get dependencies of a module.

    MODULE is the module path (e.g., foundry_mcp.core.docs).

    Use --reverse to find modules that depend on the specified module.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None)
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        if reverse:
            result = query.get_reverse_dependencies(module)
        else:
            result = query.get_dependencies(module)

        if not result.success:
            emit_error(
                result.error or "Dependencies query failed",
                code="DEPS_FAILED",
                error_type="internal",
                remediation="Check that the module exists in documentation",
                details={"module": module},
            )
            return

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success({
            "module": module,
            "direction": "reverse" if reverse else "forward",
            "dependencies": result.results,
            "count": len(result.results),
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Dependencies query failed: {e}",
            code="DEPS_FAILED",
            error_type="internal",
            remediation="Check that the module exists in documentation",
            details={"module": module},
        )


@doc_group.command("describe-module")
@click.argument("file_path")
@click.pass_context
@cli_command("doc-describe-module")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Describe module timed out")
def doc_describe_module_cmd(
    ctx: click.Context,
    file_path: str,
) -> None:
    """Describe a module (file) showing its classes and functions.

    FILE_PATH is the path to the file to describe.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None)
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        # Get classes and functions in the file
        classes_result = query.find_classes_in_file(file_path)
        functions_result = query.find_functions_in_file(file_path)

        classes = []
        for r in classes_result.results:
            cls_data = {
                "name": r.name,
                "line": r.line_number,
                "bases": r.data.get("bases", []),
                "methods": r.data.get("methods", []),
            }
            if r.data.get("docstring"):
                cls_data["docstring"] = r.data["docstring"][:200]
            classes.append(cls_data)

        functions = []
        for r in functions_result.results:
            func_data = {
                "name": r.name,
                "line": r.line_number,
                "signature": r.data.get("signature", ""),
            }
            if r.data.get("docstring"):
                func_data["docstring"] = r.data["docstring"][:200]
            functions.append(func_data)

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success({
            "file_path": file_path,
            "classes": classes,
            "functions": functions,
            "class_count": len(classes),
            "function_count": len(functions),
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"Describe module failed: {e}",
            code="DESCRIBE_FAILED",
            error_type="internal",
            remediation="Check that the file exists in documentation",
            details={"file_path": file_path},
        )


@doc_group.command("list-modules")
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of modules to return.",
)
@click.pass_context
@cli_command("doc-list-modules")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "List modules timed out")
def doc_list_modules_cmd(
    ctx: click.Context,
    limit: int,
) -> None:
    """List all modules (files) in the codebase documentation."""
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None)
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        # Collect unique files from classes and functions
        files = set()
        for cls in query._classes_by_name.values():
            if cls.get("file"):
                files.add(cls["file"])
        for func in query._functions_by_name.values():
            if func.get("file"):
                files.add(func["file"])

        # Build module list with counts
        modules = []
        sorted_files = sorted(files)[:limit]
        for file_path in sorted_files:
            class_count = len(query._classes_by_file.get(file_path, []))
            function_count = len(query._functions_by_file.get(file_path, []))
            modules.append({
                "file_path": file_path,
                "class_count": class_count,
                "function_count": function_count,
            })

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success({
            "modules": modules,
            "count": len(modules),
            "total_files": len(files),
            "truncated": len(files) > limit,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"List modules failed: {e}",
            code="LIST_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={},
        )


@doc_group.command("list-functions")
@click.option(
    "--file",
    "file_path",
    default=None,
    help="Filter to functions in a specific file.",
)
@click.option(
    "--limit",
    type=int,
    default=100,
    help="Maximum number of functions to return.",
)
@click.pass_context
@cli_command("doc-list-functions")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "List functions timed out")
def doc_list_functions_cmd(
    ctx: click.Context,
    file_path: Optional[str],
    limit: int,
) -> None:
    """List functions in the codebase documentation.

    Use --file to filter to a specific file.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)

    try:
        from foundry_mcp.core.docs import DocsQuery

        query = DocsQuery(workspace=cli_ctx.specs_dir.parent if cli_ctx.specs_dir else None)
        if not query.load():
            emit_error(
                "Codebase documentation not available",
                code="DOCS_NOT_FOUND",
                error_type="not_found",
                remediation="Run documentation generation first with: sdd llm-doc generate",
                details={"hint": "Run documentation generation first"},
            )
            return

        functions = []

        if file_path:
            # Get functions from specific file
            result = query.find_functions_in_file(file_path)
            for r in result.results[:limit]:
                functions.append({
                    "name": r.name,
                    "file_path": r.file_path,
                    "line": r.line_number,
                    "signature": r.data.get("signature", ""),
                })
        else:
            # List all functions
            count = 0
            for func_name, func in sorted(query._functions_by_name.items()):
                if count >= limit:
                    break
                functions.append({
                    "name": func_name,
                    "file_path": func.get("file"),
                    "line": func.get("line"),
                    "signature": func.get("signature", ""),
                })
                count += 1

        total_count = len(query._functions_by_name)
        if file_path:
            total_count = len(query._functions_by_file.get(file_path, []))

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_success({
            "functions": functions,
            "count": len(functions),
            "total_count": total_count,
            "truncated": len(functions) < total_count,
            "file_filter": file_path,
            "telemetry": {"duration_ms": round(duration_ms, 2)},
        })

    except ImportError:
        emit_error(
            "Docs module not available",
            code="MODULE_NOT_FOUND",
            error_type="internal",
            remediation="Check foundry_mcp installation",
            details={"hint": "Check foundry_mcp installation"},
        )
    except Exception as e:
        emit_error(
            f"List functions failed: {e}",
            code="LIST_FAILED",
            error_type="internal",
            remediation="Check that codebase documentation is valid",
            details={"file_path": file_path},
        )
