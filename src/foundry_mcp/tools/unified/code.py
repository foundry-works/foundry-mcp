"""Unified code navigation tool.

This is a lightweight replacement for the legacy doc-query family (removed in
remove-docquery-rendering-docgen). It intentionally keeps the feature set small
and safe: repo-relative searches only, strict path allowlists, and conservative
result sizing.

The goal is to provide enough structure for LLM-assisted navigation without
reintroducing the full docquery surface.
"""

from __future__ import annotations

import ast
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.context import generate_correlation_id, get_correlation_id
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()


_DEFAULT_MAX_RESULTS = 80
_DEFAULT_MAX_FILES = 200
_DEFAULT_MAX_NODES = 60
_DEFAULT_DEPTH = 2
_MAX_DEPTH = 6


_LANGUAGE_EXTENSIONS: Dict[str, Tuple[str, ...]] = {
    "python": (".py",),
    "javascript": (".js", ".jsx"),
    "typescript": (".ts", ".tsx"),
    "go": (".go",),
    "java": (".java",),
}


def _request_id() -> str:
    return get_correlation_id() or generate_correlation_id(prefix="code")


def _metric(action: str) -> str:
    return f"unified_tools.code.{action.replace('-', '_')}"


def _validation_error(
    *, message: str, request_id: str, remediation: Optional[str] = None
) -> dict:
    return asdict(
        error_response(
            message,
            error_code=ErrorCode.VALIDATION_ERROR,
            error_type=ErrorType.VALIDATION,
            remediation=remediation,
            request_id=request_id,
        )
    )


def _workspace_root(config: ServerConfig, workspace: Optional[str]) -> Path:
    if workspace:
        return Path(workspace).expanduser().resolve()
    return Path.cwd().resolve()


def _normalize_allowlist(root: Path, allowlist: Optional[Sequence[str]]) -> List[Path]:
    if not allowlist:
        return [root]

    roots: List[Path] = []
    for entry in allowlist:
        candidate = (
            (root / entry).expanduser().resolve()
            if not Path(entry).is_absolute()
            else Path(entry).expanduser().resolve()
        )
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        roots.append(candidate)

    return roots or [root]


def _iter_candidate_files(
    allowlist: Iterable[Path],
    *,
    extensions: Tuple[str, ...],
    max_files: int,
) -> List[Path]:
    files: List[Path] = []
    for root in allowlist:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix in extensions:
                files.append(root)
        else:
            for path in root.rglob("*"):
                if len(files) >= max_files:
                    return files
                if not path.is_file():
                    continue
                if path.suffix in extensions:
                    files.append(path)
    return files


def _search_regex_in_files(
    files: Sequence[Path],
    *,
    pattern: re.Pattern[str],
    root: Path,
    max_results: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for file_path in files:
        if len(results) >= max_results:
            break
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                results.append(
                    {
                        "file": str(file_path.relative_to(root)),
                        "line": idx,
                        "snippet": line.strip()[:300],
                    }
                )
                if len(results) >= max_results:
                    break
    return results


def _parse_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(minimum, parsed), maximum)


def _called_symbol(node: ast.AST) -> Optional[str]:
    """Best-effort extraction of call symbol from AST."""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        # e.g. foo.bar() -> "foo.bar" (we don't resolve types)
        parts: List[str] = []
        cursor: Optional[ast.AST] = node
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name):
            parts.append(cursor.id)
        if not parts:
            return None
        return ".".join(reversed(parts))
    return None


class _PythonCallGraphVisitor(ast.NodeVisitor):
    def __init__(self, file_rel: str) -> None:
        self.file_rel = file_rel
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.name_index: Dict[str, List[str]] = defaultdict(list)
        self.edges: List[Tuple[str, str]] = []
        self._scope: List[str] = []

    def _qualname(self, name: str) -> str:
        if not self._scope:
            return name
        return ".".join(self._scope + [name])

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        qualname = self._qualname(node.name)
        node_id = f"{self.file_rel}:{node.lineno}:{qualname}"
        self.nodes[node_id] = {
            "id": node_id,
            "symbol": qualname,
            "file": self.file_rel,
            "line": node.lineno,
            "kind": "function" if not self._scope else "method",
        }
        self.name_index[node.name].append(node_id)
        self.name_index[qualname].append(node_id)

        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        # Treat async defs the same for call graph purposes.
        qualname = self._qualname(node.name)
        node_id = f"{self.file_rel}:{node.lineno}:{qualname}"
        self.nodes[node_id] = {
            "id": node_id,
            "symbol": qualname,
            "file": self.file_rel,
            "line": node.lineno,
            "kind": "function" if not self._scope else "method",
        }
        self.name_index[node.name].append(node_id)
        self.name_index[qualname].append(node_id)

        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        if not self._scope:
            return

        callee = _called_symbol(node.func)
        if not callee:
            return

        caller_qualname = ".".join(self._scope)
        caller_id = f"{self.file_rel}:{getattr(node, 'lineno', 0)}:{caller_qualname}"
        caller_candidates = self.name_index.get(caller_qualname)
        if caller_candidates:
            caller_id = caller_candidates[0]

        callee_candidates = self.name_index.get(callee)
        if callee_candidates:
            callee_id = callee_candidates[0]
        else:
            callee_id = f"external:{callee}"
            self.nodes.setdefault(
                callee_id,
                {
                    "id": callee_id,
                    "symbol": callee,
                    "file": None,
                    "line": None,
                    "kind": "external",
                },
            )

        self.edges.append((caller_id, callee_id))


def _build_python_call_graph(
    files: Sequence[Path], *, root: Path, max_nodes: int
) -> Tuple[
    Dict[str, Dict[str, Any]], Dict[str, List[str]], List[Tuple[str, str]], List[str]
]:
    warnings: List[str] = []
    nodes: Dict[str, Dict[str, Any]] = {}
    name_index: Dict[str, List[str]] = defaultdict(list)
    edges: List[Tuple[str, str]] = []

    for file_path in files:
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        rel = str(file_path.relative_to(root))
        visitor = _PythonCallGraphVisitor(rel)
        visitor.visit(tree)

        for node_id, payload in visitor.nodes.items():
            if len(nodes) >= max_nodes:
                warnings.append(
                    "Call graph node budget reached; narrow path_allowlist."
                )
                break
            nodes.setdefault(node_id, payload)

        for key, ids in visitor.name_index.items():
            for node_id in ids:
                if node_id in nodes:
                    name_index[key].append(node_id)

        if len(nodes) >= max_nodes:
            break
        edges.extend(visitor.edges)

    return nodes, name_index, edges, warnings


def _match_node_symbol(node_symbol: str, query: str) -> bool:
    query = query.strip()
    if not query:
        return False
    if query == node_symbol:
        return True
    if "." not in query and node_symbol.split(".")[-1] == query:
        return True
    return False


def _handle_find_class(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return _validation_error(
            message="symbol is required",
            request_id=request_id,
            remediation="Provide a class name",
        )

    language = payload.get("language", "python")
    if not isinstance(language, str):
        return _validation_error(
            message="language must be a string",
            request_id=request_id,
            remediation="Use 'python' or omit to default",
        )

    extensions = _LANGUAGE_EXTENSIONS.get(language.lower())
    if extensions is None:
        return _validation_error(
            message=f"Unsupported language: {language}",
            request_id=request_id,
            remediation=f"Use one of: {', '.join(sorted(_LANGUAGE_EXTENSIONS))}",
        )

    workspace = payload.get("workspace")
    if workspace is not None and not isinstance(workspace, str):
        return _validation_error(
            message="workspace must be a string",
            request_id=request_id,
            remediation="Provide an absolute workspace path",
        )

    allowlist = payload.get("path_allowlist")
    if allowlist is not None and not isinstance(allowlist, list):
        return _validation_error(
            message="path_allowlist must be a list of paths",
            request_id=request_id,
            remediation="Provide repo-relative paths like ['src']",
        )

    root = _workspace_root(config, workspace)
    allowed_roots = _normalize_allowlist(root, allowlist)

    max_results = int(payload.get("max_results", _DEFAULT_MAX_RESULTS))
    max_files = int(payload.get("max_files", _DEFAULT_MAX_FILES))

    files = _iter_candidate_files(
        allowed_roots, extensions=extensions, max_files=max_files
    )
    pattern = re.compile(rf"^\\s*class\\s+{re.escape(symbol.strip())}\\b")

    start = time.perf_counter()
    matches = _search_regex_in_files(
        files, pattern=pattern, root=root, max_results=max_results
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    _metrics.timer(_metric("find_class") + ".duration_ms", elapsed_ms)

    warnings: List[str] = []
    if len(files) >= max_files:
        warnings.append("File scan hit max_files limit; narrow path_allowlist.")
    if len(matches) >= max_results:
        warnings.append("Results truncated; narrow search or increase max_results.")

    return asdict(
        success_response(
            classes=[{"symbol": symbol.strip(), **m} for m in matches],
            language=language,
            scanned_files=len(files),
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            warnings=warnings or None,
        )
    )


def _handle_find_function(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return _validation_error(
            message="symbol is required",
            request_id=request_id,
            remediation="Provide a function name",
        )

    language = payload.get("language", "python")
    if not isinstance(language, str):
        return _validation_error(
            message="language must be a string",
            request_id=request_id,
            remediation="Use 'python' or omit to default",
        )

    extensions = _LANGUAGE_EXTENSIONS.get(language.lower())
    if extensions is None:
        return _validation_error(
            message=f"Unsupported language: {language}",
            request_id=request_id,
            remediation=f"Use one of: {', '.join(sorted(_LANGUAGE_EXTENSIONS))}",
        )

    root = _workspace_root(
        config,
        payload.get("workspace") if isinstance(payload.get("workspace"), str) else None,
    )
    allowed_roots = _normalize_allowlist(
        root,
        payload.get("path_allowlist")
        if isinstance(payload.get("path_allowlist"), list)
        else None,
    )

    max_results = int(payload.get("max_results", _DEFAULT_MAX_RESULTS))
    max_files = int(payload.get("max_files", _DEFAULT_MAX_FILES))

    files = _iter_candidate_files(
        allowed_roots, extensions=extensions, max_files=max_files
    )

    if language.lower() == "python":
        pattern = re.compile(rf"^\\s*def\\s+{re.escape(symbol.strip())}\\b")
    else:
        pattern = re.compile(rf"\\b{re.escape(symbol.strip())}\\s*\\(")

    start = time.perf_counter()
    matches = _search_regex_in_files(
        files, pattern=pattern, root=root, max_results=max_results
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    _metrics.timer(_metric("find_function") + ".duration_ms", elapsed_ms)

    warnings: List[str] = []
    if len(files) >= max_files:
        warnings.append("File scan hit max_files limit; narrow path_allowlist.")
    if len(matches) >= max_results:
        warnings.append("Results truncated; narrow search or increase max_results.")

    return asdict(
        success_response(
            functions=[{"symbol": symbol.strip(), **m} for m in matches],
            language=language,
            scanned_files=len(files),
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            warnings=warnings or None,
        )
    )


def _handle_call_graph(
    *, config: ServerConfig, payload: Dict[str, Any], action: str
) -> dict:
    request_id = _request_id()
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return _validation_error(
            message="symbol is required",
            request_id=request_id,
            remediation="Provide a symbol name",
        )

    language = payload.get("language", "python")
    if not isinstance(language, str):
        return _validation_error(
            message="language must be a string",
            request_id=request_id,
            remediation="Use 'python' or omit to default",
        )

    depth_raw = payload.get("depth", _DEFAULT_DEPTH)
    try:
        depth = int(depth_raw)
    except (TypeError, ValueError):
        return _validation_error(
            message="depth must be an integer",
            request_id=request_id,
            remediation=f"Provide an integer between 1 and {_MAX_DEPTH}",
        )
    depth = min(max(1, depth), _MAX_DEPTH)

    max_nodes_raw = payload.get("max_nodes", _DEFAULT_MAX_NODES)
    try:
        max_nodes = int(max_nodes_raw)
    except (TypeError, ValueError):
        return _validation_error(
            message="max_nodes must be an integer",
            request_id=request_id,
            remediation="Provide an integer between 1 and 200",
        )
    max_nodes = min(max(1, max_nodes), 200)

    if language.lower() != "python":
        # Fallback to occurrence-only indexing for non-Python code.
        response = _handle_occurrences(config=config, payload=payload, action=action)
        warnings = list(response.get("meta", {}).get("warnings", []) or [])
        if depth > 1:
            warnings.append("depth > 1 is only supported for python call graphs.")
        warnings.append(
            "Non-python call graphs are occurrence-based; results may be noisy."
        )
        if warnings:
            response.setdefault("meta", {})["warnings"] = warnings
        return response

    extensions = _LANGUAGE_EXTENSIONS["python"]

    root = _workspace_root(
        config,
        payload.get("workspace") if isinstance(payload.get("workspace"), str) else None,
    )
    allowed_roots = _normalize_allowlist(
        root,
        payload.get("path_allowlist")
        if isinstance(payload.get("path_allowlist"), list)
        else None,
    )

    max_files = _parse_int(
        payload.get("max_files", _DEFAULT_MAX_FILES),
        default=_DEFAULT_MAX_FILES,
        minimum=1,
        maximum=2000,
    )

    files = _iter_candidate_files(
        allowed_roots, extensions=extensions, max_files=max_files
    )

    start = time.perf_counter()
    nodes, _, edges, build_warnings = _build_python_call_graph(
        files, root=root, max_nodes=max_nodes
    )

    start_nodes: List[str] = []
    for node_id, node in nodes.items():
        if node.get("kind") == "external":
            continue
        if _match_node_symbol(str(node.get("symbol", "")), symbol):
            start_nodes.append(node_id)

    if not start_nodes:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)
        return asdict(
            success_response(
                symbol=symbol.strip(),
                language=language,
                nodes=[],
                edges=[],
                traces=[],
                scanned_files=len(files),
                request_id=request_id,
                telemetry={"duration_ms": round(elapsed_ms, 2)},
                warnings=[
                    "No matching python definitions found; returned empty graph."
                ],
            )
        )

    outgoing: Dict[str, set[str]] = defaultdict(set)
    incoming: Dict[str, set[str]] = defaultdict(set)
    for src, dst in edges:
        outgoing[src].add(dst)
        incoming[dst].add(src)

    visited: set[str] = set(start_nodes)
    frontier: deque[str] = deque(start_nodes)
    truncated = False

    for _ in range(depth):
        if not frontier:
            break
        level_count = len(frontier)
        for _ in range(level_count):
            current = frontier.popleft()
            neighbors = incoming[current] if action == "callers" else outgoing[current]
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                if len(visited) >= max_nodes:
                    truncated = True
                    break
                visited.add(neighbor)
                frontier.append(neighbor)
            if truncated:
                break
        if truncated:
            break

    filtered_edges = [
        {"source": src, "target": dst, "kind": "calls"}
        for src, dst in edges
        if src in visited and dst in visited
    ]
    filtered_nodes = [nodes[node_id] for node_id in visited if node_id in nodes]

    traces: List[List[str]] = []
    if action == "trace":
        max_paths = 25

        def _dfs(path: List[str], remaining: int) -> None:
            nonlocal truncated
            if truncated or len(traces) >= max_paths:
                truncated = True
                return
            if remaining == 0:
                traces.append(list(path))
                return
            current = path[-1]
            neighbors = sorted(outgoing.get(current, set()))
            if not neighbors:
                traces.append(list(path))
                return
            for neighbor in neighbors:
                if neighbor in path:
                    continue
                if len(visited) >= max_nodes:
                    truncated = True
                    return
                path.append(neighbor)
                _dfs(path, remaining - 1)
                path.pop()
                if truncated:
                    return

        for start_node in start_nodes:
            _dfs([start_node], depth)
            if truncated:
                break

    elapsed_ms = (time.perf_counter() - start) * 1000
    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)

    warnings = list(build_warnings)
    if truncated:
        warnings.append("Graph traversal truncated; reduce depth or path_allowlist.")

    return asdict(
        success_response(
            symbol=symbol.strip(),
            language=language,
            nodes=sorted(
                filtered_nodes, key=lambda n: (n.get("file") or "", n.get("line") or 0)
            ),
            edges=filtered_edges,
            traces=traces,
            scanned_files=len(files),
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            warnings=warnings or None,
        )
    )


def _handle_occurrences(
    *, config: ServerConfig, payload: Dict[str, Any], action: str
) -> dict:
    request_id = _request_id()
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return _validation_error(
            message="symbol is required",
            request_id=request_id,
            remediation="Provide a symbol name",
        )

    language = payload.get("language", "python")
    extensions = _LANGUAGE_EXTENSIONS.get(str(language).lower())
    if extensions is None:
        return _validation_error(
            message=f"Unsupported language: {language}",
            request_id=request_id,
            remediation=f"Use one of: {', '.join(sorted(_LANGUAGE_EXTENSIONS))}",
        )

    root = _workspace_root(
        config,
        payload.get("workspace") if isinstance(payload.get("workspace"), str) else None,
    )
    allowed_roots = _normalize_allowlist(
        root,
        payload.get("path_allowlist")
        if isinstance(payload.get("path_allowlist"), list)
        else None,
    )

    max_results = int(payload.get("max_results", _DEFAULT_MAX_RESULTS))
    max_files = int(payload.get("max_files", _DEFAULT_MAX_FILES))

    files = _iter_candidate_files(
        allowed_roots, extensions=extensions, max_files=max_files
    )
    pattern = re.compile(rf"\\b{re.escape(symbol.strip())}\\b")

    start = time.perf_counter()
    matches = _search_regex_in_files(
        files, pattern=pattern, root=root, max_results=max_results
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    _metrics.timer(_metric(action) + ".duration_ms", elapsed_ms)

    warnings: List[str] = []
    if len(files) >= max_files:
        warnings.append("File scan hit max_files limit; narrow path_allowlist.")
    if len(matches) >= max_results:
        warnings.append("Results truncated; narrow search or increase max_results.")

    nodes = [{"id": f"{m['file']}:{m['line']}", **m} for m in matches]

    return asdict(
        success_response(
            symbol=symbol.strip(),
            language=language,
            nodes=nodes,
            edges=[],
            scanned_files=len(files),
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            warnings=warnings or None,
        )
    )


def _handle_callers(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    return _handle_call_graph(config=config, payload=payload, action="callers")


def _handle_callees(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    return _handle_call_graph(config=config, payload=payload, action="callees")


def _handle_trace(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    return _handle_call_graph(config=config, payload=payload, action="trace")


def _handle_impact(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    request_id = _request_id()
    symbols = payload.get("symbols")
    if (
        not isinstance(symbols, list)
        or not symbols
        or not all(isinstance(s, str) and s.strip() for s in symbols)
    ):
        return _validation_error(
            message="symbols must be a non-empty list of strings",
            request_id=request_id,
            remediation="Provide symbols like ['Foo', 'bar']",
        )

    language = payload.get("language", "python")
    extensions = _LANGUAGE_EXTENSIONS.get(str(language).lower())
    if extensions is None:
        return _validation_error(
            message=f"Unsupported language: {language}",
            request_id=request_id,
            remediation=f"Use one of: {', '.join(sorted(_LANGUAGE_EXTENSIONS))}",
        )

    root = _workspace_root(
        config,
        payload.get("workspace") if isinstance(payload.get("workspace"), str) else None,
    )
    allowed_roots = _normalize_allowlist(
        root,
        payload.get("path_allowlist")
        if isinstance(payload.get("path_allowlist"), list)
        else None,
    )

    files = _iter_candidate_files(
        allowed_roots, extensions=extensions, max_files=_DEFAULT_MAX_FILES
    )

    start = time.perf_counter()
    touched_files: Dict[str, List[str]] = {}

    for symbol in symbols:
        pattern = re.compile(rf"\\b{re.escape(symbol.strip())}\\b")
        matches = _search_regex_in_files(
            files, pattern=pattern, root=root, max_results=_DEFAULT_MAX_RESULTS
        )
        touched_files[symbol.strip()] = sorted({m["file"] for m in matches})

    elapsed_ms = (time.perf_counter() - start) * 1000

    file_set = {
        file for files_for_symbol in touched_files.values() for file in files_for_symbol
    }
    count = len(file_set)
    if count == 0:
        severity = "none"
    elif count < 5:
        severity = "minor"
    elif count < 20:
        severity = "moderate"
    else:
        severity = "major"

    _metrics.timer(_metric("impact") + ".duration_ms", elapsed_ms)

    return asdict(
        success_response(
            summary={
                "severity": severity,
                "touched_files": count,
            },
            details={"files_by_symbol": touched_files},
            scanned_files=len(files),
            request_id=request_id,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
        )
    )


_ACTION_SUMMARY = {
    "find-class": "Locate class definitions by symbol.",
    "find-function": "Locate function definitions by symbol.",
    "callers": "Return inbound references (python call graph when available).",
    "callees": "Return outbound references (python call graph when available).",
    "trace": "Return call traces (python call graph when available).",
    "impact": "Summarize files touched by symbol occurrences.",
}


def _build_router() -> ActionRouter:
    return ActionRouter(
        tool_name="code",
        actions=[
            ActionDefinition(
                name="find-class",
                handler=_handle_find_class,
                summary=_ACTION_SUMMARY["find-class"],
                aliases=("find_class",),
            ),
            ActionDefinition(
                name="find-function",
                handler=_handle_find_function,
                summary=_ACTION_SUMMARY["find-function"],
                aliases=("find_function",),
            ),
            ActionDefinition(
                name="callers",
                handler=_handle_callers,
                summary=_ACTION_SUMMARY["callers"],
            ),
            ActionDefinition(
                name="callees",
                handler=_handle_callees,
                summary=_ACTION_SUMMARY["callees"],
            ),
            ActionDefinition(
                name="trace", handler=_handle_trace, summary=_ACTION_SUMMARY["trace"]
            ),
            ActionDefinition(
                name="impact", handler=_handle_impact, summary=_ACTION_SUMMARY["impact"]
            ),
        ],
    )


_CODE_ROUTER = _build_router()


def _dispatch_code_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _CODE_ROUTER.dispatch(action, config=config, payload=payload)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        request_id = _request_id()
        return asdict(
            error_response(
                f"Unsupported code action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                request_id=request_id,
            )
        )


def register_unified_code_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated code tool."""

    @canonical_tool(mcp, canonical_name="code")
    @mcp_tool(tool_name="code", emit_metrics=True, audit=False)
    def code(
        action: str,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        language: str = "python",
        workspace: Optional[str] = None,
        path_allowlist: Optional[List[str]] = None,
        max_results: int = _DEFAULT_MAX_RESULTS,
        max_files: int = _DEFAULT_MAX_FILES,
        max_nodes: int = _DEFAULT_MAX_NODES,
        depth: int = _DEFAULT_DEPTH,
    ) -> dict:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "symbols": symbols,
            "language": language,
            "workspace": workspace,
            "path_allowlist": path_allowlist,
            "max_results": max_results,
            "max_files": max_files,
            "max_nodes": max_nodes,
            "depth": depth,
        }
        return _dispatch_code_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified code tool")


__all__ = [
    "register_unified_code_tool",
]
