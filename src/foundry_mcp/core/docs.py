"""
Documentation query operations for foundry-mcp.
Provides functions for querying codebase documentation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# Schema version for compatibility tracking
SCHEMA_VERSION = "1.0.0"


# Data structures

@dataclass
class QueryResult:
    """
    Result of a documentation query.
    """
    entity_type: str  # class, function, module, dependency
    name: str
    data: Dict[str, Any]
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    relevance_score: float = 1.0


@dataclass
class CallGraphEntry:
    """
    Entry in a call graph.
    """
    caller: str
    callee: str
    caller_file: Optional[str] = None
    callee_file: Optional[str] = None
    call_count: int = 1


@dataclass
class ImpactResult:
    """
    Result of impact analysis for a change.
    """
    target: str
    target_type: str  # class, function, module
    direct_impacts: List[str] = field(default_factory=list)
    indirect_impacts: List[str] = field(default_factory=list)
    impact_score: float = 0.0
    affected_files: List[str] = field(default_factory=list)


@dataclass
class DocsQueryResponse:
    """
    Standard response wrapper for documentation queries.
    """
    success: bool
    schema_version: str = SCHEMA_VERSION
    timestamp: str = ""
    query_type: str = ""
    results: List[Any] = field(default_factory=list)
    count: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.count = len(self.results)


# Main query class

class DocsQuery:
    """
    Query interface for codebase documentation.

    Provides functions for finding classes, functions, tracing calls,
    and analyzing change impact.
    """

    def __init__(self, docs_path: Optional[Path] = None, workspace: Optional[Path] = None):
        """
        Initialize documentation query.

        Args:
            docs_path: Path to codebase.json or docs directory
            workspace: Repository root (defaults to current directory)
        """
        self.workspace = workspace or Path.cwd()
        self.docs_path = self._resolve_docs_path(docs_path)
        self.data: Optional[Dict[str, Any]] = None
        self._loaded = False

        # Indexes for fast lookups
        self._classes_by_name: Dict[str, Dict[str, Any]] = {}
        self._functions_by_name: Dict[str, Dict[str, Any]] = {}
        self._classes_by_file: Dict[str, List[Dict[str, Any]]] = {}
        self._functions_by_file: Dict[str, List[Dict[str, Any]]] = {}
        self._callers_index: Dict[str, List[str]] = {}
        self._callees_index: Dict[str, List[str]] = {}

    def _resolve_docs_path(self, docs_path: Optional[Path]) -> Path:
        """Resolve the documentation path."""
        if docs_path is None:
            # Search common locations
            search_paths = [
                self.workspace / "docs" / "codebase.json",
                self.workspace / "documentation" / "codebase.json",
                self.workspace / ".docs" / "codebase.json",
            ]
            for path in search_paths:
                if path.exists():
                    return path
            return self.workspace / "docs" / "codebase.json"

        docs_path = Path(docs_path)
        if docs_path.is_dir():
            return docs_path / "codebase.json"
        return docs_path

    def load(self) -> bool:
        """
        Load documentation data.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.docs_path.exists():
                return False

            with open(self.docs_path, "r") as f:
                self.data = json.load(f)

            self._build_indexes()
            self._loaded = True
            return True

        except (OSError, json.JSONDecodeError):
            return False

    def is_loaded(self) -> bool:
        """Check if documentation is loaded."""
        return self._loaded

    def _build_indexes(self) -> None:
        """Build lookup indexes from loaded data."""
        if not self.data:
            return

        # Index classes
        for cls in self.data.get("classes", []):
            name = cls.get("name", "")
            if name:
                self._classes_by_name[name] = cls
                file_path = cls.get("file", "")
                if file_path:
                    self._classes_by_file.setdefault(file_path, []).append(cls)

        # Index functions and build call graph
        for func in self.data.get("functions", []):
            name = func.get("name", "")
            if name:
                self._functions_by_name[name] = func
                file_path = func.get("file", "")
                if file_path:
                    self._functions_by_file.setdefault(file_path, []).append(func)

                # Build caller/callee indexes
                for callee in func.get("calls", []):
                    self._callees_index.setdefault(name, []).append(callee)
                    self._callers_index.setdefault(callee, []).append(name)

    # Class queries

    def find_class(
        self,
        name: str,
        exact: bool = True,
    ) -> DocsQueryResponse:
        """
        Find a class by name.

        Args:
            name: Class name to search for
            exact: If True, exact match; if False, substring match

        Returns:
            DocsQueryResponse with matching classes
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="find_class",
                error="Documentation not loaded",
            )

        results = []

        if exact:
            cls = self._classes_by_name.get(name)
            if cls:
                results.append(QueryResult(
                    entity_type="class",
                    name=name,
                    data=cls,
                    file_path=cls.get("file"),
                    line_number=cls.get("line"),
                ))
        else:
            for cls_name, cls in self._classes_by_name.items():
                if name.lower() in cls_name.lower():
                    results.append(QueryResult(
                        entity_type="class",
                        name=cls_name,
                        data=cls,
                        file_path=cls.get("file"),
                        line_number=cls.get("line"),
                    ))

        return DocsQueryResponse(
            success=True,
            query_type="find_class",
            results=results,
        )

    def find_classes_in_file(self, file_path: str) -> DocsQueryResponse:
        """
        Find all classes in a file.

        Args:
            file_path: Path to the file

        Returns:
            DocsQueryResponse with classes in the file
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="find_classes_in_file",
                error="Documentation not loaded",
            )

        classes = self._classes_by_file.get(file_path, [])
        results = [
            QueryResult(
                entity_type="class",
                name=cls.get("name", ""),
                data=cls,
                file_path=file_path,
                line_number=cls.get("line"),
            )
            for cls in classes
        ]

        return DocsQueryResponse(
            success=True,
            query_type="find_classes_in_file",
            results=results,
            metadata={"file_path": file_path},
        )

    # Function queries

    def find_function(
        self,
        name: str,
        exact: bool = True,
    ) -> DocsQueryResponse:
        """
        Find a function by name.

        Args:
            name: Function name to search for
            exact: If True, exact match; if False, substring match

        Returns:
            DocsQueryResponse with matching functions
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="find_function",
                error="Documentation not loaded",
            )

        results = []

        if exact:
            func = self._functions_by_name.get(name)
            if func:
                results.append(QueryResult(
                    entity_type="function",
                    name=name,
                    data=func,
                    file_path=func.get("file"),
                    line_number=func.get("line"),
                ))
        else:
            for func_name, func in self._functions_by_name.items():
                if name.lower() in func_name.lower():
                    results.append(QueryResult(
                        entity_type="function",
                        name=func_name,
                        data=func,
                        file_path=func.get("file"),
                        line_number=func.get("line"),
                    ))

        return DocsQueryResponse(
            success=True,
            query_type="find_function",
            results=results,
        )

    def find_functions_in_file(self, file_path: str) -> DocsQueryResponse:
        """
        Find all functions in a file.

        Args:
            file_path: Path to the file

        Returns:
            DocsQueryResponse with functions in the file
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="find_functions_in_file",
                error="Documentation not loaded",
            )

        functions = self._functions_by_file.get(file_path, [])
        results = [
            QueryResult(
                entity_type="function",
                name=func.get("name", ""),
                data=func,
                file_path=file_path,
                line_number=func.get("line"),
            )
            for func in functions
        ]

        return DocsQueryResponse(
            success=True,
            query_type="find_functions_in_file",
            results=results,
            metadata={"file_path": file_path},
        )

    # Call graph queries

    def trace_calls(
        self,
        function_name: str,
        direction: str = "both",
        max_depth: int = 3,
    ) -> DocsQueryResponse:
        """
        Trace function calls (call graph).

        Args:
            function_name: Function to trace from
            direction: "callers", "callees", or "both"
            max_depth: Maximum traversal depth

        Returns:
            DocsQueryResponse with call graph entries
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="trace_calls",
                error="Documentation not loaded",
            )

        entries: List[CallGraphEntry] = []
        visited: Set[str] = set()

        def trace_callers(name: str, depth: int):
            if depth > max_depth or name in visited:
                return
            visited.add(name)

            for caller in self._callers_index.get(name, []):
                caller_data = self._functions_by_name.get(caller, {})
                callee_data = self._functions_by_name.get(name, {})
                entries.append(CallGraphEntry(
                    caller=caller,
                    callee=name,
                    caller_file=caller_data.get("file"),
                    callee_file=callee_data.get("file"),
                ))
                trace_callers(caller, depth + 1)

        def trace_callees(name: str, depth: int):
            if depth > max_depth or name in visited:
                return
            visited.add(name)

            for callee in self._callees_index.get(name, []):
                caller_data = self._functions_by_name.get(name, {})
                callee_data = self._functions_by_name.get(callee, {})
                entries.append(CallGraphEntry(
                    caller=name,
                    callee=callee,
                    caller_file=caller_data.get("file"),
                    callee_file=callee_data.get("file"),
                ))
                trace_callees(callee, depth + 1)

        if direction in ("callers", "both"):
            visited.clear()
            trace_callers(function_name, 0)

        if direction in ("callees", "both"):
            visited.clear()
            trace_callees(function_name, 0)

        return DocsQueryResponse(
            success=True,
            query_type="trace_calls",
            results=entries,
            metadata={
                "function": function_name,
                "direction": direction,
                "max_depth": max_depth,
            },
        )

    def get_callers(self, function_name: str) -> DocsQueryResponse:
        """
        Get functions that call the specified function.

        Args:
            function_name: Function to find callers for

        Returns:
            DocsQueryResponse with caller functions
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="get_callers",
                error="Documentation not loaded",
            )

        callers = self._callers_index.get(function_name, [])
        results = [
            QueryResult(
                entity_type="function",
                name=caller,
                data=self._functions_by_name.get(caller, {}),
                file_path=self._functions_by_name.get(caller, {}).get("file"),
            )
            for caller in callers
        ]

        return DocsQueryResponse(
            success=True,
            query_type="get_callers",
            results=results,
            metadata={"function": function_name},
        )

    def get_callees(self, function_name: str) -> DocsQueryResponse:
        """
        Get functions called by the specified function.

        Args:
            function_name: Function to find callees for

        Returns:
            DocsQueryResponse with callee functions
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="get_callees",
                error="Documentation not loaded",
            )

        callees = self._callees_index.get(function_name, [])
        results = [
            QueryResult(
                entity_type="function",
                name=callee,
                data=self._functions_by_name.get(callee, {}),
                file_path=self._functions_by_name.get(callee, {}).get("file"),
            )
            for callee in callees
        ]

        return DocsQueryResponse(
            success=True,
            query_type="get_callees",
            results=results,
            metadata={"function": function_name},
        )

    # Impact analysis

    def impact_analysis(
        self,
        target: str,
        target_type: str = "auto",
        max_depth: int = 3,
    ) -> DocsQueryResponse:
        """
        Analyze impact of changing a class or function.

        Args:
            target: Name of class or function to analyze
            target_type: "class", "function", or "auto" (detect from name)
            max_depth: Maximum depth for impact propagation

        Returns:
            DocsQueryResponse with impact analysis
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="impact_analysis",
                error="Documentation not loaded",
            )

        # Auto-detect target type
        if target_type == "auto":
            if target in self._classes_by_name:
                target_type = "class"
            elif target in self._functions_by_name:
                target_type = "function"
            else:
                return DocsQueryResponse(
                    success=False,
                    query_type="impact_analysis",
                    error=f"Target not found: {target}",
                )

        direct_impacts: List[str] = []
        indirect_impacts: List[str] = []
        affected_files: Set[str] = set()

        if target_type == "function":
            # Direct impacts are callers
            direct_impacts = self._callers_index.get(target, [])

            # Get files of callers
            for caller in direct_impacts:
                func = self._functions_by_name.get(caller, {})
                if func.get("file"):
                    affected_files.add(func["file"])

            # Indirect impacts are callers of callers
            visited = set(direct_impacts)
            for caller in direct_impacts:
                indirect_callers = self._callers_index.get(caller, [])
                for ic in indirect_callers:
                    if ic not in visited and ic != target:
                        indirect_impacts.append(ic)
                        visited.add(ic)
                        func = self._functions_by_name.get(ic, {})
                        if func.get("file"):
                            affected_files.add(func["file"])

        elif target_type == "class":
            cls = self._classes_by_name.get(target, {})
            cls_file = cls.get("file", "")

            # Find functions that reference this class (simplified)
            for func_name, func in self._functions_by_name.items():
                # Check if function is in same file or references class
                if func.get("file") == cls_file:
                    direct_impacts.append(func_name)
                    affected_files.add(cls_file)

            # Find subclasses
            for other_name, other_cls in self._classes_by_name.items():
                if target in other_cls.get("bases", []):
                    direct_impacts.append(other_name)
                    if other_cls.get("file"):
                        affected_files.add(other_cls["file"])

        # Calculate impact score
        impact_score = len(direct_impacts) + (len(indirect_impacts) * 0.5)

        result = ImpactResult(
            target=target,
            target_type=target_type,
            direct_impacts=direct_impacts,
            indirect_impacts=indirect_impacts,
            impact_score=impact_score,
            affected_files=list(affected_files),
        )

        return DocsQueryResponse(
            success=True,
            query_type="impact_analysis",
            results=[result],
            metadata={
                "target": target,
                "target_type": target_type,
                "max_depth": max_depth,
            },
        )

    # Dependency queries

    def get_dependencies(self, module: str) -> DocsQueryResponse:
        """
        Get dependencies of a module.

        Args:
            module: Module path to get dependencies for

        Returns:
            DocsQueryResponse with dependencies
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="get_dependencies",
                error="Documentation not loaded",
            )

        deps = self.data.get("dependencies", {}).get(module, [])

        return DocsQueryResponse(
            success=True,
            query_type="get_dependencies",
            results=deps,
            metadata={"module": module},
        )

    def get_reverse_dependencies(self, module: str) -> DocsQueryResponse:
        """
        Get modules that depend on the specified module.

        Args:
            module: Module path to find reverse dependencies for

        Returns:
            DocsQueryResponse with reverse dependencies
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="get_reverse_dependencies",
                error="Documentation not loaded",
            )

        reverse_deps = []
        dependencies = self.data.get("dependencies", {})
        for mod, deps in dependencies.items():
            if module in deps:
                reverse_deps.append(mod)

        return DocsQueryResponse(
            success=True,
            query_type="get_reverse_dependencies",
            results=reverse_deps,
            metadata={"module": module},
        )

    # Metadata

    def get_metadata(self) -> DocsQueryResponse:
        """
        Get documentation metadata.

        Returns:
            DocsQueryResponse with metadata
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="get_metadata",
                error="Documentation not loaded",
            )

        metadata = self.data.get("metadata", {})

        return DocsQueryResponse(
            success=True,
            query_type="get_metadata",
            results=[metadata],
            metadata={
                "docs_path": str(self.docs_path),
                "workspace": str(self.workspace),
            },
        )

    def get_stats(self) -> DocsQueryResponse:
        """
        Get documentation statistics.

        Returns:
            DocsQueryResponse with stats
        """
        if not self._loaded:
            return DocsQueryResponse(
                success=False,
                query_type="get_stats",
                error="Documentation not loaded",
            )

        stats = {
            "total_classes": len(self._classes_by_name),
            "total_functions": len(self._functions_by_name),
            "total_files": len(set(self._classes_by_file.keys()) | set(self._functions_by_file.keys())),
            "total_dependencies": sum(len(deps) for deps in self.data.get("dependencies", {}).values()),
        }

        return DocsQueryResponse(
            success=True,
            query_type="get_stats",
            results=[stats],
        )


# Convenience functions

def load_docs(
    docs_path: Optional[Path] = None,
    workspace: Optional[Path] = None,
) -> Optional[DocsQuery]:
    """
    Load documentation and return query interface.

    Args:
        docs_path: Path to codebase.json or docs directory
        workspace: Repository root

    Returns:
        DocsQuery instance if successful, None otherwise
    """
    query = DocsQuery(docs_path, workspace)
    if query.load():
        return query
    return None


def find_class(
    name: str,
    docs_path: Optional[Path] = None,
    workspace: Optional[Path] = None,
) -> DocsQueryResponse:
    """
    Find a class by name.

    Args:
        name: Class name to search for
        docs_path: Path to documentation
        workspace: Repository root

    Returns:
        DocsQueryResponse with matching classes
    """
    query = DocsQuery(docs_path, workspace)
    if not query.load():
        return DocsQueryResponse(
            success=False,
            query_type="find_class",
            error="Failed to load documentation",
        )
    return query.find_class(name)


def find_function(
    name: str,
    docs_path: Optional[Path] = None,
    workspace: Optional[Path] = None,
) -> DocsQueryResponse:
    """
    Find a function by name.

    Args:
        name: Function name to search for
        docs_path: Path to documentation
        workspace: Repository root

    Returns:
        DocsQueryResponse with matching functions
    """
    query = DocsQuery(docs_path, workspace)
    if not query.load():
        return DocsQueryResponse(
            success=False,
            query_type="find_function",
            error="Failed to load documentation",
        )
    return query.find_function(name)


def trace_calls(
    function_name: str,
    direction: str = "both",
    max_depth: int = 3,
    docs_path: Optional[Path] = None,
    workspace: Optional[Path] = None,
) -> DocsQueryResponse:
    """
    Trace function calls.

    Args:
        function_name: Function to trace from
        direction: "callers", "callees", or "both"
        max_depth: Maximum traversal depth
        docs_path: Path to documentation
        workspace: Repository root

    Returns:
        DocsQueryResponse with call graph
    """
    query = DocsQuery(docs_path, workspace)
    if not query.load():
        return DocsQueryResponse(
            success=False,
            query_type="trace_calls",
            error="Failed to load documentation",
        )
    return query.trace_calls(function_name, direction, max_depth)


def impact_analysis(
    target: str,
    target_type: str = "auto",
    max_depth: int = 3,
    docs_path: Optional[Path] = None,
    workspace: Optional[Path] = None,
) -> DocsQueryResponse:
    """
    Analyze impact of changing a class or function.

    Args:
        target: Name of class or function
        target_type: "class", "function", or "auto"
        max_depth: Maximum depth for impact propagation
        docs_path: Path to documentation
        workspace: Repository root

    Returns:
        DocsQueryResponse with impact analysis
    """
    query = DocsQuery(docs_path, workspace)
    if not query.load():
        return DocsQueryResponse(
            success=False,
            query_type="impact_analysis",
            error="Failed to load documentation",
        )
    return query.impact_analysis(target, target_type, max_depth)
