"""
Batch operation response schemas (Pydantic) for MCP tool responses.

Provides type-safe Pydantic models for batch operation data:
dependency graphs, batch task contexts, prepare/start/complete responses.
All models are conditionally defined behind a PYDANTIC_AVAILABLE guard.
"""

from typing import Any, Dict, Optional

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


if PYDANTIC_AVAILABLE:

    class DependencyNode(BaseModel):
        """A node in the dependency graph representing a task."""

        id: str = Field(..., description="Task identifier")
        title: str = Field(default="", description="Task title")
        status: str = Field(default="", description="Task status")
        file_path: Optional[str] = Field(
            default=None, description="File path associated with the task"
        )
        is_target: bool = Field(
            default=False, description="Whether this is a target task in the batch"
        )

    class DependencyEdge(BaseModel):
        """An edge in the dependency graph representing a dependency relationship."""

        from_id: str = Field(..., alias="from", description="Source task ID")
        to_id: str = Field(..., alias="to", description="Target task ID")
        edge_type: str = Field(
            default="blocks", alias="type", description="Type of dependency (blocks)"
        )

        model_config = {"populate_by_name": True}

    class DependencyGraph(BaseModel):
        """Dependency graph structure for batch tasks.

        Contains nodes (tasks) and edges (dependency relationships) to visualize
        task dependencies for parallel execution planning.
        """

        nodes: list[DependencyNode] = Field(
            default_factory=list, description="Task nodes in the graph"
        )
        edges: list[DependencyEdge] = Field(
            default_factory=list, description="Dependency edges between tasks"
        )

    class BatchTaskDependencies(BaseModel):
        """Dependency status for a task in a batch."""

        task_id: str = Field(..., description="Task identifier")
        can_start: bool = Field(
            default=True, description="Whether the task can be started"
        )
        blocked_by: list[str] = Field(
            default_factory=list, description="IDs of tasks blocking this one"
        )
        soft_depends: list[str] = Field(
            default_factory=list, description="IDs of soft dependencies"
        )
        blocks: list[str] = Field(
            default_factory=list, description="IDs of tasks this one blocks"
        )

    class BatchTaskContext(BaseModel):
        """Context for a single task in a batch prepare response.

        Contains all information needed to execute a task in parallel with others.
        """

        task_id: str = Field(..., description="Unique task identifier")
        title: str = Field(default="", description="Task title")
        task_type: str = Field(
            default="task", alias="type", description="Task type (task, subtask, verify)"
        )
        status: str = Field(default="pending", description="Current task status")
        metadata: Dict[str, Any] = Field(
            default_factory=dict,
            description="Task metadata including file_path, description, etc.",
        )
        dependencies: Optional[BatchTaskDependencies] = Field(
            default=None, description="Dependency status for the task"
        )
        phase: Optional[Dict[str, Any]] = Field(
            default=None, description="Phase context (id, title, progress)"
        )
        parent: Optional[Dict[str, Any]] = Field(
            default=None, description="Parent task context (id, title, position_label)"
        )

        model_config = {"populate_by_name": True}

    class StaleTaskInfo(BaseModel):
        """Information about a stale in_progress task."""

        task_id: str = Field(..., description="Task identifier")
        title: str = Field(default="", description="Task title")

    class BatchPrepareResponse(BaseModel):
        """Response schema for prepare_batch_context operation.

        Contains independent tasks that can be executed in parallel along with
        context, dependency information, and warnings.
        """

        tasks: list[BatchTaskContext] = Field(
            default_factory=list, description="Tasks ready for parallel execution"
        )
        task_count: int = Field(default=0, description="Number of tasks in the batch")
        spec_complete: bool = Field(
            default=False, description="Whether the spec has no remaining tasks"
        )
        all_blocked: bool = Field(
            default=False, description="Whether all remaining tasks are blocked"
        )
        warnings: list[str] = Field(
            default_factory=list, description="Non-fatal warnings about the batch"
        )
        stale_tasks: list[StaleTaskInfo] = Field(
            default_factory=list, description="In-progress tasks exceeding time threshold"
        )
        dependency_graph: DependencyGraph = Field(
            default_factory=DependencyGraph,
            description="Dependency graph for batch tasks",
        )
        token_estimate: Optional[int] = Field(
            default=None, description="Estimated token count for the batch context"
        )

    class BatchStartResponse(BaseModel):
        """Response schema for start_batch operation.

        Confirms which tasks were atomically started and when.
        """

        started: list[str] = Field(
            default_factory=list, description="IDs of tasks successfully started"
        )
        started_count: int = Field(
            default=0, description="Number of tasks started"
        )
        started_at: Optional[str] = Field(
            default=None, description="ISO timestamp when tasks were started"
        )
        errors: Optional[list[str]] = Field(
            default=None, description="Validation errors if operation failed"
        )

    class BatchTaskCompletion(BaseModel):
        """Input schema for a single task completion in complete_batch.

        Used to specify outcome for each task being completed.
        """

        task_id: str = Field(..., description="Task identifier to complete")
        success: bool = Field(
            ..., description="True if task succeeded, False if failed"
        )
        completion_note: str = Field(
            default="", description="Note describing what was accomplished or why it failed"
        )

    class BatchTaskResult(BaseModel):
        """Result for a single task in the complete_batch response."""

        status: str = Field(
            ..., description="Result status: completed, failed, skipped, error"
        )
        completed_at: Optional[str] = Field(
            default=None, description="ISO timestamp when completed (if successful)"
        )
        failed_at: Optional[str] = Field(
            default=None, description="ISO timestamp when failed (if unsuccessful)"
        )
        retry_count: Optional[int] = Field(
            default=None, description="Updated retry count (if failed)"
        )
        error: Optional[str] = Field(
            default=None, description="Error message (if status is error or skipped)"
        )

    class BatchCompleteResponse(BaseModel):
        """Response schema for complete_batch operation.

        Contains per-task results and summary counts for the batch completion.
        """

        results: Dict[str, BatchTaskResult] = Field(
            default_factory=dict,
            description="Per-task results keyed by task_id",
        )
        completed_count: int = Field(
            default=0, description="Number of tasks successfully completed"
        )
        failed_count: int = Field(
            default=0, description="Number of tasks that failed"
        )
        total_processed: int = Field(
            default=0, description="Total number of completions processed"
        )

    # Export Pydantic models
    __all_pydantic__ = [
        "DependencyNode",
        "DependencyEdge",
        "DependencyGraph",
        "BatchTaskDependencies",
        "BatchTaskContext",
        "StaleTaskInfo",
        "BatchPrepareResponse",
        "BatchStartResponse",
        "BatchTaskCompletion",
        "BatchTaskResult",
        "BatchCompleteResponse",
    ]

else:
    # Pydantic not available - provide None placeholders
    DependencyNode = None  # type: ignore[misc,assignment]
    DependencyEdge = None  # type: ignore[misc,assignment]
    DependencyGraph = None  # type: ignore[misc,assignment]
    BatchTaskDependencies = None  # type: ignore[misc,assignment]
    BatchTaskContext = None  # type: ignore[misc,assignment]
    StaleTaskInfo = None  # type: ignore[misc,assignment]
    BatchPrepareResponse = None  # type: ignore[misc,assignment]
    BatchStartResponse = None  # type: ignore[misc,assignment]
    BatchTaskCompletion = None  # type: ignore[misc,assignment]
    BatchTaskResult = None  # type: ignore[misc,assignment]
    BatchCompleteResponse = None  # type: ignore[misc,assignment]
    __all_pydantic__ = []
