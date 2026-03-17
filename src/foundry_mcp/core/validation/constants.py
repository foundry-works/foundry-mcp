"""Validation constants for SDD spec files."""

STATUS_FIELDS = {"pending", "in_progress", "completed", "blocked"}
VALID_NODE_TYPES = {"spec", "phase", "group", "task", "subtask", "verify"}
VALID_STATUSES = {"pending", "in_progress", "completed", "blocked", "archived"}
VALID_TASK_CATEGORIES = {
    "investigation",
    "implementation",
    "refactoring",
    "decision",
}
VALID_VERIFICATION_TYPES = {"run-tests", "fidelity", "manual"}

# Common field name typos/alternatives
FIELD_NAME_SUGGESTIONS = {
    "category": "task_category",
    "type": "node type or verification_type",
    "desc": "description",
    "details": "description",
}

# Valid verification results
VERIFICATION_RESULTS = ("PASSED", "FAILED", "PARTIAL")
