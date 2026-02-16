"""Validation constants for SDD spec files."""

STATUS_FIELDS = {"pending", "in_progress", "completed", "blocked"}
VALID_NODE_TYPES = {"spec", "phase", "group", "task", "subtask", "verify", "research"}
VALID_STATUSES = {"pending", "in_progress", "completed", "blocked", "archived"}
VALID_TASK_CATEGORIES = {
    "investigation",
    "implementation",
    "refactoring",
    "decision",
    "research",
}
VALID_VERIFICATION_TYPES = {"run-tests", "fidelity", "manual"}

# Research node constants
VALID_RESEARCH_TYPES = {"chat", "consensus", "thinkdeep", "ideate", "deep-research"}
VALID_RESEARCH_RESULTS = {"completed", "inconclusive", "blocked", "cancelled"}
RESEARCH_BLOCKING_MODES = {"none", "soft", "hard"}

# Common field name typos/alternatives
FIELD_NAME_SUGGESTIONS = {
    "category": "task_category",
    "type": "node type or verification_type",
    "desc": "description",
    "details": "description",
}

# Valid verification results
VERIFICATION_RESULTS = ("PASSED", "FAILED", "PARTIAL")
