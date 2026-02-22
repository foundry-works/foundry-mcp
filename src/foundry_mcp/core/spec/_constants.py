"""
Shared constants for Foundry spec operations.

These constants are used across spec sub-modules. Import from here
rather than defining inline to avoid duplication.
"""

# Valid templates and categories for spec creation
TEMPLATES = ("empty",)
TEMPLATE_DESCRIPTIONS = {
    "empty": "Blank spec with no phases - use phase-add-bulk to add structure",
}
CATEGORIES = ("investigation", "implementation", "refactoring", "decision", "research")

# Valid complexity levels for task nodes
COMPLEXITY_LEVELS = ("low", "medium", "high")

# Valid verification types for verify nodes
# - run-tests: Automated tests via mcp__foundry-mcp__test-run
# - fidelity: Implementation-vs-spec comparison via mcp__foundry-mcp__spec-review-fidelity
# - manual: Manual verification steps
VERIFICATION_TYPES = ("run-tests", "fidelity", "manual")

# Default retention policy for versioned backups
DEFAULT_MAX_BACKUPS = 10

# Default pagination settings for backup listing
DEFAULT_BACKUP_PAGE_SIZE = 50
MAX_BACKUP_PAGE_SIZE = 100

# Default settings for diff operations
DEFAULT_DIFF_MAX_RESULTS = 100

# Valid frontmatter keys that can be updated
# Note: assumptions and revision_history have dedicated functions
FRONTMATTER_KEYS = (
    "title",
    "description",
    "mission",
    "objectives",
    "status",
    "progress_percentage",
    "current_phase",
)
