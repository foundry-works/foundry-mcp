"""Phase mixins for DeepResearchWorkflow.

Each mixin contributes a disjoint set of methods implementing one workflow phase.
They are combined via multiple inheritance in the main DeepResearchWorkflow class.
"""

from .planning import PlanningPhaseMixin

__all__ = [
    "PlanningPhaseMixin",
]
