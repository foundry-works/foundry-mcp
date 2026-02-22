"""Phase mixins for DeepResearchWorkflow.

Each mixin contributes a disjoint set of methods implementing one workflow phase.
They are combined via multiple inheritance in the main DeepResearchWorkflow class.
"""

from .analysis import AnalysisPhaseMixin
from .clarification import ClarificationPhaseMixin
from .gathering import GatheringPhaseMixin
from .planning import PlanningPhaseMixin
from .refinement import RefinementPhaseMixin
from .synthesis import SynthesisPhaseMixin

__all__ = [
    "ClarificationPhaseMixin",
    "PlanningPhaseMixin",
    "GatheringPhaseMixin",
    "AnalysisPhaseMixin",
    "SynthesisPhaseMixin",
    "RefinementPhaseMixin",
]
