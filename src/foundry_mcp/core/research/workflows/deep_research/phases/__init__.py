"""Phase mixins for DeepResearchWorkflow.

Each mixin contributes a disjoint set of methods implementing one workflow phase.
They are combined via multiple inheritance in the main DeepResearchWorkflow class.
"""

from .planning import PlanningPhaseMixin
from .gathering import GatheringPhaseMixin
from .analysis import AnalysisPhaseMixin
from .synthesis import SynthesisPhaseMixin
from .refinement import RefinementPhaseMixin

__all__ = [
    "PlanningPhaseMixin",
    "GatheringPhaseMixin",
    "AnalysisPhaseMixin",
    "SynthesisPhaseMixin",
    "RefinementPhaseMixin",
]
