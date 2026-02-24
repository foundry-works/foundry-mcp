"""Phase mixins for DeepResearchWorkflow.

Each mixin contributes a disjoint set of methods implementing one workflow phase.
They are combined via multiple inheritance in the main DeepResearchWorkflow class.
"""

from .analysis import AnalysisPhaseMixin
from .brief import BriefPhaseMixin
from .clarification import ClarificationPhaseMixin
from .compression import CompressionMixin
from .gathering import GatheringPhaseMixin
from .planning import PlanningPhaseMixin
from .refinement import RefinementPhaseMixin
from .supervision import SupervisionPhaseMixin
from .synthesis import SynthesisPhaseMixin
from .topic_research import TopicResearchMixin

__all__ = [
    "BriefPhaseMixin",
    "ClarificationPhaseMixin",
    "PlanningPhaseMixin",
    "CompressionMixin",
    "GatheringPhaseMixin",
    "TopicResearchMixin",
    "SupervisionPhaseMixin",
    "AnalysisPhaseMixin",
    "SynthesisPhaseMixin",
    "RefinementPhaseMixin",
]
