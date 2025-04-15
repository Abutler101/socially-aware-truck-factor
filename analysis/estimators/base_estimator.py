import abc
from typing import Tuple, List

from pydantic import BaseModel

from shared_models.analysis_target import AnalysisTarget


class EstimatorConfig(BaseModel):
    pass


class TruckFactorEstimator(abc.ABC):
    config: EstimatorConfig
    target: AnalysisTarget

    @abc.abstractmethod
    def __init__(self, config: EstimatorConfig):
        pass

    @abc.abstractmethod
    def load_project(self, project_data: AnalysisTarget):
        pass

    @abc.abstractmethod
    def run_estimation(self) -> Tuple[int, List[str]]:
        pass
