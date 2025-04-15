from typing import List, Tuple, Dict, Optional

from pydantic import BaseModel


class EstimatorFailure(BaseModel):
    false_positive_contribs: List[str] = []  # Those who were wrongly INCLUDED in the TF count
    false_negative_contribs: List[str] = []  # Those who were wrongly EXCLUDED from the TF count


class EstimatorPerf(BaseModel):
    correct_estimates: List[str] = []
    over_estimates: Dict[str, EstimatorFailure] = dict()
    under_estimates: Dict[str, EstimatorFailure] = dict()
    accuracy: float = 0.0  # should only be set through parent container calc_accuracies method


class EstimatorPerfContainer(BaseModel):
    project_count: int
    performance_entries: Dict[str, EstimatorPerf] = dict()

    def calc_accuracies(self):
        for entry in self.performance_entries.values():
            entry.accuracy = len(entry.correct_estimates)/self.project_count

    def get(self, e_key: str) -> Optional[EstimatorPerf]:
        return self.performance_entries.get(e_key, None)
