from typing import List

from pydantic import BaseModel


class GroundTruthEntry(BaseModel):
    """
    Representation of Ground truth for Truck Factor for a given project.
    If extension_target is True, this project was added as part of Ferreira et al.'s
    extension of Avelino et al.'s validation dataset and therefore could not be fully verified.
    extension_target ground truths therefore carry uncertainty as to when exactly the ground
    truth was captured
    """
    name: str
    truck_factor: int
    truck_factor_contributors: List[str]
    extension_target: bool


class GroundTruth(BaseModel):
    projects: List[GroundTruthEntry]

    def get_project(self, project_name: str) -> GroundTruthEntry:
        matches = [entry for entry in self.projects if entry.name == project_name]
        return matches[0]

    @property
    def project_count(self) -> int:
        return len(self.projects)
