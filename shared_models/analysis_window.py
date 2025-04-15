from __future__ import annotations
from typing import Optional
from datetime import datetime

from pydantic import BaseModel, model_validator
from pydantic_core import ValidationError


class PointInTime(BaseModel):
    """
    A point in repository history.
    At least one of: date_time, issue_number, commit_hash needs to be set.
    If multiple values are set, the order of precedence is: datetime, issue_number, commit_hash
    """
    date_time: Optional[datetime] = None
    issue_number: Optional[int] = None
    commit_hash: Optional[str] = None

    @model_validator(mode="after")
    def check_at_least_one_not_none(self) -> PointInTime:
        if self.date_time is None and self.commit_hash is None and self.issue_number is None:
            raise ValidationError("At least one of: date_time, commit_hash, issue_number has to be set")
        return self


class AnalysisWindow(BaseModel):
    """
    Start and end points for analysis of a project.
    If Start is None, analysis will run from project creation
    If End is None, analysis will run to end of project history
    """
    start: Optional[PointInTime] = None
    end: Optional[PointInTime] = None

    @classmethod
    def from_dates(cls, start: Optional[datetime], end: Optional[datetime]) -> AnalysisWindow:
        start_pot = None
        end_pot = None
        if start is not None:
            start_pot = PointInTime(date_time=start)
        if end is not None:
            end_pot = PointInTime(date_time=end)
        return AnalysisWindow(start=start_pot, end=end_pot)

    @classmethod
    def from_issue_numbers(cls, start: Optional[int], end: Optional[int]) -> AnalysisWindow:
        start_pot = None
        end_pot = None
        if start is not None:
            start_pot = PointInTime(issue_number=start)
        if end is not None:
            end_pot = PointInTime(issue_number=end)
        return AnalysisWindow(start=start_pot, end=end_pot)

    @classmethod
    def from_commit_hashes(cls, start: Optional[str], end: Optional[str]) -> AnalysisWindow:
        start_pot = None
        end_pot = None
        if start is not None:
            start_pot = PointInTime(commit_hash=start)
        if end is not None:
            end_pot = PointInTime(commit_hash=end)
        return AnalysisWindow(start=start_pot, end=end_pot)
