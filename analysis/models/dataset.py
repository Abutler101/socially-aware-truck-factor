from __future__ import annotations
from datetime import datetime
from typing import List, Any, Union
from pathlib import Path, PosixPath

from pydantic import BaseModel, model_validator

DATASET_PATH = Path(__file__).parents[2].joinpath("data_collection/output")


class DatasetEntry(BaseModel):
    path: Union[Path, PosixPath]
    end_date: datetime

    @model_validator(mode="after")
    def expand_path(self) -> DatasetEntry:
        self.path = DATASET_PATH.joinpath(self.path)
        return self


class Dataset(BaseModel):
    targets: List[DatasetEntry]
