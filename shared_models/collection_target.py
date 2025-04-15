from datetime import date

from pydantic import BaseModel, AnyUrl


class CollectionTarget(BaseModel):
    repo_url: str
    clone_url: AnyUrl
    target_commit: str
    target_date: date
