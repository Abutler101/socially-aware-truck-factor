from typing import List

from pydantic import BaseModel


class CandidateAuthorInfo(BaseModel):
    first_author: bool = False
    change_count: int = 0
    prs_created: int = 0
    prs_cmntd_on: int = 0
    issues_created: int = 0
    issues_cmntd_on: int = 0
