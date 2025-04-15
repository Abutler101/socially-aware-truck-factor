from typing import List, Optional

from pydantic import BaseModel

from shared_models.commit import CommitSM
from shared_models.issue import IssueSM, PullRequestSM


class RepoData(BaseModel):
    commits: List[CommitSM]
    issues: Optional[List[IssueSM]]
    prs: Optional[List[PullRequestSM]]
