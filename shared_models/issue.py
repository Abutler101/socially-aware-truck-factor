from typing import Optional, List
from datetime import datetime

from github.Issue import Issue
from github.IssueComment import IssueComment
from github.NamedUser import NamedUser
from github.PullRequestReview import PullRequestReview
from pydantic import BaseModel

from shared_models.commit import CommitSM


class UserSM(BaseModel):
    id: int
    username: str
    created_at: datetime
    name: Optional[str]
    email: Optional[str]
    # Less interesting stuff
    company: Optional[str]
    location: Optional[str]
    bio: Optional[str]
    blog_url: Optional[str]
    contributions: Optional[int]

    @classmethod
    def from_named_user(cls, user: NamedUser):
        return UserSM(
            id=user.id,
            username=user.login,
            name=user.name,
            email=user.email,
            created_at=user.created_at,
            # Less interesting stuff
            company=user.company,
            location=user.location,
            bio=user.bio,
            blog_url=user.blog,
            contributions=user.contributions
        )


class CommentSM(BaseModel):
    id: int
    author: Optional[UserSM]
    created_at: datetime
    body: str

    @classmethod
    def from_gh_comment(cls, comment:IssueComment):
        if comment.user is None:
            author = None
        else:
            author = UserSM.from_named_user(comment.user)
        return CommentSM(
            id=comment.id,
            author=author,
            created_at=comment.created_at,
            body=comment.body
        )


class IssueSM(BaseModel):
    id: int
    number: int
    title: str
    created_at: datetime
    created_by: UserSM
    labels: List[str]
    body: str
    comments: List[CommentSM]
    assignees: List[UserSM]
    assignee: Optional[UserSM]

    @classmethod
    def from_gh_issue(cls, issue: Issue):
        assignee = None if issue.assignee is None else UserSM.from_named_user(issue.assignee)
        return IssueSM(
            id=issue.id,
            number=issue.number,
            title=issue.title,
            created_at=issue.created_at,
            created_by=UserSM.from_named_user(issue.user),
            labels=[label.name for label in issue.get_labels()],
            body=issue.body or "",
            comments=[CommentSM.from_gh_comment(comment) for comment in issue.get_comments()],
            assignees=[UserSM.from_named_user(user) for user in issue.assignees],
            assignee=assignee
        )


class PrReviewSM(CommentSM):
    commit_hash: Optional[str]
    state: str

    @classmethod
    def from_review(cls, review: PullRequestReview):
        if review.user is None:
            author = None
        else:
            author = UserSM.from_named_user(review.user)
        return PrReviewSM(
            id=review.id,
            author=author,
            created_at=review.submitted_at,
            body=review.body,
            commit_hash=review.commit_id,
            state=review.state,
        )


class PullRequestSM(IssueSM):
    changed_files: int
    commits: List[CommitSM]
    additions: int
    deletions: int
    requested_reviewers: List[UserSM]
    reviews: List[PrReviewSM]
