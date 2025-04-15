from typing import Optional, List
from datetime import datetime

from github import UnknownObjectException, IncompletableObject
from github.Commit import Commit
from github.File import File
from pydantic import BaseModel
from loguru import logger


class CommitChangeStat(BaseModel):
    file: str
    addition_count: int
    deletion_count: int

    @classmethod
    def from_api_file_changes(cls, api_file_change: File):
        return CommitChangeStat(
            file=api_file_change.filename,
            addition_count=api_file_change.additions,
            deletion_count=api_file_change.deletions
        )


class CommitSM(BaseModel):
    tree_hash: str
    commit_hash: str
    author_name: str
    author_email: Optional[str]
    author_date: datetime
    committer_name: str
    committer_email: Optional[str]
    committer_date: datetime
    subject: str
    body: Optional[str]
    changes: List[CommitChangeStat]

    @classmethod
    def from_api_commit(cls, api_commit: Commit):
        commit_git_obj = api_commit.commit
        if commit_git_obj.author is not None and commit_git_obj.committer is not None:
            author = commit_git_obj.author
            commiter = commit_git_obj.committer
        else:
            author = commit_git_obj.committer
            commiter = commit_git_obj.committer

        # Try to get author and commiter email addresses
        try:
            author_email = author.email
        except Exception as err:
            logger.warning(f"Error Retrieving email address from author of Commit {api_commit.sha}")
            logger.debug(f"Problematic author Model: {author}")
            author_email = None
        try:
            commiter_email = commiter.email
        except Exception as err:
            logger.warning(f"Error Retrieving email address from committer of Commit {api_commit.sha}")
            logger.debug(f"problematic commiter model: {commiter}")
            commiter_email = None

        # Try to get author and commiter names
        try:
            author_name = author.name
        except Exception as err:
            logger.warning(f"Error Retrieving name from author of Commit {api_commit.sha}")
            logger.debug(f"Problematic author Model: {author}")
            author_name = None
        try:
            commiter_name = commiter.name
        except Exception as err:
            logger.warning(f"Error Retrieving name from committer of Commit {api_commit.sha}")
            logger.debug(f"problematic commiter model: {commiter}")
            commiter_name = None

        # Build Model
        return CommitSM(
            tree_hash=api_commit.commit.tree.sha,
            commit_hash=api_commit.sha,
            author_name=author_name,
            author_email=author_email,
            author_date=api_commit.last_modified_datetime,
            committer_name=commiter_name,
            committer_email=commiter_email,
            committer_date=api_commit.last_modified_datetime,
            subject=api_commit.commit.message,
            body=None,
            changes=[CommitChangeStat.from_api_file_changes(file_change) for file_change in api_commit.files]
        )
