from __future__ import annotations

import re
from functools import cached_property
from typing import Dict, Optional, List, Union

from loguru import logger
from pytz import utc

from shared_models.analysis_window import AnalysisWindow
from shared_models.contributor_store import UniqueContributorStore, Contributor
from shared_models.commit import CommitSM
from shared_models.issue import IssueSM, PullRequestSM
from shared_models.repo_data import RepoData


FILE_PATH_REGEX = re.compile("(\/?[\w~,;\-\.\/?%&+#=]+\.[a-z]+)") # Leading slash is optional, must have an extension


class AnalysisTarget(RepoData):
    unique_contributors: Optional[UniqueContributorStore] = None

    repo_name: Optional[str] = None
    commits_by_file: Optional[Dict[str, List[str]]] = None
    prs_by_file: Optional[Dict[str, List[int]]] = None
    issues_by_file: Optional[Dict[str, List[int]]] = None

    @classmethod
    def from_repo_data(cls, name: str, repo_data: RepoData, window: AnalysisWindow) -> AnalysisTarget:
        """
        Returns an analysis target by applying window (a filter based on points in time) to the repo_data
        """
        logger.info(f"Constraining repo data for {name}")
        if window.start is None and window.end is None:
            logger.info(f"Given window has no set start or end so taking full history")
            return AnalysisTarget(commits=repo_data.commits, issues=repo_data.issues, prs=repo_data.prs)

        both_pots_set = window.start is not None and window.end is not None
        both_dates_set = False
        both_issues_set = False
        issue_nums_present = False
        commit_hashes_present = False

        if both_pots_set:
            both_dates_set = both_pots_set and window.start.date_time is not None and window.end.date_time is not None
            both_issues_set = window.start.issue_number is not None and window.end.issue_number is not None
        else:
            if window.start is not None:
                issue_nums_present = window.start.issue_number is not None
                commit_hashes_present = window.start.commit_hash is not None
            if window.end is not None:
                issue_nums_present = issue_nums_present or window.end.issue_number is not None
                commit_hashes_present = commit_hashes_present or window.end.commit_hash is not None

        if both_dates_set and window.start.date_time > window.end.date_time:
            logger.warning(
                f"Start Date of {window.start.date_time} is after {window.end.date_time} so taking full history"
            )
            return AnalysisTarget(commits=repo_data.commits, issues=repo_data.issues, prs=repo_data.prs)

        if both_issues_set and window.start.issue_number > window.end.issue_number:
            logger.warning(
                f"Start Issue of {window.start.issue_number} is after {window.end.issue_number} so taking full history"
            )
            return AnalysisTarget(commits=repo_data.commits, issues=repo_data.issues, prs=repo_data.prs)

        result = AnalysisTarget(repo_name=name, commits=[], issues=[], prs=[])

        _start_d = None
        _end_d = None
        _start_i = None
        _end_i = None
        _start_c = None
        _end_c = None
        if window.start is not None:
            _start_d = window.start.date_time
            _start_i = window.start.issue_number
            _start_c = window.start.commit_hash
        if window.end is not None:
            _end_d = window.end.date_time
            _end_i = window.end.issue_number
            _end_c = window.end.commit_hash

        if issue_nums_present and _start_d is None and _end_d is None:
            if _start_i is not None:
                target_issue = cls.get_issues_or_prs(repo_data.issues, repo_data.prs, _start_i)[0]
                _start_d = target_issue.created_at
            if _end_i is not None:
                target_issue = cls.get_issues_or_prs(repo_data.issues, repo_data.prs, _end_i)[0]
                _end_d = target_issue.created_at

        elif commit_hashes_present and _start_d is None and _end_d is None:
            if _start_c is not None:
                target_commit = cls.get_commits(repo_data.commits, _start_c)[0]
                _start_d = target_commit.committer_date
            if _end_c is not None:
                target_commit = cls.get_commits(repo_data.commits, _end_c)[0]
                _end_d = target_commit.committer_date

        if _start_d is not None and _start_d.tzinfo is None:
            _start_d = utc.localize(_start_d)
        if _end_d is not None and _end_d.tzinfo is None:
            _end_d = utc.localize(_end_d)
        logger.info(
            f"Converted Constraints to datetime constraints - "
            f"StartDate: {_start_d} - "
            f"EndDate: {_end_d}"
        )
        # Repo data has commits issues and prs with index 0 being most recent and working back in time
        # ━━━━ Constrain Commits ━━━━ #
        for commit in repo_data.commits:
            commit_date = utc.localize(commit.committer_date) if commit.committer_date.tzinfo is None else commit.committer_date
            if _start_d is not None and _end_d is not None:
                # Add commit if commit.committer_date is between the two
                if _start_d < commit_date < _end_d:
                    result.commits.append(commit)
            elif _start_d is not None and _end_d is None:
                # Add commit if commit.Committer is newer than _start_d
                if _start_d < commit_date:
                    result.commits.append(commit)
                else:
                    break
            elif _start_d is None and _end_d is not None:
                # Add commit if commit.committer_date is older than _end_d
                if commit_date < _end_d:
                    result.commits.append(commit)

        # ━━━━ Constrain Issues ━━━━ #
        for issue in repo_data.issues:
            issue_date = utc.localize(issue.created_at) if issue.created_at.tzinfo is None else issue.created_at
            if _start_d is not None and _end_d is not None:
                # Add issue if issue.created_at is between the two
                if _start_d < issue_date < _end_d:
                    result.issues.append(issue)
            elif _start_d is not None and _end_d is None:
                # Add issue if issue.created_at is newer than _start_d
                if _start_d < issue_date:
                    result.issues.append(issue)
                else:
                    break
            elif _start_d is None and _end_d is not None:
                # Add issue if issue.created_at is older than _end_d
                if issue_date < _end_d:
                    result.issues.append(issue)

        # ━━━━ Constrain PRs ━━━━ #
        for pr in repo_data.prs:
            pr_date = utc.localize(pr.created_at) if pr.created_at.tzinfo is None else pr.created_at
            if _start_d is not None and _end_d is not None:
                # Add pr if pr.created_at is between the two
                if _start_d < pr_date < _end_d:
                    result.prs.append(pr)
            elif _start_d is not None and _end_d is None:
                # Add pr if pr.created_at is newer than _start_d
                if _start_d < pr_date:
                    result.prs.append(pr)
                else:
                    break
            elif _start_d is None and _end_d is not None:
                # Add pr if pr.created_at is older than _end_d
                if pr_date < _end_d:
                    result.prs.append(pr)

        return result

    def identify_unique_contributors(self):
        """
        Populates the unique contributors store for this analysis target,
        while also grouping commits, issues and PRs by User
        """
        self.unique_contributors = UniqueContributorStore()
        for commit in self.commits:
            self.unique_contributors.add_or_update(
                Contributor(
                    name=commit.author_name,
                    email=commit.author_email,
                    commits_made={commit.commit_hash}
                )
            )
        contrib_count = len(self.unique_contributors.store.keys())
        logger.info(f"{contrib_count} Unique Contributors found in commit history")

        for issue in self.issues:
            self.unique_contributors.add_or_update(
                Contributor(
                    name=issue.created_by.username,
                    email=issue.created_by.email,
                    issues_created={issue.number}
                )
            )
            for assignee in issue.assignees:
                self.unique_contributors.add_or_update(
                    Contributor(name=assignee.username, email=assignee.email)
                )
            for comment in issue.comments:
                self.unique_contributors.add_or_update(
                    Contributor(
                        name=comment.author.username,
                        email=comment.author.email,
                        issues_commented_on={issue.number}
                    )
                )
        logger.info(
            f"{len(self.unique_contributors.store.keys())-contrib_count} New Unique Contributors found in Issues"
        )
        contrib_count=len(self.unique_contributors.store.keys())-contrib_count

        for pr in self.prs:
            self.unique_contributors.add_or_update(
                Contributor(
                    name=pr.created_by.username,
                    email=pr.created_by.email,
                    prs_created={pr.number}
                )
            )
            for assignee in pr.assignees:
                self.unique_contributors.add_or_update(
                    Contributor(name=assignee.username, email=assignee.email)
                )
            for comment in pr.comments:
                self.unique_contributors.add_or_update(
                    Contributor(
                        name=comment.author.username,
                        email=comment.author.email,
                        prs_commented_on={pr.number}
                    )
                )
            for review in pr.reviews:
                if review.author is None:
                    continue
                self.unique_contributors.add_or_update(
                    Contributor(
                        name=review.author.username,
                        email=review.author.email,
                        prs_commented_on={pr.number}
                    )
                )
        logger.info(
            f"{len(self.unique_contributors.store.keys())-contrib_count} New Unique Contributors found in PRs"
        )

# ━━━━━━━━━━━━━━━━━━    Group X by File    ━━━━━━━━━━━━━━━━━━ #
    def group_commits_by_file(self) -> Dict[str, List[str]]:
        """
        Groups all the commits of the project by which file they touch.
        A commit can appear in several groups if multiple files are touched.
        Result stored to avoid re-compute
        """
        result = dict()
        for commit in self.commits:
            for change_log in commit.changes:
                if change_log.file in result:
                    result[change_log.file].append(commit.commit_hash)
                else:
                    result[change_log.file] = [commit.commit_hash]
        self.commits_by_file = result
        logger.info(f"{len(self.commits)} Commits touching {len(self.commits_by_file.keys())} files grouped")
        return self.commits_by_file

    def group_prs_by_file(self) -> Dict[str, List[int]]:
        """
        Groups all the Pull Requests of the project by which files they touch.
        A PR can appear in several groups if multiple files are touched.
        Result stored to avoid re-compute
        """
        result=dict()
        for pr in self.prs:
            for commit in pr.commits:
                for change_log in commit.changes:
                    if change_log.file in result:
                        result[change_log.file].append(pr.number)
                    else:
                        result[change_log.file] = [pr.number]
        self.prs_by_file = result
        logger.info(f"{len(self.prs)} PRs touching {len(self.prs_by_file.keys())} files grouped")
        return self.prs_by_file

    def group_issues_by_file(self) -> Dict[str, List[int]]:
        """
        Groups all the Issues of the project by which files they mention.
        An issue can appear in several groups if multiple files are mentioned.
        Where no File is mentioned the Issues are placed under the key: NO-RELATED-FILE
        Result stored to avoid re-compute
        """
        result = {"NO-RELATED-FILE": []}
        for issue in self.issues:
            related_file_paths = []
            file_paths_in_title = FILE_PATH_REGEX.findall(issue.title)
            for path in file_paths_in_title:
                matching_known_path = self.file_path_in_known_files(path)
                if matching_known_path is not None:
                    related_file_paths.append(matching_known_path)

            file_paths_in_body = FILE_PATH_REGEX.findall(issue.body)
            for path in file_paths_in_body:
                matching_known_path = self.file_path_in_known_files(path)
                if matching_known_path is not None:
                    related_file_paths.append(matching_known_path)

            file_paths_in_comments = []
            for comment in issue.comments:
                paths_in_comment = FILE_PATH_REGEX.findall(comment.body)
                file_paths_in_comments += set(paths_in_comment)
            for path in file_paths_in_comments:
                matching_known_path = self.file_path_in_known_files(path)
                if matching_known_path is not None:
                    related_file_paths.append(matching_known_path)

            if len(related_file_paths) == 0:
                result["NO-RELATED-FILE"].append(issue.number)
                continue

            for related_path in related_file_paths:
                if related_path in result:
                    result[related_path].append(issue.number)
                else:
                    result[related_path] = [issue.number]

        # De-dupe entries in each of the lists - arrises from multiple mentions of same file in single issue
        result = {k: list(set(v)) for k, v in result.items()}

        self.issues_by_file = result
        logger.info(f"{len(self.issues)} Issues mentioning {len(self.issues_by_file.keys())} files grouped")
        return self.issues_by_file

# ━━━━━━━━━━━━━━━━━━━━━━━━━━    Helpers    ━━━━━━━━━━━━━━━━━━━━━━━━━━ #
    @staticmethod
    def get_issues_or_prs(
        issues: List[IssueSM] = None, prs: List[PullRequestSM] = None, issue_numbers: Union[int, List[int]] = None
    ) -> Optional[List[Union[IssueSM, PullRequestSM]]]:

        if issue_numbers is None:
            return None
        if issues is None:
            issues = []
        if prs is None:
            prs = []

        if not isinstance(issue_numbers, list):
            issue_numbers = [issue_numbers]
        result = []
        for issue in issues:
            if issue.number in issue_numbers:
                result.append(issue)
        for pr in prs:
            if pr.number in issue_numbers:
                result.append(pr)
        if len(result) == 0:
            return None
        return result

    @staticmethod
    def get_commits(commits: List[CommitSM], commit_hashes: Union[str, List[str]]) -> Optional[List[CommitSM]]:
        if not isinstance(commit_hashes, list):
            commit_hashes = [commit_hashes]
        result = []
        for commit in commits:
            if commit.commit_hash in commit_hashes:
                result.append(commit)
                if len(result) == len(commit_hashes):
                    return result
        if len(result) == 0:
            return None

    @cached_property
    def known_files(self) -> List[str]:
        if self.prs_by_file is not None:
            files_in_prs = set(self.prs_by_file.keys())
        else:
            files_in_prs = set()

        if self.commits_by_file is not None:
            files_in_commits = set(self.commits_by_file.keys())
        else:
            files_in_commits = set()

        if self.issues_by_file is not None:
            file_in_issues = set(self.issues_by_file.keys())
        else:
            file_in_issues = set()
        return list(files_in_prs.union(files_in_commits).union(file_in_issues))

    def file_path_in_known_files(self, file_path: str) -> Optional[str]:
        known_files = self.known_files
        matches = [f for f in known_files if file_path in f]
        if len(matches) == 0:
            return None
        return matches[0]
