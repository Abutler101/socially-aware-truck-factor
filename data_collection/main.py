import csv
import shutil
import time
from pathlib import Path
from datetime import datetime, date
from subprocess import Popen, PIPE
from typing import List, Tuple

from git import Repo
from github.Issue import Issue
from github.PaginatedList import PaginatedList
from loguru import logger
from requests.exceptions import RetryError

from data_collection.api_clients.github_client import GithubClient
from shared_models.collection_target import CollectionTarget
from shared_models.commit import CommitSM, CommitChangeStat
from shared_models.issue import IssueSM, PullRequestSM, UserSM, PrReviewSM
from shared_models.repo_data import RepoData

COMMIT_DELIM = "----START~~~COMMIT----\n"
BODY_DELIM = "--END~~OF~~BODY--"
TARGETS_FILE_PATH = Path(__file__).parent.joinpath("targets.csv")
OUTPUT_PATH = Path(__file__).parent.joinpath("output")
LOG_PATH = OUTPUT_PATH.joinpath("logs")
SCRATCH_PATH = OUTPUT_PATH.joinpath("scratch")

OUTPUT_PATH.mkdir(exist_ok=True)
LOG_PATH.mkdir(exist_ok=True)
SCRATCH_PATH.mkdir(exist_ok=True)

logger.add(LOG_PATH.joinpath("{time}.log"), rotation="5h")


@logger.catch
def main():
    OUTPUT_PATH.mkdir(exist_ok=True)
    SCRATCH_PATH.mkdir(exist_ok=True)
    shutil.rmtree(SCRATCH_PATH)

    collection_targets = _read_targets_file()

    for target in collection_targets:
        repo_name = target.repo_url.replace("/", "-")
        clone_path = SCRATCH_PATH.joinpath(repo_name)
        repo_data = collect_repo_data(repo_name, clone_path, target)
        write_repo_data(repo_name, repo_data)


def write_repo_data(repo_name: str, repo_data: RepoData):
    with OUTPUT_PATH.joinpath(f"{repo_name}.json").open("w") as f:
        logger.info(f"Writing repo data to {f.name}")
        f.write(repo_data.model_dump_json(indent=4))


def collect_repo_data(repo_name: str, clone_path: Path, targeting_info: CollectionTarget):
    logger.info(f"Collecting Commits, Issues, and PRs for {repo_name}")
    clone_repo(targeting_info.clone_url.unicode_string(), clone_path, targeting_info.target_commit)
    logger.info(f"{repo_name} cloned to {clone_path}")

    all_commits = run_git_log(repo_name, clone_path)
    logger.info(f"{len(all_commits)} commits captured for {repo_name}")

    issues_and_prs: Tuple[List[IssueSM], List[PullRequestSM]] = get_issues(targeting_info)
    all_issues: List[IssueSM] = issues_and_prs[0]
    all_prs: List[PullRequestSM] = issues_and_prs[1]
    logger.info(f"{len(all_issues)} issues captured for {repo_name}")
    logger.info(f"{len(all_prs)} prs captured for {repo_name}")
    return RepoData(commits=all_commits, issues=all_issues, prs=all_prs)


def clone_repo(clone_url: str, clone_path: Path, commit_hash: str):
    repo = Repo.clone_from(url=clone_url, to_path=clone_path)
    repo.git.checkout(commit_hash)
    repo.close()


def run_git_log(repo: str, clone_path: Path) -> List[CommitSM]:
    """
    Runs the command:
    git log -M -C --no-merges --numstat --format=----START~~~COMMIT----%n%T%n%H%n%aN%n%aE%n%aI%n%cN%n%cE%n%cI%n%s%n%b%n
    and parses the resultant string
    """
    cmd = [
        "git", "log", "-M", "-C", "--no-merges", "--numstat",
        f"--format={COMMIT_DELIM}%T%n%H%n%aN%n%aE%n%aI%n%cN%n%cE%n%cI%n%s%n%b%n{BODY_DELIM}"
    ]
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=clone_path)
    stdout, stderr = p.communicate()
    raw_commit_log = stdout.decode("utf-8", errors="ignore")
    parsed_commit_entries = _parse_raw_git_log(raw_commit_log)

    return parsed_commit_entries


def _parse_raw_git_log(raw_commit_log: str) -> List[CommitSM]:
    raw_commit_strings = [
        raw_string for raw_string in raw_commit_log.split(COMMIT_DELIM) if len(raw_string) > 0
    ]

    parsed_commit_entries: List[CommitSM] = []
    for raw_commit in raw_commit_strings:
        split_on_newlines = raw_commit.splitlines()

        # Find index of line marking end of the commit msg body
        body_delim_index = split_on_newlines.index(BODY_DELIM)
        # Bodies can be multiline, so build the body out of all lines between
        # end of fixed length section and body end index
        body = "".join(
            [val for val in split_on_newlines[9::body_delim_index] if val != BODY_DELIM]
        )

        # List of changes is multiline, so parse each change line from end of body marker to end of commit entry
        raw_change_strings = [val for val in split_on_newlines[body_delim_index + 1::] if len(val) > 0]
        parsed_change_entries: List[CommitChangeStat] = []
        for raw_change in raw_change_strings:
            split_on_tabs = raw_change.split("\t")
            change_entry = CommitChangeStat(
                file=split_on_tabs[2],
                addition_count=0 if split_on_tabs[0] == "-" else split_on_tabs[0],
                deletion_count=0 if split_on_tabs[1] == "-" else split_on_tabs[1]
            )
            parsed_change_entries.append(change_entry)

        # Deal with TZinfo being weird / mallformed by dropping it
        try:
            author_date = datetime.fromisoformat(split_on_newlines[4])
        except ValueError as err:
            author_date = datetime.fromisoformat(split_on_newlines[4][:19])

        try:
            committer_date = datetime.fromisoformat(split_on_newlines[7])
        except ValueError as err:
            committer_date = datetime.fromisoformat(split_on_newlines[7][:19])

        entry = CommitSM(
            tree_hash=split_on_newlines[0],
            commit_hash=split_on_newlines[1],
            author_name=split_on_newlines[2],
            author_email=split_on_newlines[3],
            author_date=author_date,
            committer_name=split_on_newlines[5],
            committer_email=split_on_newlines[6],
            committer_date=committer_date,
            subject=split_on_newlines[8],
            body=body,
            changes=parsed_change_entries
        )
        parsed_commit_entries.append(entry)

    return parsed_commit_entries


def get_issues(target: CollectionTarget) -> Tuple[List[IssueSM], List[PullRequestSM]]:
    github_client = GithubClient()
    repo = github_client.get_git_repo(target.repo_url)
    issues: PaginatedList[Issue] = repo.get_issues(state="all")
    parsed_issues: List[IssueSM] = []
    parsed_prs: List[PullRequestSM] = []
    logger.debug(f"{target.repo_url} has {issues.totalCount} Issues+PRs")
    for raw_issue in issues:
        try:
            parsed_issue = IssueSM.from_gh_issue(raw_issue)
        except RetryError as err:
            if err.__cause__ is not None and "too many 503" in err.__cause__.__str__():
                logger.info(f"Got a too many 503s error. Backing off for 3 hours...")
                time.sleep(10800)
                logger.info(f"Re-attempting to parse the issue")
                parsed_issue = IssueSM.from_gh_issue(raw_issue)
            else:
                raise err

        if raw_issue.pull_request is not None:
            raw_pr = raw_issue.as_pull_request()
            parsed_pr = PullRequestSM(
                **parsed_issue.model_dump(),
                changed_files=raw_pr.changed_files,
                additions=raw_pr.additions,
                deletions=raw_pr.deletions,
                commits=[CommitSM.from_api_commit(commit) for commit in raw_pr.get_commits()],
                requested_reviewers=[UserSM.from_named_user(user) for user in raw_pr.requested_reviewers],
                reviews=[PrReviewSM.from_review(review) for review in raw_pr.get_reviews()]
            )
            parsed_prs.append(parsed_pr)
        else:
            parsed_issues.append(parsed_issue)
        logger.debug(f"{len(parsed_issues)+len(parsed_prs)} / {issues.totalCount} Issues parsed for {target.repo_url}")
    return parsed_issues, parsed_prs


def _read_targets_file() -> List[CollectionTarget]:
    with TARGETS_FILE_PATH.open("r") as f:
        collection_targets = []
        for row in csv.DictReader(f, skipinitialspace=True):
            collection_targets.append(
                CollectionTarget(
                    repo_url=row["repo_url"],
                    clone_url=row["clone_url"],
                    target_commit=row["target_commit"],
                    target_date=date.fromisoformat(row["target_date"]),
                )
            )
    return collection_targets


if __name__ == '__main__':
    main()
