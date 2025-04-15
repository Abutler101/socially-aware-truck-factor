"""
Extended DoK estimator makes use of the two part Degree of Knowledge model defined in:
    Fritz2010: https://doi.org/10.1145/1806799.1806856
    Fritz2014: https://doi.org/10.1145/2512207

Where Degree of Knowledge DoK = (Wa * DoA) + (Wb * DoI)
Degree of Authorship (DoA) is calculated the same way as in the Avelino estimator:
    Avelino2016: https://doi.org/10.1109/ICPC.2016.7503718
Where DoA = BA + (Wc * FA) + (Wd * DL) - (Wf * ln(1+AC))
With BA = base authorship, FA = first Authorship, DL = Deliveries = num changes made,
    AC = Accepted Changes = num changes made by others
Degree of Interest (DoI) is calculated based on a linear combination of metrics capturing issue and PR engagement.
Where DoI = (Wg * PrCr) + (Wh * PrCo) + (Wi * ICr) + (Wj * ICo)
With PrCr = num PRs Created, PrCo = num Comments made on PRs,
    ICr = num Issues Created, ICo = num Comments made on Issues

Follows the same TF approach as avelino, just replacing DoA with the DoK

weights expressed in above definitions are given readable names in EDoKConfig. Weights in the DoA equation are taken
from the Avelino estimator, the rest were established through empirical experimentation (see paper for details).
"""
import math
from typing import Tuple, List, Dict, Optional

from analysis.estimators.base_estimator import TruckFactorEstimator, EstimatorConfig
from shared_models.analysis_target import AnalysisTarget
from analysis.models.candidate_author import CandidateAuthorInfo
from analysis.shared_utils.third_party_files import load_path_regexes
from shared_models.commit import CommitSM
from shared_models.issue import PullRequestSM, IssueSM


class EDoKConfig(EstimatorConfig):
    # DoA related Weights
    base_doa: float = 3.293
    first_authorship_weight: float = 1.098
    num_changes_weight: float = 0.164
    others_change_count_weight: float = 0.321

    # DoI related Weights
    num_prs_created_weight: float = 1.0
    num_pr_cmnts_made_weight: float = 1.0
    num_issues_created_weight: float = 1.0
    num_issue_cmnts_made_weight: float = 1.0

    # DoK Weights
    doa_weight: float = 1.0
    doi_weight: float = 1.0

    # Thresholds
    dok_norm_threshold: float = 0.75
    dok_abs_threshold: float = 3.293
    coverage_threshold: float = 0.5


class EDoKEstimator(TruckFactorEstimator):
    config: EDoKConfig
    target: AnalysisTarget

    def __init__(self, config: EDoKConfig):
        self.config = config
        self._third_party_detection_regex = load_path_regexes()

    def load_project(self, project_data: AnalysisTarget):
        self.target = project_data

    def run_estimation(self) -> Tuple[int, List[str]]:
        truck_factor = 0
        truck_factor_contributors = []
        file_author_map: Dict[str, List[str]] = dict()

        # Identify authors for all non-third-party files
        for file_path, commit_hashes in self.target.commits_by_file.items():
            if self._is_third_party(file_path):
                continue

            if self.target.prs_by_file is not None:
                pr_nums = self.target.prs_by_file.get(file_path, [])
            else:
                pr_nums = []

            if self.target.issues_by_file is not None:
                issue_nums = self.target.issues_by_file.get(file_path, [])
            else:
                issue_nums = []

            relevant_commits = self.target.get_commits(self.target.commits, commit_hashes)[::-1]
            relevant_prs = self.target.get_issues_or_prs(prs=self.target.prs, issue_numbers=pr_nums)
            relevant_issues = self.target.get_issues_or_prs(issues=self.target.issues, issue_numbers=issue_nums)

            # Each candidate_authors entry is a CandidateAuthorInfo model
            candidate_authors = self._extract_candidate_authors(relevant_commits, relevant_prs, relevant_issues)
            # Calculate DoK for each candidate author
            abs_doks = self._calc_dok_vals(candidate_authors)
            # Normalize the DoKs between 0 and 1
            abs_dok_max = max(abs_doks.values())
            normed_doas = {author: abs_dok/abs_dok_max for author, abs_dok in abs_doks.items()}
            # Assign File Authorship
            file_author_map[file_path] = self._find_authors(normed_doas, abs_doks)

        # Perform Greedy Truck Factor Estimation
        author_file_map = self._invert_file_author_map(file_author_map)
        authors_ranked_by_file_count = sorted(
            [(author, len(files)) for author, files in author_file_map.items()],
            key=lambda entry: entry[1],
            reverse=True
        )
        coverage = self._calc_coverage(file_author_map)
        while coverage > self.config.coverage_threshold:
            top_author = authors_ranked_by_file_count[truck_factor][0]
            self._pop_author_from_map(file_author_map, top_author)
            truck_factor_contributors.append(top_author)
            truck_factor += 1
            coverage = self._calc_coverage(file_author_map)
        return truck_factor, truck_factor_contributors

    def _is_third_party(self, file_path: str) -> bool:
        """
        Detects using the patterns defined in github-linguist/linguist if the file_path
        points to a third party file or if its actually part of the code base.
        """
        is_third_party = False
        is_third_party = bool(self._third_party_detection_regex.match(file_path))
        return is_third_party

    def _extract_candidate_authors(
        self,
        relevant_commits: List[CommitSM],
        relevant_prs: Optional[List[PullRequestSM]],
        relevant_issues: Optional[List[IssueSM]]
    ) -> Dict[str, CandidateAuthorInfo]:
        candidate_authors: Dict[str, CandidateAuthorInfo] = dict()

        if relevant_prs is None:
            relevant_prs = []
        if relevant_issues is None:
            relevant_issues = []

        # Extract author info from commits
        for commit in relevant_commits:
            if commit.author_email is not None and "@" in commit.author_email:
                email = commit.author_email
            else:
                email = None
            author = self.target.unique_contributors.get(commit.author_name, email)
            author_key = author.get_key()
            additions = 0
            deletions = 0
            for change_stat in commit.changes:
                additions += change_stat.addition_count
                deletions += change_stat.deletion_count
            if author_key in candidate_authors:
                candidate_authors[author_key].change_count += additions + deletions
            else:
                # First author of the file will be whoever wrote the first commit
                is_first_author = relevant_commits.index(commit) == 0
                candidate_authors[author_key] = CandidateAuthorInfo(
                    first_author=is_first_author, change_count=additions + deletions
                )

        # Extract author info from Prs
        for pr in relevant_prs:
            if pr.created_by.email is not None and "@" in pr.created_by.email:
                email = pr.created_by.email
            else:
                email = None
            if pr.created_by.username is not None:
                name = pr.created_by.username
            else:
                name = pr.created_by.name

            author = self.target.unique_contributors.get(name, email)
            author_key = author.get_key()
            if author_key in candidate_authors:
                candidate_authors[author_key].prs_created += 1
            else:
                candidate_authors[author_key] = CandidateAuthorInfo(prs_created=1)

            for comment in pr.comments:
                if comment.author is None:
                    continue
                if comment.author.email is not None and "@" in comment.author.email:
                    email = comment.author.email
                else:
                    email = None
                if comment.author.username is not None:
                    name = comment.author.username
                else:
                    name = comment.author.name

                commenter = self.target.unique_contributors.get(name, email)
                commenter_key = commenter.get_key()
                if commenter_key in candidate_authors:
                    candidate_authors[commenter_key].prs_cmntd_on +=1
                else:
                    candidate_authors[commenter_key] = CandidateAuthorInfo(prs_cmntd_on=1)

        # Extract author info from issues
        for issue in relevant_issues:
            if issue.created_by.email is not None and "@" in issue.created_by.email:
                email = issue.created_by.email
            else:
                email = None
            if issue.created_by.username is not None:
                name = issue.created_by.username
            else:
                name = issue.created_by.name

            author = self.target.unique_contributors.get(name, email)
            author_key = author.get_key()
            if author_key in candidate_authors:
                candidate_authors[author_key].prs_created += 1
            else:
                candidate_authors[author_key] = CandidateAuthorInfo(issues_created=1)
            for comment in issue.comments:
                if comment.author is None:
                    continue
                if comment.author.email is not None and "@" in comment.author.email:
                    email = comment.author.email
                else:
                    email = None
                if comment.author.username is not None:
                    name = comment.author.username
                else:
                    name = comment.author.name
                commenter = self.target.unique_contributors.get(name, email)
                if commenter is None:
                    continue
                commenter_key = commenter.get_key()
                if commenter_key in candidate_authors:
                    candidate_authors[commenter_key].issues_cmntd_on += 1
                else:
                    candidate_authors[commenter_key] = CandidateAuthorInfo(issues_cmntd_on=1)

        return candidate_authors

    def _calc_dok_vals(self, candidate_authors: Dict[str, CandidateAuthorInfo]) -> Dict[str, float]:
        abs_degrees_of_knowledge: Dict[str, float] = dict()
        DEBUG_dok_components: Dict[str, Tuple[float, float]] = dict()
        for author in candidate_authors.keys():
            others_change_count = 0
            for key, info in candidate_authors.items():
                if key == author:
                    continue
                others_change_count += info.change_count

            # Calculate Degree of Authorship
            doa_abs = self.config.base_doa
            weighted_first_authorship = candidate_authors[author].first_author * self.config.first_authorship_weight
            weighted_change_count = candidate_authors[author].change_count * self.config.num_changes_weight
            weighted_others_change_count = math.log(1+others_change_count) * self.config.others_change_count_weight
            doa_abs += weighted_first_authorship + weighted_change_count - weighted_others_change_count

            # Calculate Degree of Interest
            wghtd_pr_crt_count = candidate_authors[author].prs_created * self.config.num_prs_created_weight
            wghtd_pr_cmnt_count = candidate_authors[author].prs_cmntd_on * self.config.num_pr_cmnts_made_weight
            wghtd_issue_crt_count = candidate_authors[author].issues_created * self.config.num_issues_created_weight
            wghtd_issue_cmnt_count = candidate_authors[author].issues_cmntd_on * self.config.num_issue_cmnts_made_weight
            doi_abs = wghtd_pr_crt_count + wghtd_pr_cmnt_count + wghtd_issue_crt_count + wghtd_issue_cmnt_count

            DEBUG_dok_components[author] = (doa_abs, doi_abs)
            abs_degrees_of_knowledge[author] = doa_abs + doi_abs
        return abs_degrees_of_knowledge

    def _find_authors(self, norm_doks: Dict[str, float], abs_doks: Dict[str, float]) -> List[str]:
        authors = []
        for author, dok_norm in norm_doks.items():
            if dok_norm > self.config.dok_norm_threshold and abs_doks[author] >= self.config.dok_abs_threshold:
                authors.append(author)
        return authors

    @staticmethod
    def _calc_coverage(file_author_map: Dict[str, List[str]]) -> float:
        file_count = len(file_author_map.keys())
        orphan_files = [file_path for file_path, authors in file_author_map.items() if len(authors) == 0]
        return (file_count-len(orphan_files)) / file_count

    @staticmethod
    def _invert_file_author_map(file_author_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
        author_file_map = dict()
        for file_path, authors in file_author_map.items():
            for author in authors:
                if author not in author_file_map:
                    author_file_map[author] = [file_path]
                else:
                    author_file_map[author].append(file_path)
        return author_file_map

    @staticmethod
    def _pop_author_from_map(file_author_map: Dict[str, List[str]], author: str):
        for file_path, authors in file_author_map.items():
            if author in authors:
                authors.remove(author)
