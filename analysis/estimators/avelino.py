import math
import re
from typing import Tuple, List, Dict

from analysis.estimators.base_estimator import TruckFactorEstimator, EstimatorConfig
from shared_models.analysis_target import AnalysisTarget
from analysis.shared_utils.third_party_files import load_path_regexes
from shared_models.commit import CommitSM


class AvelinoConfig(EstimatorConfig):
    base_doa: float = 3.293
    first_authorship_weight: float = 1.098
    num_changes_weight: float = 0.164
    others_change_count_weight: float = 0.321

    doa_norm_threshold: float = 0.75
    doa_abs_threshold: float = 3.293
    coverage_threshold: float = 0.5


class AvelinoEstimator(TruckFactorEstimator):
    config: AvelinoConfig
    target: AnalysisTarget
    _third_party_detection_regex: re.Pattern

    def __init__(self, config: AvelinoConfig):
        self.config = config
        self._third_party_detection_regex = load_path_regexes()

    def load_project(self, project_data: AnalysisTarget):
        self.target = project_data

    def run_estimation(self) -> Tuple[int, List[str]]:
        contributors: List[Tuple[str, float]] = []
        truck_factor = 0
        truck_factor_contributors = []
        file_author_map: Dict[str, List[str]] = dict()

        # Identify authors for all non-third-party files
        for file_path, commit_hashes in self.target.commits_by_file.items():
            if self._is_third_party(file_path):
                continue
            relevant_commits = self.target.get_commits(self.target.commits, commit_hashes)[::-1]
            # Each candidate_authors entry is Tuple: (first contributor flag, num changes made)
            candidate_authors = self._extract_candidate_authors(relevant_commits)
            # Calculate DoA for each candidate author
            abs_doas = self._calc_doa_vals(candidate_authors)
            # Normalize the DoAs between 0 and 1
            abs_doa_max = max(abs_doas.values())
            normed_doas = {author: abs_doa/abs_doa_max for author, abs_doa in abs_doas.items()}
            # Assign File Authorship
            file_author_map[file_path] = self._find_authors(normed_doas, abs_doas)

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

    def _extract_candidate_authors(self, relevant_commits: List[CommitSM]) -> Dict[str, Tuple[bool, float]]:
        candidate_authors: Dict[str, Tuple[bool, float]] = dict()
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
                _scratch = candidate_authors[author_key]
                candidate_authors[author_key] = (_scratch[0], _scratch[1] + additions + deletions)
            else:
                # First author of the file will be whoever wrote the first commit
                is_first_author = relevant_commits.index(commit) == 0
                candidate_authors[author_key] = (is_first_author, (additions + deletions))
        return candidate_authors

    def _calc_doa_vals(self, candidate_authors: Dict[str, Tuple[bool, float]]) -> Dict[str, float]:
        abs_degrees_of_authorship: Dict[str, float] = dict()
        for author in candidate_authors.keys():
            others_change_count = 0
            for key, info in candidate_authors.items():
                if key == author:
                    continue
                others_change_count += info[1]

            doa_abs = self.config.base_doa
            weighted_first_authorship = candidate_authors[author][0] * self.config.first_authorship_weight
            weighted_change_count = candidate_authors[author][1] * self.config.num_changes_weight
            weighted_others_change_count = math.log(1+others_change_count) * self.config.others_change_count_weight
            doa_abs += weighted_first_authorship + weighted_change_count - weighted_others_change_count
            abs_degrees_of_authorship[author] = doa_abs
        return abs_degrees_of_authorship

    def _find_authors(self, norm_doas: Dict[str, float], abs_doas: Dict[str, float]) -> List[str]:
        authors = []
        for author, doa_norm in norm_doas.items():
            if doa_norm > self.config.doa_norm_threshold and abs_doas[author] >= self.config.doa_abs_threshold:
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
