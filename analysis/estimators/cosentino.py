"""
Estimator built using code from the tool package: https://github.com/SOM-Research/busfactor
released alongside:
V. Cosentino, J. L. C. Izquierdo and J. Cabot - "Assessing the bus factor of Git repositories" - SANER2015
See Cosentino-License for the license under which portions of the cited code were copied

The estimator presented in Cosentino's work is highly configurable.
This implementation only supports File level analysis, however does support all 4 presented metrics.
M1 - Last change takes it all
M2 - Multiple changes equally considered
M3 - Non-consecutive changes
M4 - Weighted non-consecutive changes
"""
from enum import Enum
from typing import Tuple, List, Dict, Callable

from analysis.estimators.base_estimator import TruckFactorEstimator, EstimatorConfig
from shared_models.analysis_target import AnalysisTarget
from shared_models.commit import CommitSM


class KnowledgeMetric(str, Enum):
    M1 = "last-change-takes-all"
    M2 = "equal-weight-multi-change"
    M3 = "non-consec-changes"
    M4 = "weighted-non-consec-changes"


class CosentinoConfig(EstimatorConfig):
    knowledge_metric: KnowledgeMetric = KnowledgeMetric.M4
    secondary_expert_threshold: float = 0.5


class CosentinoEstimator(TruckFactorEstimator):
    config: CosentinoConfig
    target: AnalysisTarget

    _DOKS_METRIC_MAP: Dict[KnowledgeMetric, Callable[[List[CommitSM]], Dict[str, float]]] = dict()

    def __init__(self, config: CosentinoConfig):
        self.config = config
        self._DOKS_METRIC_MAP = {
            KnowledgeMetric.M1: self._calc_last_change_takes_all_doks,
            KnowledgeMetric.M2: self._calc_equal_multi_change_doks,
            KnowledgeMetric.M3: self._calc_non_consec_changes_doks,
            KnowledgeMetric.M4: self._calc_weighted_non_consec_doks,
        }

    def load_project(self, project_data: AnalysisTarget):
        self.target = project_data

    def run_estimation(self) -> Tuple[int, List[str]]:
        truck_factor = 0
        truck_factor_contributors = []
        file_expert_map: Dict[str, List[Tuple[str, float]]] = dict()
        for file_path, commit_hashes in self.target.commits_by_file.items():
            file_expert_map[file_path] = []
            relevant_commits = self.target.get_commits(self.target.commits, commit_hashes)[::-1]
            # Get Degree of Knowledge for all authors that've touched the file (in percentage)
            author_doks = self._DOKS_METRIC_MAP[self.config.knowledge_metric](relevant_commits)
            authors_by_dok: List[str] = [
                author for author, dok in sorted(author_doks.items(), key=lambda kv_tuple: kv_tuple[1], reverse=True)
            ]
            # Identify Primary Experts for the file
            primary_experts = self._identify_primary_experts(author_doks, authors_by_dok)
            primary_expertise_coverage = sum(pct for expert, pct in primary_experts)
            # Bus threshold allows for inclusion of primary and secondary experts in the BF estimate
            if len(primary_experts) == 0:
                bus_threshold = (1/len(author_doks)) * self.config.secondary_expert_threshold
            else:
                bus_threshold = (primary_expertise_coverage/len(primary_experts))*self.config.secondary_expert_threshold

            for author in authors_by_dok:
                if author_doks[author] >= bus_threshold:
                    file_expert_map[file_path].append((author, author_doks[author]))
                else:
                    break

        # Now that file level Bus experts are identified, repeat similar pattern at the project level
        truck_factor_contributors = self._file_experts_to_proj_experts(file_expert_map)
        return len(truck_factor_contributors), truck_factor_contributors

    def _calc_last_change_takes_all_doks(self, commits: List[CommitSM]) -> Dict[str, float]:
        """
        The latest commit author gets credited with total knowledge of the file
        """
        return {self._get_author_key(commits[-1]): 1}

    def _calc_equal_multi_change_doks(self, commits: List[CommitSM]) -> Dict[str, float]:
        """
        All authors are credited with knowledge based on the number of commits made
        """
        raw_author_doks = dict()
        for commit in commits:
            author = self._get_author_key(commit)
            if author in raw_author_doks:
                raw_author_doks[author] += 1
            else:
                raw_author_doks[author] = 1
        # Convert values into percentage of total knowledge
        divisor = sum(raw_author_doks.values())
        return {author: raw / divisor for author, raw in raw_author_doks.items()}

    def _calc_non_consec_changes_doks(self, commits: List[CommitSM]) -> Dict[str, float]:
        """
        All authors are credited with knowledge based on the number of non-consecutive commits.
        Deals with issue of people spamming multiple small commits in a row
        """
        raw_author_doks = dict()
        previous_author: str = ""
        for commit in commits:
            author = self._get_author_key(commit)
            if author != previous_author:
                if author in raw_author_doks:
                    raw_author_doks[author] += 1
                else:
                    raw_author_doks[author] = 1
                previous_author = author
        # Convert values into percentage of total knowledge
        divisor = sum(raw_author_doks.values())
        return {author: raw / divisor for author, raw in raw_author_doks.items()}

    def _calc_weighted_non_consec_doks(self, commits: List[CommitSM]) -> Dict[str, float]:
        """
        All authors are credited with knowledge based on number of non-consecutive commits, with a weighting
        applied to value recent commits over older commits.
        """
        raw_author_doks = dict()
        previous_author: str = ""
        author_sequence = []
        # Index 0 of commits is oldest commit
        for commit in commits:
            author = self._get_author_key(commit)
            if author != previous_author:
                author_sequence.append(author)
                previous_author = author
        for idx, author in enumerate(author_sequence):
            if author in raw_author_doks:
                raw_author_doks[author] += (idx+1)
            else:
                raw_author_doks[author] = (idx+1)
        # Convert values into percentage of total knowledge
        divisor = sum(raw_author_doks.values())
        return {author: raw / divisor for author, raw in raw_author_doks.items()}

    def _get_author_key(self, commit: CommitSM) -> str:
        if commit.author_email is not None and "@" in commit.author_email:
            email = commit.author_email
        else:
            email = None
        author_key = self.target.unique_contributors.get(commit.author_name, email).get_key()
        return author_key

    @staticmethod
    def _identify_primary_experts(author_doks: Dict[str, float], authors_by_dok:List[str]) -> List[Tuple[str, float]]:
        primary_experts: List[Tuple[str, float]] = []
        # To be an expert on the file an author needs to have DoK >= 1/N pct
        # where N is number of authors who have touched the file
        primary_expert_knowledge_threshold = 1 / len(authors_by_dok)
        for author in authors_by_dok:
            if author_doks[author] >= primary_expert_knowledge_threshold:
                primary_experts.append((author, author_doks[author]))
            else:
                break
        return primary_experts

    def _file_experts_to_proj_experts(self, file_expert_map: Dict[str, List[Tuple[str, float]]]):
        # Calculate coverage of each expert
        expert_file_counts: Dict[str, int] = dict()
        total_file_count = len(file_expert_map)
        for file, experts in file_expert_map.items():
            for expert, _ in experts:
                if expert in expert_file_counts:
                    expert_file_counts[expert] += 1
                else:
                    expert_file_counts[expert] = 1
        expert_coverage: Dict[str, float] = {
            exp: round((fc / total_file_count), 4) for exp, fc in expert_file_counts.items()
        }
        exps_by_cov = [
            exp for exp, cov in sorted(expert_coverage.items(), key=lambda kv_tuple: kv_tuple[1], reverse=True)
        ]
        # Calculate BF Coverage Threshold
        pct_coverage = sum(pct for expert, pct in expert_coverage.items())
        bus_threshold = (pct_coverage / len(expert_coverage)) * self.config.secondary_expert_threshold
        # Apply BF Coverage Threshold
        bus_factor_experts: List[str] = []
        for expert in exps_by_cov:
            if expert_coverage[expert] >= bus_threshold:
                bus_factor_experts.append(expert)
            else:
                break
        return bus_factor_experts
