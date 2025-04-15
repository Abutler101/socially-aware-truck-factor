import re
from enum import Enum
from pathlib import Path
from typing import Tuple, List, Dict, Callable, Any, Union

import networkx as nx
from loguru import logger

from analysis.estimators.base_estimator import TruckFactorEstimator, EstimatorConfig
from analysis.shared_utils.graph_measures import weighted_in_deg_cent, weighted_out_deg_cent, weighted_deg_cent
from shared_models.analysis_target import AnalysisTarget
from shared_models.issue import IssueSM, PullRequestSM

OUTPUT_PATH = Path(__file__).parents[1].joinpath("output/social_networks")
BOT_DETECT_PATTERN = re.compile("(\s|\[|\()(b|B)ot(\]|\))?")

class ImportanceMetric(str, Enum):
    PAGE_RANK = "PageRank"
    IN_DEG = "InDegree"
    OUT_DEG = "OutDegree"
    BET_CENT = "BetweenessCentrality"
    DEG_CENT = "DegreeCentrality"


class ThresholdMode(str, Enum):
    """
    How importance threshold is treated
    if ABS, the importance threshold is the floor and contribs with importance higher than the threshold are counted,
            threshold_pct ignored.
    if PERC, the importance threshold is the floor and top threshold_pct of contribs over the threshold are counted.
    """
    ABS = "Absolute"
    PERC = "TopXPercent"


class SocialGraphConfig(EstimatorConfig):
    fresh_connection_weight: float = 1.0
    contrib_importance_metric: ImportanceMetric = ImportanceMetric.DEG_CENT
    threshold_mode: ThresholdMode = ThresholdMode.ABS
    threshold_pct: float = 0.1
    importance_threshold: float = 0.55


class SocialGraphEstimator(TruckFactorEstimator):
    config: SocialGraphConfig
    target: AnalysisTarget

    _IMPORTANCE_METRIC_MAP: Dict[ImportanceMetric, Callable[[Any, str], Dict[str, float]]] = dict()

    def __init__(self, config: SocialGraphConfig):
        self.config = config

        self._IMPORTANCE_METRIC_MAP = {
            ImportanceMetric.PAGE_RANK: nx.pagerank,
            ImportanceMetric.IN_DEG: weighted_in_deg_cent,
            ImportanceMetric.OUT_DEG: weighted_out_deg_cent,
            ImportanceMetric.BET_CENT: nx.betweenness_centrality,
            ImportanceMetric.DEG_CENT: weighted_deg_cent,
        }

        OUTPUT_PATH.mkdir(exist_ok=True)

    def load_project(self, project_data: AnalysisTarget):
        self.target = project_data

    def run_estimation(self) -> Tuple[int, List[str]]:
        truck_factor = 0
        truck_factor_contribs = []
        # Check if social network has already been constructed, if so load, if not build
        graph_path = OUTPUT_PATH.joinpath(f"{self.target.repo_name}.gml")
        if graph_path.exists():
            proj_social_network = nx.read_gml(graph_path)
        else:
            proj_social_network = self._build_social_network()
            if proj_social_network is not None:
                nx.write_gml(proj_social_network, graph_path)

        # Rank contributors by Importance Metric - idx 0 is most important
        contrib_importance_scores = self._calc_contributor_importance(proj_social_network)
        contributors_by_importance = sorted(
            [(name, score) for name, score in contrib_importance_scores.items()],
            key=lambda entry: entry[1],
            reverse=True
        )

        # Apply threshold to importance ranked contributors
        filtered_contrib_score_pairs = [
            entry for entry in contributors_by_importance if entry[1] > self.config.importance_threshold
        ]
        if self.config.threshold_mode is ThresholdMode.PERC:
            expected_count = round(self.config.threshold_pct * len(filtered_contrib_score_pairs))
            filtered_contrib_score_pairs = filtered_contrib_score_pairs[:expected_count]
        truck_factor_contribs = [name for name, score in filtered_contrib_score_pairs]
        truck_factor = len(truck_factor_contribs)

        return truck_factor, truck_factor_contribs

    def _build_social_network(self) -> nx.DiGraph:
        """
        Parse all Issues and PRs, make following connections: (Based on Zanetti2013)
            Creator --> Reviewer                                DONE
            Reviewer --> Creator (only if actually reviewed)    DONE
            Creator --> Asignee                                 DONE
            Commenter --> Creator                               DONE
        Ignoring self references
        Maybe do connection decay?
            If connection already present weight = weight + X where:
                X = fresh_connection_weight + (decay_strength * (-ln(1+age_in_days)))
            Initial Values of:
                fresh_connection_weight = 1
                decay_strength = 0.25
            Give Positive weight increase up to age_in_days = 50 --> enforces freshness
            If new weight <= 0, drop the edge between contributors
        """
        social_network = nx.DiGraph()
        # For Issues, For PRs starting with oldest, extract contribs and contrib links
        # Start with oldest to allow for notion of decaying connections
        all_issues_and_prs: List[Union[IssueSM, PullRequestSM]] = sorted(
            [*self.target.prs, *self.target.issues], key=lambda entry: entry.created_at
        )

        # Prefer using names of contribs but where that fails, resort to usernames
        for entry in all_issues_and_prs:
            creator = entry.created_by.username if entry.created_by.name is None else entry.created_by.name
            if isinstance(entry, PullRequestSM):
                for reviewer in entry.requested_reviewers:
                    # Creator asked reviewer for review
                    dest_user = reviewer.username if reviewer.name is None else reviewer.name
                    social_network = self._add_or_update_social_edge(social_network, creator, dest_user)
                for review in entry.reviews:
                    # Reviewer reviewed Creator's work
                    if review.author is None:
                        continue
                    source_user = review.author.username if review.author.name is None else review.author.name
                    social_network = self._add_or_update_social_edge(social_network, source_user, creator)

            for assignee in entry.assignees:
                # Creator Assigned Assignee
                dest_user = assignee.username if assignee.name is None else assignee.name
                social_network = self._add_or_update_social_edge(social_network, creator, dest_user)

            for comment in entry.comments:
                # Commenter Talks to Creator
                commenter = comment.author.username if comment.author.name is None else comment.author.name
                social_network = self._add_or_update_social_edge(social_network, commenter, creator)

        return social_network

    def _calc_contributor_importance(self, social_network: nx.DiGraph) -> Dict[str, float]:
        importance_metric = self._IMPORTANCE_METRIC_MAP[self.config.contrib_importance_metric]
        importance_scores = importance_metric(social_network, weight="weight")
        return importance_scores

    def _add_or_update_social_edge(self, social_network, source, dest) -> nx.DiGraph:
        if source == dest:
            # Ignore self loops
            logger.debug(f"Ignored Link {source}->dest - Reason: Self Loop")
            return social_network
        if source == "Deleted user" or dest == "Deleted user":
            # Ignore deleted users - can't distinguish between them
            logger.debug(f"Ignored Link {source}->{dest} - Reason: Deleted User")
            return social_network
        if BOT_DETECT_PATTERN.search(source) or BOT_DETECT_PATTERN.search(dest):
            # Ignore Bots
            logger.debug(f"Ignored Link {source}->{dest} - Reason: Suspected Bot")
            return social_network
        if social_network.has_edge(source, dest):
            current_weight = social_network.get_edge_data(source, dest)["weight"]
            connection_weight = current_weight + self.config.fresh_connection_weight
            update_dict = {(source, dest): {"weight": connection_weight}}
            nx.set_edge_attributes(social_network, update_dict)
        else:
            connection_weight = self.config.fresh_connection_weight
            social_network.add_edge(source, dest, weight=connection_weight)
        return social_network
