from typing import List

import numpy as np

from .avelino import AvelinoEstimator, AvelinoConfig
from .cosentino import CosentinoEstimator, CosentinoConfig, KnowledgeMetric
from .extended_dok import EDoKEstimator, EDoKConfig
from .haratian import HaratianEstimator, HaratianConfig, FileImportanceMetric
from .social_graph import SocialGraphEstimator, SocialGraphConfig, ImportanceMetric, ThresholdMode

from .base_estimator import TruckFactorEstimator, EstimatorConfig


ESTIMATORS_BASELINE: List[tuple] = [
    # ━━━━━━ Baseline Estimators ━━━━━━ #
    (AvelinoEstimator, AvelinoConfig()),
    (CosentinoEstimator, CosentinoConfig(knowledge_metric=KnowledgeMetric.M1)),
    (CosentinoEstimator, CosentinoConfig(knowledge_metric=KnowledgeMetric.M2)),
    (CosentinoEstimator, CosentinoConfig(knowledge_metric=KnowledgeMetric.M3)),
    (CosentinoEstimator, CosentinoConfig(knowledge_metric=KnowledgeMetric.M4)),
    (HaratianEstimator, HaratianConfig(file_importance_metric=FileImportanceMetric.BET_CENT)),
    (HaratianEstimator, HaratianConfig(file_importance_metric=FileImportanceMetric.PAGE_RANK)),
    (HaratianEstimator, HaratianConfig(file_importance_metric=FileImportanceMetric.IN_DEG)),
    (HaratianEstimator, HaratianConfig(file_importance_metric=FileImportanceMetric.OUT_DEG)),
    (HaratianEstimator, HaratianConfig(file_importance_metric=FileImportanceMetric.DEG_CENT)),
]


ESTIMATORS_EDOK_OPTIMIZATION: List[tuple] = []
for pcrw in np.arange(2.0, 10.0, 1.0):
    for pcmw in np.arange(5.0, 15.0, 1.0):
        for icrw in np.arange(2.0, 10.0, 1.0):
            for icmw in np.arange(5.0, 15.0, 1.0):
                ESTIMATORS_EDOK_OPTIMIZATION.append(
                    (EDoKEstimator, EDoKConfig(
                        num_prs_created_weight=pcrw,
                        num_pr_cmnts_made_weight=pcmw,
                        num_issues_created_weight=icrw,
                        num_issue_cmnts_made_weight=icmw
                    )),
                )

ESTIMATORS_SOCIAL_GRAPH_INITIAL_OPTIMIZATION: List[tuple] = []
for importance_measure in ImportanceMetric:
    for threshold in np.arange(0.1,1.0,0.05):
        ESTIMATORS_SOCIAL_GRAPH_INITIAL_OPTIMIZATION.append(
            (SocialGraphEstimator, SocialGraphConfig(
                contrib_importance_metric=importance_measure,
                importance_threshold=threshold
            ))
        )


ESTIMATORS_SOCIAL_GRAPH_SECOND_OPTIMIZATION: List[tuple] = [
    # Most Accurate Config from initial optimization - acc = 0.5769
    (SocialGraphEstimator, SocialGraphConfig(
        contrib_importance_metric=ImportanceMetric.DEG_CENT, importance_threshold=0.8, threshold_mode=ThresholdMode.ABS
    )),
]
for importance_floor in np.arange(0.1, 1.0, 0.05):
    for top_x_pct in np.arange(0.1, 0.9, 0.05):
        ESTIMATORS_SOCIAL_GRAPH_SECOND_OPTIMIZATION.append(
            (SocialGraphEstimator, SocialGraphConfig(
                contrib_importance_metric=ImportanceMetric.DEG_CENT,
                importance_threshold=importance_floor,
                threshold_pct=top_x_pct,
                threshold_mode=ThresholdMode.PERC
            )),
        )

ESTIMATORS_FINAL_EVALUATION: List[tuple] = [
    (AvelinoEstimator, AvelinoConfig()),
    (HaratianEstimator, HaratianConfig(file_importance_metric=FileImportanceMetric.DEG_CENT)),
    (EDoKEstimator, EDoKConfig(
        num_prs_created_weight=5.0,
        num_pr_cmnts_made_weight=10.0,
        num_issues_created_weight=10.0,
        num_issue_cmnts_made_weight=5.0
    )),
    (SocialGraphEstimator, SocialGraphConfig(
        contrib_importance_metric=ImportanceMetric.DEG_CENT,
        importance_threshold=0.75,
        threshold_pct=0.55,
        threshold_mode=ThresholdMode.PERC
    ))
]
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ SET WHICH ESTIMATOR LIST TO RUN ━━━━━━━━━━━━━━━━━━━━━━━━━━━ #
ESTIMATORS: List[tuple] = ESTIMATORS_FINAL_EVALUATION
