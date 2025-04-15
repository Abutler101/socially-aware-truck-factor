import hashlib
import time
from pathlib import Path
from typing import Dict

import pandas as pd
from loguru import logger

from analysis.estimators import ESTIMATORS, TruckFactorEstimator
from shared_models.analysis_target import AnalysisTarget
from shared_models.analysis_window import AnalysisWindow
from analysis.models.dataset import Dataset
from shared_models.repo_data import RepoData

TARGETS_LIST_PATH = Path(__file__).parent.joinpath("extended-targets.json")
OUTPUT_PATH = Path(__file__).parent.joinpath("output")
ESTIMATOR_RESULTS_PATH = OUTPUT_PATH.joinpath("estimator_results")
LOG_PATH = OUTPUT_PATH.joinpath("logs")


def load_target_list() -> Dataset:
    """
    The Paths specified in targets.json are assumed to be relative to
    the const DATASET_PATH defined in: analysis/models/dataset.py
    """
    dataset = Dataset.model_validate_json(TARGETS_LIST_PATH.open("r").read())
    return dataset


def run_estimators(target: AnalysisTarget, target_name: str) -> pd.DataFrame:
    logger.info(f"Running Common Prep work for {target_name}")
    # Do the compute heavy tasks that likely all estimators will need the results from
    target.identify_unique_contributors()
    target.group_commits_by_file()
    target.group_prs_by_file()
    target.group_issues_by_file()

    logger.info(f"Running Estimators Against: {target_name}")
    estimator: TruckFactorEstimator
    results: Dict[str, Dict] = dict()
    compute_perf: Dict[str, int] = dict()
    for Estimator, config in ESTIMATORS:
        estimator_key = f"{Estimator.__name__}-{hashlib.md5(str(config.__dict__.values()).encode('utf8')).hexdigest()}"
        logger.info(f"━━━━━━ Running {Estimator.__name__} ━━━━━━")
        logger.info(f"Config: {config}")
        estimator = Estimator(config)
        start = time.perf_counter_ns()
        estimator.load_project(target)
        result = estimator.run_estimation()
        stop = time.perf_counter_ns()
        compute_perf[estimator_key] = stop-start

        logger.info(f"{Estimator.__name__} Result: TF={result[0]} | TF-Contributors={result[1]}")
        results[estimator_key] = {
            "estim_config": config,
            "tf": result[0],
            "tf_contributors": result[1]
        }
    pd.DataFrame.from_dict(compute_perf, orient="index").to_json(
        OUTPUT_PATH.joinpath(f"{target_name}-rt_ns.json"), index=True
    )
    return pd.DataFrame.from_dict(results, orient="index")


@logger.catch
def main():
    target_list = load_target_list()
    for target_entry in target_list.targets:
        repo_name = target_entry.path.name.split(".")[0]
        logger.info(f"━━━━━━━━━━━━ Loading Repo Data for: {repo_name} ━━━━━━━━━━━━")
        repo_data = RepoData.model_validate_json(target_entry.path.open("r").read())
        analysis_window = AnalysisWindow.from_dates(start=None, end=target_entry.end_date)
        analysis_target = AnalysisTarget.from_repo_data(repo_name, repo_data, analysis_window)
        estimator_results = run_estimators(analysis_target, target_entry.path.name)
        estimator_results.to_csv(
            ESTIMATOR_RESULTS_PATH.joinpath(f"{target_entry.path.name}.csv"),
            index_label="estimator"
        )


if __name__ == '__main__':
    OUTPUT_PATH.mkdir(exist_ok=True)
    LOG_PATH.mkdir(exist_ok=True)
    ESTIMATOR_RESULTS_PATH.mkdir(exist_ok=True)
    logger.add(LOG_PATH.joinpath("{time}.log"), rotation="5h")
    main()
