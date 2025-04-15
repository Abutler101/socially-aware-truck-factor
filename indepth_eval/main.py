import ast
import hashlib
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from indepth_eval.models import ContribClassificationAnalysis, EstimatorPerf, EstimatorPerfContainer
from indepth_eval.utils import contributor_in_gt, name_in_estim_output, FUZZY_NAME_MAP, FUZZY_NAME_REJECTIONS
from shared_models.ground_truth import GroundTruth


GROUND_TRUTH_PATH = Path(__file__).parents[1].joinpath("truck_factors.json")
ESTIMATOR_RESULTS_PATH = Path(__file__).parents[1].joinpath("post_processing/output")
OUTPUT_PATH = Path(__file__).parent.joinpath("output")
LOG_PATH = OUTPUT_PATH.joinpath("logs")


@logger.catch
def main():
    ground_truth = GroundTruth.model_validate_json(GROUND_TRUTH_PATH.open("r").read())
    estim_perfs: EstimatorPerfContainer = EstimatorPerfContainer(project_count=ground_truth.project_count)

    for project_estim_outputs in list(ESTIMATOR_RESULTS_PATH.glob("*.csv")):
        target_project = project_estim_outputs.name.replace("cleaned-", "").replace(".json.csv", "")
        logger.info(f"━━━━━━ Loaded Estim Outputs for {target_project} ━━━━━━")
        estimator_outputs_raw = pd.read_csv(project_estim_outputs).to_dict()
        parsed_estimator_outputs = {k: v for k, v in estimator_outputs_raw.items() if k != "tf_contributors"}
        parsed_estimator_outputs["tf_contributors"] = dict()
        for idx, val in estimator_outputs_raw["tf_contributors"].items():
            parsed_estimator_outputs["tf_contributors"][idx] = ast.literal_eval(val)
        estimator_outputs = pd.DataFrame.from_dict(parsed_estimator_outputs)
        project_ground_truth = ground_truth.get_project(target_project)
        p_gt_names = project_ground_truth.truck_factor_contributors

        for _, row in estimator_outputs.iterrows():
            estim_name = row.estimator.split("-")[0]
            config_string = row.estim_config if isinstance(row.estim_config, str) else "NO-CONFIG"
            estim_id: str = estim_name + "-" + hashlib.md5(config_string.encode("utf8")).hexdigest()
            estim_tf: int = row.tf
            estim_contribs: List[str] = row.tf_contributors
            perf_data = EstimatorPerf()
            if estim_id in estim_perfs.performance_entries:
                perf_data = estim_perfs.performance_entries[estim_id]

            # Calculate Norm'd Abs Error
            nae = abs(project_ground_truth.truck_factor - estim_tf) / project_ground_truth.truck_factor

            # Do classification Analysis
            if len(p_gt_names) == 0 and project_ground_truth.truck_factor > 0:
                tp_contribs = []
                fp_contribs = []
                fn_contribs = []
            elif len(p_gt_names) != project_ground_truth.truck_factor:
                # Can't work with partial gt data
                tp_contribs = []
                fp_contribs = []
                fn_contribs = []
            else:
                tp_contribs: List[str] = [
                    contrib for contrib in estim_contribs if contributor_in_gt(contrib, p_gt_names)
                ]
                fp_contribs: List[str] = [
                    contrib for contrib in estim_contribs if not contributor_in_gt(contrib, p_gt_names)
                ]
                fn_contribs: List[str] = [
                    name for name in p_gt_names if not name_in_estim_output(name, estim_contribs)
                ]
            classification_analysis = ContribClassificationAnalysis(
                true_positive_contribs=tp_contribs,
                false_positive_contribs=fp_contribs,
                false_negative_contribs=fn_contribs,
            )

            perf_data.estimate_breakdowns[target_project] = (nae, classification_analysis)
            estim_perfs.performance_entries[estim_id] = perf_data

    estim_perfs.calc_metrics()
    with OUTPUT_PATH.joinpath(f"estimator-performance.json").open("w") as f:
        f.write(estim_perfs.model_dump_json(indent=4))

    estim_perfs.plot_classification_metrics(OUTPUT_PATH)


if __name__ == '__main__':
    OUTPUT_PATH.mkdir(exist_ok=True)
    LOG_PATH.mkdir(exist_ok=True)
    logger.add(LOG_PATH.joinpath("{time}.log"), rotation="5h")
    main()
