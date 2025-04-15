import ast
import copy
import re
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
from loguru import logger

INPUT_PATH = Path(__file__).parents[1].joinpath("analysis/output/estimator_results")
OUTPUT_PATH = Path(__file__).parent.joinpath("output")
LOG_PATH = OUTPUT_PATH.joinpath("logs")


@logger.catch
def main():
    result_file_paths = list(INPUT_PATH.glob("*.csv"))
    cache: Dict[str, List[int]] = dict()
    for result_file_path in result_file_paths:
        logger.info(f"━━━━━━ Post Processing {result_file_path.name} ━━━━━━")
        results_before = pd.read_csv(result_file_path, index_col="estimator")

        results_after: Dict[str, pd.Series] = dict()
        result_row: Tuple[str, pd.Series]
        for result_row in results_before.iterrows():
            contributors: List[str] = ast.literal_eval(result_row[1].tf_contributors)
            if len(contributors) <= 1:
                copied_series = copy.deepcopy(result_row[1])
                copied_series.tf_contributors = ast.literal_eval(copied_series.tf_contributors)
                results_after[result_row[0]] = copied_series
                continue
            # Take and Parse user input for which idxs to drop
            fmtd_contrib_list = "".join([f"{idx}: {contrib}\n" for idx, contrib in enumerate(contributors)])
            if fmtd_contrib_list in cache:
                logger.info(f"Post Process for combo of contribs has already been done")
                contrib_idxs_to_drop = cache[fmtd_contrib_list]
            else:
                logger.info(f"Insert Comma seperated list of contributors to drop")
                logger.info(f"Truck Factor Contributors detected:\n{fmtd_contrib_list}")
                raw_input = input(f"Comma seperated list of contributor numbers to drop. Leave blank to not drop any\n")
                contrib_idxs_to_drop = parse_input(raw_input, len(contributors)-1)
                cache[fmtd_contrib_list] = contrib_idxs_to_drop

            # Drop targeted idxs and build results df
            processed_contributors = filter_contribs(contributors, contrib_idxs_to_drop)
            modified_result_series = copy.deepcopy(result_row[1])
            modified_result_series.tf_contributors = processed_contributors
            modified_result_series.tf = len(processed_contributors)
            results_after[result_row[0]] = modified_result_series

        post_processed_df = pd.DataFrame.from_dict(results_after, orient="index")
        post_processed_df.to_csv(
            OUTPUT_PATH.joinpath(f"cleaned-{result_file_path.name}"),
            index_label="estimator"
        )
        logger.info(f"━━━━━━ Cleaned File Saved as cleaned-{result_file_path.name} ━━━━━━")


def parse_input(raw_input, max_idx) -> List[int]:
    COMMA_SEPERATED_NUMS = re.compile("^\d+(,\d)*")
    if len(raw_input) == 0:
        logger.info(f"No idxs specified to drop")
        return []
    no_spaces = raw_input.replace(" ", "")
    if not COMMA_SEPERATED_NUMS.match(no_spaces):
        logger.info(f"Raw Input {raw_input} is unexpected format. Ignoring")
        return []
    to_drop = [int(val) for val in no_spaces.split(",") if int(val) <= max_idx]
    logger.info(f"Will drop idxs: {to_drop}")
    return to_drop


def filter_contribs(contributors: List[str], idxs_to_drop: List[int]):
    filtered = [contrib for idx, contrib in enumerate(contributors) if idx not in idxs_to_drop]
    return filtered


if __name__ == '__main__':
    OUTPUT_PATH.mkdir(exist_ok=True)
    LOG_PATH.mkdir(exist_ok=True)
    logger.add(LOG_PATH.joinpath("{time}.log"), rotation="5h")
    main()
