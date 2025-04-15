import re
from pathlib import Path

import yaml

PATTERN_FILE_PATH = Path(__file__).parent.joinpath("third_party_paths.yml")


def load_path_regexes() -> re.Pattern:
    """
    Loads and merges the regexes defined to detect third party files in
    https://github.com/github-linguist/linguist
    Local File: analysis/shared_utils/third_party_paths.yml
    is copy of: https://github.com/github-linguist/linguist/blob/main/lib/linguist/vendor.yml
    """
    with PATTERN_FILE_PATH.open("r") as fp:
        regex_strings = yaml.safe_load(fp)
    return re.compile("(" + ")|(".join(regex_strings) + ")")
