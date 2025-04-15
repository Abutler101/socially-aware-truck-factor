from typing import Dict, List

import numpy as np
from fuzzywuzzy import fuzz
from loguru import logger

FUZZY_NAME_MAP: Dict[str, str] = dict()
FUZZY_NAME_REJECTIONS: Dict[str, str] = dict()


def contributor_in_gt(contributor_key: str, gt_names: List[str]) -> bool:
    """
    Checks if contributor_key (name-email) is in gt_names (list of readable name formatted names)
    """
    name = contributor_key.split("-")[0]
    if name.title() in gt_names:
        return True
    if len([n for n in gt_names if name.title() in n]) > 0:
        return True

    # Look for a fuzzy match as last resort
    return _fuzzy_name_check(name, gt_names)


def _fuzzy_name_check(name: str, names: List[str]) -> bool:
    """runs a manually validated fuzzy search for name in names"""
    if len(names) == 0:
        return False

    fuzzy_confidences = [fuzz.partial_ratio(name, entry) for entry in names]
    most_likely_match = names[np.argmax(fuzzy_confidences)]
    if max(fuzzy_confidences) < 40:  # Don't waste time with similarity under 40%
        return False
    if name in FUZZY_NAME_REJECTIONS and FUZZY_NAME_REJECTIONS[name].lower() == most_likely_match.lower():
        logger.debug(f"Closest fuzzy match of: {name} ~= {most_likely_match} was previously rejected")
        return False
    if name in FUZZY_NAME_MAP and FUZZY_NAME_MAP[name].lower() == most_likely_match.lower():
        logger.debug(f"Found previously accepted fuzzy match: {name} ~= {most_likely_match}")
        return True

    logger.info(f"Is {name} ~= {most_likely_match}")
    accept_match = input("y/n >")
    if accept_match.lower() == "y":
        logger.info(f"fuzzy match accepted")
        FUZZY_NAME_MAP[name] = most_likely_match
        return True
    logger.info(f"fuzzy match rejected")
    FUZZY_NAME_REJECTIONS[name] = most_likely_match
    return False


def name_in_estim_output(contributor_name: str, estim_out: List[str]) -> bool:
    """
    Checks if contributor_name (readable name format) is in esim_out (list of contributor keys)
    """
    estim_names = [key.split("-")[0].title() for key in estim_out]
    if contributor_name in estim_names:
        return True
    if len([n for n in estim_names if (n in contributor_name or contributor_name in n)]) > 0:
        return True
    # Look for a fuzzy match as last resort
    return _fuzzy_name_check(contributor_name, estim_names)
