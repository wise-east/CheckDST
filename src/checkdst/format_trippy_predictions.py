from cmath import e
from xml.dom import NotFoundErr
import pandas as pd
from checkdst.utils import normalize_dial_ids, ALL_SLOTS
from pathlib import Path, PosixPath
from loguru import logger
from typing import Tuple, List, Dict
import sys
import os

import glob
import sys
import os
from subprocess import run
import shlex
from tqdm import tqdm
import re

TASK = "multiwoz21"
# DATA_DIR="data/MULTIWOZ2.1"
DATA_DIR = "data/MULTIWOZ2.3"
# NOW=$(date +"%Y-%m-%d_%T")
LR = "1e-4"

# dir_ = sys.argv[1]

# parent_dir = "/data/home/justincho/CheckDST/trippy-public-master/results/of_interest"
# parent_dir = "/data/home/justincho/CheckDST/trippy-public-master/results/arxiv_results"
parent_dir = "/data/home/justincho/CheckDST/dialoglue/trippy/results"


def find_aug_type_from_fn(fn: PosixPath) -> str:
    """Find the augmentation type from a filename for trippy

    Args:
        fn (PosixPath): filename of prediction file

    Raises:
        NotFoundErr: if none of the expected augmentation types match

    Returns:
        str: augmentation type
    """
    name = fn.name
    if "dev" in name:
        return "valid"

    if "NEI" in name or "NED" in name:
        return "NED"
    elif "SD" in name or "SDI" in name:
        return "SDI"
    elif "TP" in name or "PI" in name:
        return "PI"
    elif "test" in name:
        return "orig"
    else:
        logger.error(f"Augmentation type not found in: {name}")
        raise NotFoundErr


def normalize_slot_key(given_key):

    key = given_key.lower()
    key = key.replace("book_", "")

    return key


def parse_trippy_prediction_row(csv_row: pd.Series) -> Tuple[str]:
    pred = []
    gold = []
    for slot, slot_value in csv_row.items():
        if slot_value == "none":
            continue

        if slot == "id":
            continue

        domain, slot_key = slot.split("-")
        slot_key = slot_key.replace("_gold", "").replace("_pred", "")
        slot_key = normalize_slot_key(slot_key)

        assert f"{domain}--{slot_key}" in ALL_SLOTS

        formatted_slot = f"{domain} {slot_key} {slot_value}"
        if slot.split("_")[-1] == "pred":
            pred.append(formatted_slot)
        elif slot.split("_")[-1] == "gold":
            gold.append(formatted_slot)
        else:
            logger.error("Slot is neither a prediction or gold reference: {slot}")

    def combine_formatted_slots(slots: List[str]) -> str:
        """get collected slots into the format: 'domain key value, domain key value,'

        Args:
            slots (List[str]): list of normalized slots

        Returns:
            str: properly formatted slots as a single string
        """
        return (", ".join(slots) + ", ").strip()

    pred = combine_formatted_slots(pred)
    gold = combine_formatted_slots(gold)

    # if "hotel type" in gold and "hotel type true" not in gold:
    #     import pdb; pdb.set_trace()

    return pred, gold


def format_trippy_predictions(fn: PosixPath) -> PosixPath:
    """Main function for formatting trippy prediction files into

    Args:
        fn (PosixPath): _description_

    Returns:
        PosixPath: _description_
    """

    df = pd.read_csv(fn)

    aug_type = find_aug_type_from_fn(fn)

    jsonl_items = []
    for idx, row in df.iterrows():
        # import pdb; pdb.set_trace()

        pred, gold = parse_trippy_prediction_row(row)

        dial_id = normalize_dial_ids(row["id"])

        item = {
            "context": "",
            "dial_id": dial_id,
            "aug_type": aug_type,
            "pred": pred,
            "gold": gold,
            "requires_coref": None,
        }
        jsonl_items.append(item)

    new_df = pd.DataFrame(jsonl_items)

    target_fn = fn.with_suffix(".checkdst.jsonl")
    new_df.to_json(target_fn, orient="records", lines=True)

    assert target_fn.is_file()
    logger.info(f"CheckDST compatible prediction file saved to: {str(target_fn)}")
    return target_fn


def check_and_format_trippy_pred(pred_fn: PosixPath) -> PosixPath:
    checkdst_formatted_fn = pred_fn.with_suffix(".checkdst.jsonl")
    if not checkdst_formatted_fn.is_file():
        checkdst_formatted_fn = format_trippy_predictions(pred_fn)
    return checkdst_formatted_fn


if __name__ == "__main__":
    CHECKDST_DIR = os.environ["CHECKDST_DIR"]

    fn = sys.argv[1]
    # fn = (
    #     Path(CHECKDST_DIR)
    #     / "trippy-public-master/results/of_interest/multiwoz23_lr1e-4_2022-01-05_09:19:55_fewshot_False_42/pred_res.test.2366.csv"
    # )
    target_fn = format_trippy_predictions(fn)
