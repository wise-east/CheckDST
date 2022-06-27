# Format a world logs file to use for calculating CheckDST metrics

# output should simply be a list of dictionaries with keys: "context", "dial_id", "aug_type", "gold", "pred", "requires_coref"

import json
from pathlib import Path, PosixPath
import pandas as pd
import sys
import os
from typing import Dict
from checkdst.utils import load_jsonl
from loguru import logger


def extract_target_components_from_worldlogs_line(line_item: Dict) -> Dict[str, str]:

    # import pdb; pdb.set_trace()
    turn_input = line_item["dialog"][0][0]
    turn_output = line_item["dialog"][0][1]

    # CheckDST prediction format
    extracted = {
        "context": turn_input["text"],
        "dial_id": f"{turn_input['dial_id'].replace('.json', '')}-{turn_input['turn_num']}",
        "aug_type": turn_input.get("aug_type", "Not found"),
        "gold": turn_input["eval_labels"][0],
        "pred": turn_output["text"],
        "requires_coref": "coref_jga" in turn_output["metrics"],
    }
    # context may not be necessary as it can be mapped back to the original data file

    return extracted


def format_parlai_world_logs(sample_fn: PosixPath) -> PosixPath:
    """_summary_

    Args:
        sample_fn (PosixPath): file to format into CheckDST prediction format

    Returns:
        PosixPath: path to file with CheckDST prediction format
    """
    data = load_jsonl(sample_fn)

    checkdst_formatted = [
        extract_target_components_from_worldlogs_line(d) for d in data
    ]
    df = pd.DataFrame(checkdst_formatted)

    target_fn = sample_fn.with_suffix(".checkdst_prediction.jsonl")
    df.to_json(target_fn, orient="records", lines=True)

    return target_fn


def check_and_format_parlai_pred(pred_fn: PosixPath) -> bool:
    """Check if prediction file exists and format it first if it doesn't exist in checkdst format
    Args:
        pred_fn (PosixPath): file to format into CheckDST prediction format

    Returns:
        PosixPath: path to file with CheckDST prediction format
    """

    if isinstance(pred_fn, str):
        pred_fn = Path(pred_fn)

    if not pred_fn.is_file():
        logger.warning(
            f"CheckDST format prediction file not found: {str(pred_fn)}. Attempting to format original ParlAI prediction file"
        )
        parlai_world_logs_fn = pred_fn.with_suffix("").with_suffix(".jsonl")
        if not parlai_world_logs_fn.is_file():
            logger.error(
                f"Original ParlAI prediction file not found at: {str(parlai_world_logs_fn)}. Check whether predictions were generated. "
            )
            return False
        else:
            pred_fn = format_parlai_world_logs(parlai_world_logs_fn)
            assert pred_fn.is_file()
            logger.info("Successfully formatted original prediciton file.")

    return True


if __name__ == "__main__":

    PARLAI_DIR = os.environ["PARLAI_DIR"]
    # sample_fn = (
    #     Path(PARLAI_DIR)
    #     / "models/pre_emnlp/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1_sd0/model.checkpoint_step3839.NEI_world_logs_fs_False.jsonl"
    # )
    sample_fn = Path(sys.argv[1])
    format_parlai_world_logs(sample_fn)
