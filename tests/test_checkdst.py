# test with `pytest` in `tests` directory

import json
from pathlib import Path
from checkdst import CheckDST
from checkdst.format_parlai_world_logs import format_parlai_world_logs
import os

DATAPATH = os.environ.get("DATAPATH", "../data")
CHECKDST_DIR = os.environ.get("CHECKDST_DIR", "../data")


# calculate checkdst metrics with single directory


def test_checkdst_for_single_dir():
    gold_data_fn = (
        Path(DATAPATH) / "checkdst/NED/data_reformat_official_v2.3_slots_test.json"
    )
    assert (
        gold_data_fn.is_file()
    ), "Make sure that you have set up the data with CheckDST/data/prepare_multiwoz_dst.sh"

    pred_fn = (
        Path(CHECKDST_DIR)
        / "tests/example_data/ckpt_step928_multiwoz_checkdst:{}_example.checkdst_prediction.jsonl"
    )

    checkdst = CheckDST(gold_data_fn=gold_data_fn)
    for aug_type in ["orig", "NED", "SDI", "PI"]:
        print(str(pred_fn).format(aug_type))
        checkdst.add_preds(str(pred_fn).format(aug_type), aug_type=aug_type)

    # calculate cjga
    for aug_type in ["NED", "SDI", "PI"]:
        checkdst.compute_cjga(orig="orig", aug=aug_type)

    print(checkdst)
    assert checkdst.get_results()["jga_orig"] == 0.36


def test_checkdst_without_gold_data():

    pred_fn = (
        Path(CHECKDST_DIR)
        / "tests/example_data/ckpt_step928_multiwoz_checkdst:{}_example.checkdst_prediction.jsonl"
    )

    checkdst = CheckDST()
    for aug_type in ["orig", "NED", "SDI", "PI"]:
        checkdst.add_preds(str(pred_fn).format(aug_type), aug_type=aug_type)

    # calculate cjga
    for aug_type in ["NED", "SDI", "PI"]:
        checkdst.compute_cjga(orig="orig", aug=aug_type)

    print(checkdst)
    assert checkdst.get_results()["jga_orig"] == 0.36


def test_format_parlai_world_logs():

    sample_fn = (
        Path(CHECKDST_DIR) / "tests/example_data/example_parlai_world_logs.jsonl"
    )

    target_fn = format_parlai_world_logs(sample_fn)
    assert target_fn.is_file()


if __name__ == "__main__":
    test_checkdst_for_single_dir()
    test_format_parlai_world_logs()
    test_checkdst_without_gold_data()
