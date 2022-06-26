# calculate checkdst metrics with single file

import json
from pathlib import Path, PosixPath
from checkdst import CheckDST
from checkdst.utils import find_seed
from checkdst.parlai_utils import find_all_checkpoints, find_epoch_for_checkpoint
from checkdst.format_parlai_world_logs import check_and_format_parlai_pred
from checkdst.format_trippy_predictions import format_trippy_predictions
from pprint import pprint
from tqdm import tqdm
import os
from loguru import logger
from collections import defaultdict
from typing import Dict, List, Tuple
import re
import pickle

DATAPATH = os.environ.get("DATAPATH", "../../data")
PARLAI_DIR = os.environ.get("PARLAI_DIR", "../../ParlAI")
CHECKDST_DIR = os.environ.get("CHECKDST_DIR", "../../ParlAI")

GOLD_DATA_FN = (
    Path(DATAPATH) / "checkdst/NED/data_reformat_official_v2.3_slots_test.json"
)

PARLAI_TEST_PREDICTIONS_SUFFIX = ".test_world_logs_multiwoz_checkdst:augmentation_method={}.checkdst_prediction.jsonl"
PARLAI_VALID_PREDICTIONS_SUFFIX = ".valid_world_logs.checkdst_prediction.jsonl"

AUGTYPES = ["orig", "NED", "SDI", "PI"]
EPOCHS = [0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]
# EPOCHS = [ 0.5, 0.75, 1, 1.5, 2, 5, 10]
# for handling any legacy names used
AUG_MAP = {"NEI": "NED", "TP": "PI", "SD": "SDI"}

TRIPPY_STEPS2EPOCH_RATIO = 118 / 0.05


def find_trippy_pred_files(dir_: PosixPath) -> List[PosixPath]:
    pred_files = [
        fn
        for fn in dir_.glob("*")
        if "final" not in str(fn)  # redundant
        and re.search("pred_res\..*\.csv$", str(fn))
    ]

    return pred_files


def get_all_trippy_checkdst_results(
    main_dir: PosixPath, epochs: List[float] = EPOCHS, print_=False, force_reload=False
) -> Tuple[Dict, Dict]:

    all_objs, all_results = None, None
    # load if previously computed and saved already
    obj_pickle_path = main_dir / "all_objs.pkl"
    results_pickle_path = main_dir / "all_results.pkl"
    if not force_reload:
        if obj_pickle_path.is_file():
            logger.info("Found pickled objs file. Loading...")
            with obj_pickle_path.open("rb") as f:
                all_objs = pickle.load(f)
        if results_pickle_path.is_file():
            logger.info("Found pickled results file. Loading...")
            with results_pickle_path.open("rb") as f:
                all_results = pickle.load(f)

        if all_objs is not None and all_results is not None:
            logger.info("Loaded both necessary files")
            return all_objs, all_results

    all_objs = defaultdict(dict)
    all_results = defaultdict(dict)
    runs = list(main_dir.glob("*"))
    # iterate through runs (differentiated by seed values)
    logger.info(f"Found {len(runs)} runs at {main_dir.name}")
    for run in tqdm(runs):
        if run.suffix == ".pkl":
            continue
        run_config = run.name
        seed = find_seed(run_config)
        if seed is None:
            seed = run_config.split("_")[-1]
        try:
            int(seed)
        except Exception as e:
            import pdb

            pdb.set_trace()
            logger.error(e)

        def find_step_in_name(fn: str) -> int:
            return int(re.sub("[^0-9]", "", fn))

        pred_files = list(find_trippy_pred_files(run))
        # example file format: */parent_dir/pred_res.dev.118.csv
        logger.info(f"Found {len(pred_files)} prediction files at {run.name}")

        steps = sorted([find_step_in_name(pred_fn.name) for pred_fn in pred_files])
        steps = list(set(steps))
        for step in steps:
            checkdst = CheckDST(gold_data_fn=GOLD_DATA_FN)
            # map steps to epochs
            epoch = round(step / TRIPPY_STEPS2EPOCH_RATIO, 2)
            for ep in epochs:
                if abs(ep - epoch) <= 0.05:
                    epoch = ep
            print(epoch, seed)

            if epoch not in epochs:
                logger.error(f"Epoch {epoch} not found {epochs}")
                # import pdb; pdb.set_trace()
                continue

            pred_fns = [fn for fn in pred_files if step == find_step_in_name(fn.name)]

            for pred_fn in pred_fns:
                name = pred_fn.name

                aug_type = re.sub("[0-9\.]", "", Path(name).with_suffix("").suffix)
                if aug_type == "":
                    if "test" in name:
                        aug_type = "orig"
                    else:
                        aug_type = "valid"
                else:
                    # augmentation type must be mapped to an expected augmentation type
                    aug_type = AUG_MAP.get(aug_type, aug_type)
                    if aug_type not in AUG_MAP.values():
                        logger.error(
                            f"Augmentation type: {aug_type} is not expected. Please make sure it maps to one of {AUG_MAP.values()}"
                        )
                        import pdb

                        pdb.set_trace()

                checkdst_formatted_fn = pred_fn.with_suffix(".checkdst.jsonl")
                if not checkdst_formatted_fn.is_file():
                    checkdst_formatted_fn = format_trippy_predictions(pred_fn)
                checkdst.add_preds(checkdst_formatted_fn, aug_type=aug_type)
            # calculate cjga
            for aug_type in AUG_MAP.values():
                checkdst.compute_cjga(orig="orig", aug=aug_type)

            # for qualitative analysis
            all_objs[epoch][seed] = checkdst

            # for main results
            all_results[epoch][seed] = checkdst.get_results()
            if print_:
                pprint(checkdst.get_results())

    logger.info("Pickling both objs and results files...")
    with obj_pickle_path.open("wb") as f:
        pickle.dump(all_objs, f)
    with results_pickle_path.open("wb") as f:
        pickle.dump(all_results, f)
    logger.info("Successfully pickled both objs and results files.")

    return all_objs, all_results


def get_all_parlai_checkdst_results(
    main_dir: PosixPath, epochs: List[float] = EPOCHS, print_=False, force_reload=False
) -> Tuple[Dict, Dict]:

    all_objs, all_results = None, None
    # load if previously computed and saved already
    obj_pickle_path = main_dir / "all_objs.pkl"
    results_pickle_path = main_dir / "all_results.pkl"

    if not force_reload:
        if obj_pickle_path.is_file():
            logger.info("Found pickled objs file. Loading...")
            with obj_pickle_path.open("rb") as f:
                all_objs = pickle.load(f)
        if results_pickle_path.is_file():
            logger.info("Found pickled results file. Loading...")
            with results_pickle_path.open("rb") as f:
                all_results = pickle.load(f)

        if all_objs is not None and all_results is not None:
            logger.info("Loaded both necessary files")
            return all_objs, all_results

    if not force_reload and (all_objs is None or all_results is None):
        logger.info("Previous load not found. Loading...")
    else:
        logger.info("Force reloading...")

    all_objs = defaultdict(dict)
    all_results = defaultdict(dict)
    runs = list(main_dir.glob("*"))
    # iterate through runs (differentiated by seed values)
    logger.info(f"Found {len(runs)} runs at {main_dir.name}")
    for run in tqdm(runs):
        if run.suffix == ".pkl":
            continue
        run_config = run.name
        seed = find_seed(run)
        try:
            int(seed)
        except Exception as e:
            import pdb

            pdb.set_trace()
            logger.error(e)

        checkpoints = list(find_all_checkpoints(run))
        # iterate through checkpoints
        logger.info(f"Found {len(checkpoints)} checkpoints at {run_config}")
        for ckpt in tqdm(sorted(list(checkpoints))):
            ckpt_name = ckpt.name
            epoch = find_epoch_for_checkpoint(ckpt)
            # only compute for epochs of interest
            if epoch not in epochs:
                continue
            logger.info(f"Processing epoch {epoch}...")

            checkdst = CheckDST(gold_data_fn=GOLD_DATA_FN)

            for aug_type in AUGTYPES:
                pred_fn = str(ckpt) + PARLAI_TEST_PREDICTIONS_SUFFIX.format(aug_type)
                # format it first if checkpoint prediction files doesn't exist in checkdst format
                if check_and_format_parlai_pred(pred_fn):
                    checkdst.add_preds(pred_fn, aug_type=aug_type)

            # add validation set results
            val_pred_fn = str(ckpt) + PARLAI_VALID_PREDICTIONS_SUFFIX
            if check_and_format_parlai_pred(val_pred_fn):
                checkdst.add_preds(val_pred_fn, aug_type="valid")

            # calculate cjga
            for aug_type in AUG_MAP.values():
                checkdst.compute_cjga(orig="orig", aug=aug_type)

            # for qualitative analysis
            all_objs[epoch][seed] = checkdst

            # for main results
            all_results[epoch][seed] = checkdst.get_results()
            if print_:
                pprint(checkdst.get_results())

    logger.info("Pickling both objs and results files...")
    with obj_pickle_path.open("wb") as f:
        pickle.dump(all_objs, f)
    with results_pickle_path.open("wb") as f:
        pickle.dump(all_results, f)
    logger.info("Successfully pickled both objs and results files.")

    return all_objs, all_results


if __name__ == "__main__":
    trippy_dir = Path(CHECKDST_DIR) / "trippy-public-master/results/emnlp/"
    _, _ = get_all_trippy_checkdst_results(trippy_dir, print_=True, force_reload=True)

    trippy_dir = Path(CHECKDST_DIR) / "dialoglue/trippy/results/emnlp"
    _, _ = get_all_trippy_checkdst_results(trippy_dir, print_=True, force_reload=True)

    # main_dir=Path(PARLAI_DIR) / "models/bart_pft_multiwoz2.3/"
    # assert main_dir.is_dir()
    # scratch_dir=Path(PARLAI_DIR) / "models/bart_scratch_multiwoz2.3/"
    # assert scratch_dir.is_dir()
    # soloist_dir=Path(PARLAI_DIR) / "models/bart_soloist_multiwoz2.3/"
    # assert soloist_dir.is_dir()
    # muppet_dir=Path(PARLAI_DIR) / "models/bart_muppet_multiwoz2.3/"
    # assert muppet_dir.is_dir()

    # pft_objs, pft_results = get_all_parlai_checkdst_results(main_dir, force_reload=True)
    # scratch_objs, scratch_results = get_all_parlai_checkdst_results(scratch_dir, force_reload=True)
    # soloist_objs, soloist_results = get_all_parlai_checkdst_results(soloist_dir, force_reload=True)
    # muppet_objs, muppet_results = get_all_parlai_checkdst_results(muppet_dir, force_reload=True)
# plot any diagrams of interest (trippy vs bart-dst)

# extract consistent JGA examples

# perturbed JGA vs consistent JGA comparison

# performan any qualitative anlaysis and add any additional metrics to checkdst
