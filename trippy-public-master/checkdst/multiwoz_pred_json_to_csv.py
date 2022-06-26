# To transform json prediction files to csv files in case if it had not already been done from fine-tuning.
# u

import glob
import sys
import os
from subprocess import run
import shlex
from tqdm import tqdm
import re
from pathlib import Path

TASK = "multiwoz21"


def transform_trippy_predictions_from_json_to_csv(subdirs):
    ct = 0
    for dir_ in subdirs:
        for split in ["test", "dev"]:
            fps = glob.glob(os.path.join(dir_, f"pred_res.{split}*.json"))

            for fp in tqdm(fps):
                print(fp)
                csv_file = fp.replace(".json", ".csv")
                if Path(csv_file).is_file():
                    print("found csv")
                    # # # force it for some subset
                    # if "NEI" in fp or "SD" in fp or "TP" in fp:
                    #     continue
                    continue
                # continue
                command = (
                    f"sbatch multiwoz_pred_json_to_csv_batch_parallel.sh {TASK} {fp}"
                )

                # command = f"python3 metric_bert_dst.py {TASK} dataset_config/{TASK}.json {fp}"
                print(command)
                run(shlex.split(command))
                ct += 1
                # if ct == 10:
                #     return

    return


if __name__ == "__main__":

    # parent_dir = "/data/home/justincho/CheckDST/dialoglue/trippy/results/emnlp"
    parent_dir = "/data/home/justincho/CheckDST/trippy-public-master/results/emnlp"

    subdirs = glob.glob(os.path.join(parent_dir, "*"))

    transform_trippy_predictions_from_json_to_csv(subdirs)
