import glob
import sys
import os 
from subprocess import run 
import shlex 
from tqdm import tqdm 
import re

TASK="multiwoz21"
# DATA_DIR="data/MULTIWOZ2.1"
DATA_DIR="data/MULTIWOZ2.3"
# NOW=$(date +"%Y-%m-%d_%T")
LR="1e-4"

# dir_ = sys.argv[1]

# parent_dir = "/data/home/justincho/trippy-public-master/results/of_interest"
# parent_dir = "/data/home/justincho/trippy-public-master/results/arxiv_results"
parent_dir = "/data/home/justincho/dialoglue/trippy/results"


subdirs =glob.glob(os.path.join(parent_dir, "*")) 

for dir_ in subdirs: 
    # fps = glob.glob(os.path.join(dir_, "pred_res.test*.json"))
    fps = glob.glob(os.path.join(dir_, "pred_res.dev*.json"))

    for fp in tqdm(fps): 
        print(fp)
        if os.path.isfile(fp.replace(".json", ".csv")): 
            print("found csv")
            # # # force it for some subset 
            # if "NEI" in fp or "SD" in fp or "TP" in fp: 
            #     continue    
            continue 
        # continue 
        command = f"sbatch multiwoz_pred_json_to_csv_batch_parallel.sh {TASK} {fp}"

        # command = f"python3 metric_bert_dst.py {TASK} dataset_config/{TASK}.json {fp}" 
        print(command)
        run(shlex.split(command))

    # break