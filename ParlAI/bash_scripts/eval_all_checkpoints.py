# Usage: python eval_all_checkpoints <dir with model checkpoints>

from subprocess import run
import sys
import glob
import shlex
import re
import time
from pathlib import Path
import json
import os


def eval_checkpoints_in_dir(main_dir):
    ct = 0
    fn_list = glob.glob(f"{main_dir}/**checkpoint**")

    fn_list = [f for f in fn_list if re.search('(.*checkpoint_step[0-9]+$)', f)]
    # fn_list = glob.glob(f"{main_dir}/model")

    for f in fn_list:
        print(f)

    # return

    for ckpoint in fn_list:
        if "fs_True" in ckpoint or "fewshot_True" in ckpoint:
            is_fewshot = True
        else:
            is_fewshot = False

        for split in ["valid", "test"]:
            eval_cmd = f"sbatch /data/home/justincho/ParlAI/bash_scripts/eval_checkpoint.sh -c {ckpoint} -d {split} -t False -f {is_fewshot}"

            report_file = ckpoint + ".test_report"
            if not os.path.isfile(report_file):
                run(shlex.split(eval_cmd))
                ct += 1
            else:
                with open(report_file, "r") as f:
                    report = json.load(f)

                if report["report"]["joint goal acc"] < 0.1:
                    ct += 1
                    run(shlex.split(eval_cmd))

        for aug in ["NEI", "SD", "TP"]:
            inv_cmd = f"sbatch /data/home/justincho/ParlAI/bash_scripts/evaluate_laug_invariance.sh -m {ckpoint} -i {aug} -f {is_fewshot}"

            report_file = ckpoint + f".{aug}_report_fs_{is_fewshot}.json"
            if not os.path.isfile(report_file):
                run(shlex.split(inv_cmd))
                ct += 1
            else:
                with open(report_file, "r") as f:
                    report = json.load(f)

                if report["report"]["jga_original"] < 0.1:
                    ct += 1
                    run(shlex.split(inv_cmd))
    print(ct)


def main():
    # main_dir = sys.argv[1]  # pass main directory

    # subdirs = glob.glob("/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/*")
    # subdirs = glob.glob(
    #     "/data/home/justincho/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.3/*"
    # )
    subdirs = glob.glob("/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.3/*")
    for sd in sorted(subdirs):
        # if sd != "/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/lr5e-05_ngpu1_bs4_uf1_fewshot_True_prompts_True_sd4_2022-01-04_05:17:22":
        print(sd)
        eval_checkpoints_in_dir(sd)
    pass

    # eval_checkpoints_in_dir()


if __name__ == "__main__":
    main()
