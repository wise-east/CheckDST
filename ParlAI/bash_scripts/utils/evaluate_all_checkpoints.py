# Usage: python eval_all_checkpoints <dir with model checkpoints>
# Main purpose: convenience script for submitting evaluation jobs for checkpoints that were problematic or that were not evaluated 

from subprocess import run
import sys
import glob
import shlex
import re
import json
import os
from loguru import logger 

PARLAI_DIR = os.environ.get("PARLAI_DIR", "")

def eval_checkpoints_in_dir(main_dir):
    n_submitted_jobs = 0
    fn_list = glob.glob(f"{main_dir}/**checkpoint**")

    fn_list = [f for f in fn_list if re.search('(.*checkpoint_step[0-9]+$)', f)]

    for f in fn_list:
        logger.info(f"Checkpoint: {f}")

    for ckpoint in fn_list:
        if "fs_True" in ckpoint or "fewshot_True" in ckpoint:
            is_fewshot = True
        else:
            is_fewshot = False
        logger.info(f"Evaluating in Fewshot = {is_fewshot} setting")

        logger.info("Submitting original valid & test set evaluations...")
        for split in ["valid", "test"]:
            script_fn = os.path.join(PARLAI_DIR, "bash_scripts/evaluate_original.sh")
            eval_cmd = f"sbatch {script_fn} -c {ckpoint} -d {split} -t False -f {is_fewshot}"
            eval_exec_msg = "Running evaluation with `{ckpoint}` on {split} set fewshot={is_fewshot}" 

            report_file = ckpoint + ".test_report"
            # if evaluation is not already done, run evaluation 
            if not os.path.isfile(report_file):
                run(shlex.split(eval_cmd))
                logger.info(f"Submitted: Evaluation results not found. {eval_exec_msg}")
                n_submitted_jobs += 1
            else:
                # if evaluation was done but model seems faulty (checked with JGA <0.1), overwrite evaluation 
                with open(report_file, "r") as f:
                    report = json.load(f)

                if report["report"]["joint goal acc"] < 0.1:
                    n_submitted_jobs += 1
                    run(shlex.split(eval_cmd))
                    logger.info(f"Submitted: Evaluation results were found but previous results seem faulty: (JGA = {jga}) {eval_exec_msg}")
                else: 
                    logger.info(f"Skipped: Evaluation results found and previous results seem okay (JGA={jga}).") 

        print("\n"*5)
        logger.info("Submitting CheckDST evaluations...")
        for aug in ["NED", "SDI", "PI"]:
            script_fn = os.path.join(PARLAI_DIR, "bash_scripts/evaluate_on_checkdst.sh")
            inv_cmd = f"sbatch {script_fn} -m {ckpoint} -i {aug} -f {is_fewshot}"

            report_file = ckpoint + f".{aug}_report_fs_{is_fewshot}.json"
            eval_exec_msg = f"Running evaluation with `{ckpoint}` on {aug} fewshot={is_fewshot}"
            # if evaluation is not already done, run evaluation 
            if not os.path.isfile(report_file):
                run(shlex.split(inv_cmd))
                n_submitted_jobs += 1
                logger.info(f"Submitted: Evaluation results not found. {eval_exec_msg}")
            else:
                # if evaluation was done but model seems faulty (checked with JGA <0.1), overwrite evaluation 
                with open(report_file, "r") as f:
                    report = json.load(f)
                
                jga = report["report"]["jga_original"]
                if jga < 0.1:
                    n_submitted_jobs += 1
                    run(shlex.split(inv_cmd))
                    logger.info(f"Submitted: Evaluation results were found, but previous results seem faulty (JGA = {jga}) {eval_exec_msg}")
                else: 
                    logger.info(f"Skipped: Evaluation results found and previous results seem okay (JGA={jga}).")

    logger.info(f"Number of jobs submitted: {n_submitted_jobs}")


def main():
    # pass main directory via command line 
    main_dir = sys.argv[1]  
    # or hardcode main directory path 
    custom_dir = ""

    if custom_dir: 
        subdirs = glob.glob(os.path.join(custom_dir, "*"))
    else:
        subdirs = glob.glob(os.path.join(main_dir, "*"))

    for sd in sorted(subdirs):
        logger.info(f"Submitting jobs for {sd}")
        eval_checkpoints_in_dir(sd)

if __name__ == "__main__":
    main()
