from argparse import ArgumentParser
import shlex
from subprocess import run
from pathlib import Path
import os
from loguru import logger
from glob import glob
import time
import json

parser = ArgumentParser()

# parser.add_argument("-mf", "--model_file", type=str, help="path to model checkpoint")
parser.add_argument(
    "-fs", "--fewshot", type=bool, default=False, help="Use few shot data"
)
parser.add_argument(
    "-f",
    "--force",
    type=bool,
    default=False,
    help="Overwrite previous invariance report results if report is already there",
)
parser.add_argument(
    "--no_execute", action="store_true", help="Whether to not execute commands"
)


args = parser.parse_args()

invariances = ["TP", "SD", "NEI"]
# invariances = ["NEI"]
# invariances = ["TP", "SD"]

fps = []
fps += glob(
    "/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1*"
)
# fps += glob("/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/fs_True_prompts_True_lr5e-05_bs4_uf1*")
# # fps += glob("/data/home/justincho/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1*")
# fps += glob("/data/home/justincho/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.3/fs_True_prompts_True_lr1e-05_bs4_uf2*")
# fps += glob("/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.3/fs_True_prompts_True_lr5e-05_bs4_uf1*")
# # fps += glob("/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.3/fs_False_prompts_True_lr1e-04_bs4_uf1*")
# # fps += glob("/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.1/fs_False_prompts_True_lr5e-05_bs8_uf1*")
# fps += glob("/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.1/fs_True_prompts_True_lr5e-05_bs4_uf1*")
# # fps += glob("/data/home/justincho/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.1/fs_False_prompts_True_lr1e-05_bs8_uf2*")
# fps += glob("/data/home/justincho/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.1/fs_True_prompts_True_lr5e-05_bs4_uf2*")
# # fps += glob("/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.1/fs_False_prompts_True_lr1e-04_bs8_uf1*")
# fps += glob("/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.1/fs_True_prompts_True_lr1e-04_bs4_uf1*")


problematic_dirs = []
run_count = 0
for fp in fps:
    filename = Path(fp).name
    model_opt_fn = os.path.join(fp, "model.opt")
    if not os.path.isfile(model_opt_fn):
        logger.error(
            f"No model.opt file found in {fp}. Skipping evaluation for this directory"
        )
        problematic_dirs.append(fp)
        continue

    # if there is no test results, there was a problem during training or the training is incomplete. skip
    if not os.path.isfile(os.path.join(fp, "model.test")):
        logger.error(
            f"No model.test file found in {fp}. Skipping evaluation for this directory"
        )
        problematic_dirs.append(fp)
        continue

    with open(model_opt_fn, "r") as f:
        opt = json.load(f)

    if "model" not in filename and os.path.isfile(os.path.join(fp, "model")):
        logger.warning(
            "Make sure to add the full path to the model file, not the path. Automatically adding 'model' to the model path..."
        )
        fp = os.path.join(fp, "model")

    for inv in invariances:
        fewshot = opt.get('few_shot', False)
        report_fn = fp + f".{inv}_report_fs_{fewshot}.json"

        # detect any faulty reports that were created by mistake
        faulty_report_fn = fp + f".{inv}_report_fs_{not fewshot}.json"
        faulty_world_logs_fn = fp + f".{inv}_world_logs_fs_{not fewshot}.jsonl"
        faulty_metadata_fn = fp + f".{inv}_world_logs_fs_{not fewshot}.metadata"
        # print(faulty_report_fn)
        if (
            os.path.isfile(faulty_report_fn)
            or os.path.isfile(faulty_world_logs_fn)
            or os.path.isfile(faulty_metadata_fn)
        ):
            logger.error(
                f"Found faulty report/worldlogs/metadata filename in {fp}. Deleting:"
            )
            if os.path.isfile(faulty_report_fn):
                logger.info(faulty_report_fn)
                run(shlex.split(f"rm -rf {faulty_report_fn}"))
            if os.path.isfile(faulty_metadata_fn):
                logger.info(faulty_metadata_fn)
                run(shlex.split(f"rm -rf {faulty_metadata_fn}"))
            if os.path.isfile(faulty_world_logs_fn):
                logger.info(faulty_world_logs_fn)
                run(shlex.split(f"rm -rf {faulty_world_logs_fn}"))

        # skip if we already have the invariance results
        if not args.force and os.path.isfile(report_fn):
            logger.info(
                f"Invariance report already found. skipping for {fp} for invariance {inv}"
            )
            continue

        logger.info(f"Saving report to {report_fn}...")

        # cmd = f"sbatch --output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/eval_laug_inv_{inv}-%j.log --job-name={inv}_inv_eval "
        cmd = f"sbatch "

        if "bart" in fp:
            model_type = "bart"
        else:
            model_type = "hugging_face/gpt2"

        cmd += (
            f"evaluate_laug_invariance.sh -m {fp} -i {inv} -f {fewshot} -d {model_type}"
        )

        if not args.no_execute:
            print(f"Executing command: \n\t{cmd}")
            run(shlex.split(cmd))
            run_count += 1

            if "muppet" in cmd:
                time.sleep(20)

    # break

logger.info(f"Total number of jobs submitted: {run_count}")
prob_dirs_str = '\n'.join(problematic_dirs)
print(f"problematic dirs to delete:\n{prob_dirs_str}")
for dir in problematic_dirs:
    run(shlex.split(f"rm -rf {dir}"))
