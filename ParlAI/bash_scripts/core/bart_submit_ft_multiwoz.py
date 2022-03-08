import subprocess
import shlex
from loguru import logger
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
import time
import os

parser = ArgumentParser()

parser.add_argument(
    "-np",
    "--no_prompt",
    action='store_true',
    help="whether to not use prompts for inputs",
)
parser.add_argument(
    "-f",
    "--fewshot",
    action='store_true',
    help="whether to use fewshot setting for finetuning",
)
parser.add_argument(
    "-v",
    "--version",
    type=str,
    default="2.3",
    help="version of multiwoz to use for training",
)
parser.add_argument(
    "-i", "--init", type=str, default="pft", help="choices: [scratch, pft, muppet]"
)
parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
parser.add_argument("-u", "--update_freq", type=int, default=1, help="update frequency")
parser.add_argument(
    "-lr", "--learning_rate", type=float, default=-1, help="learning rate"
)

parser.add_argument(
    "-t", "--test", type=bool, default=False, help="test with small subset"
)

parser.add_argument(
    "-m", "--model", type=str, default="bart", help="type of model to use [bart, gpt2]"
)

parser.add_argument("-g", "--ngpu", type=int, default=1, help="number of gpus to use")

args = parser.parse_args()
print(args)

init_model_list = [
    # "/data/home/justincho/ParlAI/models/bart_all_pft/lr5e-6_eps10_ngpu8_bs8_2021-11-11_22:26:26",  # all (PrefineDST)
    "/data/home/justincho/ParlAI/models/bart_scratch_sgd_ft/lr5e-6_bs8_uf1_npu4_sd0_2021-11-19_07:00:21"  # for SOLOIST-BART
    # "/data/home/justincho/bart_-paraphrase_pft/lr1e-6_eps10_ngpu8_bs8_2021-11-11_10:26:37",
    # "/data/home/justincho/ParlAI/models/bart_-qa_pft/lr1e-6_eps10_ngpu8_bs8_2021-11-11_10:26:20",
    # "/data/home/justincho/ParlAI/models/bart_+wikisql_pft/lr5e-6_eps10_ngpu8_bs8_2021-11-11_18:28:31",
    # "/data/home/justincho/ParlAI/models/bart_-google_sgd_pft/lr1e-6_eps10_ngpu8_bs8_2021-11-11_22:26:31",
    # "/data/home/justincho/ParlAI/models/bart_-wikisql_pft/lr5e-6_eps10_ngpu8_bs8_2021-11-11_10:26:30",
    # "/data/home/justincho/ParlAI/models/bart_all_pft/lr5e-6_eps10_ngpu8_bs8_2021-11-11_22:26:26",
    # "/data/home/justincho/ParlAI/models/bart_+sgd_pft/lr5e-6_eps10_ngpu8_bs8_2021-11-11_22:27:15",
    # "/data/home/justincho/ParlAI/models/bart_+qa_pft/lr1e-6_eps10_ngpu8_bs8_2021-11-11_22:27:03",
    # "/data/home/justincho/ParlAI/models/bart_-copy_pft/lr1e-6_eps10_ngpu8_bs8_2021-11-11_22:26:54",
    # "/data/home/justincho/ParlAI/models/bart_-exp_coref_pft/lr5e-6_eps10_ngpu8_bs8_2021-11-11_22:26:39",
    # "/data/home/justincho/ParlAI/models/bart_-all_coref_pft/lr1e-5_eps10_ngpu8_bs8_2021-11-11_22:26:56",
    # "/data/home/justincho/ParlAI/models/bart_-paraphrase_pft/lr1e-6_eps10_ngpu8_bs8_2021-11-11_10:26:37",
    # "/data/home/justincho/ParlAI/models/bart_+all_coref_pft/lr1e-5_eps10_ngpu8_bs8_2021-11-11_18:28:25",
    # "/data/home/justincho/ParlAI/models/bart_+paraphrase_pft/lr1e-5_eps10_ngpu8_bs8_2021-11-11_22:27:05",
    # "/data/home/justincho/ParlAI/models/bart_+copy_pft/lr1e-5_eps10_ngpu8_bs8_2021-11-11_22:27:10",
    # "/data/home/justincho/ParlAI/models/bart_+exp_coref_pft/lr1e-6_eps10_ngpu8_bs8_2021-11-11_18:28:23"
]

init_model_list = [os.path.join(md, "model") for md in init_model_list]

init_config = args.init
if init_config != "pft":
    init_model_list = ["/data/home/justincho/ParlAI/data/models/bart/bart_large/model"]
if args.model == "gpt2":
    assert init_config in ["simpletod", "soloist"]
    if init_config == "simpletod":
        init_model_list = [""]
    elif init_config == "soloist":
        init_model_list = [
            "/data/home/justincho/ParlAI/models/gpt2_+sgd_pft/lr5e-06_eps10_ngpu8_bs8_2022-01-13_19:01:54"
        ]
    else:
        raise NotImplementedError
        # init_model_list = []


use_prompts = not args.no_prompt
version = args.version
batch_size = args.batch_size
update_freq = args.update_freq

if args.learning_rate == -1:
    lrs = ["1e-4", "5e-5", "1e-5"]
else:
    lrs = [args.learning_rate]

seeds = [
    "0",
    "1",
    "2",
    "3",
    "4",
    # "5",
    # "6",
]
fewshot_configs = [
    # False,
    # True,
    args.fewshot
]

count = 0
for sd in seeds:
    for lr in lrs:
        for fewshot in fewshot_configs:

            if fewshot:
                batch_size = 4
                update_freq = 1
            else:
                batch_size = args.batch_size
                update_freq = args.update_freq

            for init_model in init_model_list:
                # force more updates before each evaluation for fewshot setting
                time.sleep(1.1)
                now = datetime.now()
                now = datetime.strftime(now, "%Y-%m-%d_%T")

                # directory to save models to
                mf_dir = f"models/{args.model}_{init_config}_multiwoz{version}/"

                # modify directory when using prefinetuned model weights
                if init_config == "pft":
                    idx = init_model.index("lr")
                    init_model_lr = init_model[idx + 2 : init_model.rfind("_")]
                    # directory should contain the alias used for prefinetuning
                    mf_dir = (
                        str(Path(init_model).parent.parent)
                        + f"_lr{init_model_lr}_multiwoz{version}"
                    )

                mf_dir = os.path.join(
                    mf_dir,
                    f"lr{lr}_ngpu1_bs{batch_size}_uf{update_freq}_fewshot_{fewshot}_prompts_{use_prompts}_sd{sd}_{now}/",
                )

                init_command = f"--init-model {init_model}"
                # change init model command for muppet (it was trained with fairseq)
                if init_config == "muppet":
                    init_command = "--init-fairseq-model /data/home/justincho/ParlAI/data/models/bart_muppett/model.pt"

                # parameters for sbatch for organization purposes
                job_name = f"{init_config}_ft_{lr}"
                log_folder = f"/data/home/justincho/ParlAI/bash_scripts/slurm_logs/{init_config}_ft_{version}_lr{lr}_bs_{batch_size}_uf_{update_freq}_fs_{fewshot}/"

                subprocess.run(shlex.split(f"mkdir -p {log_folder}"))
                log_fn = os.path.join(log_folder, f"{now}.log")

                # form the command
                base_cmd = f"sbatch --job-name {job_name} --output={log_fn} --gres=gpu:{args.ngpu} "
                cmd = (
                    base_cmd
                    + f"bart_finetune_multiwoz.sh -m {mf_dir} -l {lr} -s {sd} -f {fewshot} -v {version} -p {use_prompts} -b {batch_size} -i \'{init_command}\' -g {log_fn} -u {args.update_freq} -t {args.test}"
                )

                if args.model == "gpt2":
                    if init_config == "simpletod":
                        init_command = ""
                    cmd = (
                        base_cmd
                        + f"gpt2_finetune_multiwoz.sh -m {mf_dir} -l {lr} -s {sd} -f {fewshot} -v {version} -p {use_prompts} -b {batch_size}  -g {log_fn} -u {args.update_freq} -t {args.test}"
                    )

                # Execute the command
                logger.info(f"Executing command: {cmd}")
                subprocess.run(shlex.split(cmd))
                if "muppet" in cmd:
                    time.sleep(20)
                count += 1


logger.info(f"Total number of jobs submitted: {count}")
