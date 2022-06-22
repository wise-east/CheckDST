import json
from pathlib import Path
from pprint import pprint
import re
import shlex
import subprocess

# PATH="/data/home/justincho/CheckDST/ParlAI/models/bart_scratch_multiwoz2.3"
PATH = "/data/home/justincho/CheckDST/ParlAI/models/bart_pft_multiwoz2.3"

runs = Path(PATH).glob("*sd[0-9]")
# pprint(sorted(list(runs)))
def find_step(fn):
    step = re.sub("[^0-9]", "", fn.name)
    if step:
        step = int(step)
    else:
        step = float('inf')
    return step


def sort_by_steps(fns):
    return sorted(fns, key=lambda x: find_step(x))


def find_seed(fn):
    # import pdb; pdb.set_trace()
    seed = re.search("sd([0-9])", str(fn))
    if seed:
        return seed[1]
    else:
        return None


MODEL = "bart"
ADDITIONAL = "--init-fairseq-model None"

BASE_SCRIPT_NAME = "bart_checkdst_eval.sh"
ALL_TASK = "multiwoz_checkdst:augmentation_method=PI,multiwoz_checkdst:augmentation_method=SDI,multiwoz_checkdst:augmentation_method=NED,multiwoz_checkdst:augmentation_method=orig"
ORIG_TASK = "multiwoz_checkdst:augmentation_method=orig"
DO_TEST = False
FEWSHOT = False
USEPROMPT = False
BATCH_SIZE = 128
EPOCHS = [0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]


with open(BASE_SCRIPT_NAME, "r") as f:
    lines = f.readlines()

for run in runs:

    CMD = ""
    seed = find_seed(run)
    print(run, seed)
    checkpoints = [
        fn
        for fn in run.glob("*")
        if re.match(".*checkpoint_step[0-9]*$", str(fn))
        or re.match(".*checkpoint$", str(fn))
        or re.match(".*model$", str(fn))
    ]
    checkpoints = sort_by_steps(checkpoints)
    # pprint(list(checkpoints))
    for ckpt in checkpoints:

        trainstat_fn = ckpt.parent / (ckpt.name + '.trainstats')
        # print(trainstat_fn)

        with trainstat_fn.open("r") as f:
            trainstat = json.load(f)
            epoch = round(trainstat["total_epochs"], 2)

        if epoch not in EPOCHS:
            continue

        print(epoch)
        CHECKPOINT = str(ckpt)
        model_config = Path(PATH).name
        # custom arguments needed for GPT-2 vs BART
        if "gpt2" in CHECKPOINT:
            MODEL = "hugging_face/gpt2"
            ADDITIONAL = "--add-special-tokens True"

        # evaluate for all checkdst augmentations and test set
        DATA_TYPE = "test"
        REPORT_FN = f"{CHECKPOINT}.{DATA_TYPE}_report"
        WORLDLOGS_FN = f"{CHECKPOINT}.{DATA_TYPE}_world_logs.jsonl"
        if not Path(WORLDLOGS_FN).is_file():
            CMD += f"""
            
parlai eval_model \
    -m {MODEL} \
    -mf {CHECKPOINT} \
    -t {ALL_TASK} -bs {BATCH_SIZE} \
    -dt {DATA_TYPE} \
    --just_test {DO_TEST} \
    --report-filename {REPORT_FN} \
    --world-logs {WORLDLOGS_FN} \
    --skip-generation False \
    --few_shot {FEWSHOT} \
    --use_prompts {USEPROMPT} \
    {ADDITIONAL} |& tee {CHECKPOINT}_{DATA_TYPE}.log.txt

"""

        # evaluate for validation loss
        DATA_TYPE = "valid"
        REPORT_FN = f"{CHECKPOINT}.{DATA_TYPE}_report"
        WORLDLOGS_FN = f"{CHECKPOINT}.{DATA_TYPE}_world_logs.jsonl"

        if not Path(WORLDLOGS_FN).is_file():
            CMD += f"""
            
parlai eval_model \
    -m {MODEL} \
    -mf {CHECKPOINT} \
    -t {ORIG_TASK} -bs {BATCH_SIZE} \
    -dt {DATA_TYPE} \
    --just_test {DO_TEST} \
    --report-filename {REPORT_FN} \
    --world-logs {WORLDLOGS_FN} \
    --skip-generation False \
    --few_shot {FEWSHOT} \
    --use_prompts {USEPROMPT} \
    {ADDITIONAL} |& tee {CHECKPOINT}_{DATA_TYPE}.log.txt 

"""

    JOBNAME = f"checkdst_{model_config}_sd{seed}"
    print(JOBNAME)
    print(lines)
    print(CMD)

    new_script_fn = Path(BASE_SCRIPT_NAME).with_suffix(f".{JOBNAME}.temp")
    with new_script_fn.open("w") as f:
        f.writelines(lines)
        f.write(CMD)

    # run the command
    full_cmd = f'sbatch -J {JOBNAME} {str(new_script_fn)}'
    subprocess.run(shlex.split(full_cmd))

    # delete file
    # new_script_fn.unlink()
