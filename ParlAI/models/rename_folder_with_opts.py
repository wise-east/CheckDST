# Use model.opt to rename file directories that makes
# organziation easier with show_test_results.py

from pathlib import Path
import sys
import json
import subprocess
import shlex

path = sys.argv[1]

subdirs = Path(path).glob("*")

for subdir in subdirs:

    # clean up if there is no model.test file for some reason
    test_path = subdir / "model.test"
    if not test_path.is_file():
        subprocess.run(shlex.split(f"rm -rf {subdir}"))
        continue

    opt_path = subdir / "model.opt"

    opts = json.load(opt_path.open("r"))

    lr = opts['learningrate']
    batch_size = opts['batchsize']
    update_freq = opts['update_freq']
    fewshot = opts['few_shot']
    useprompts = opts['use_prompts']
    seed = opts['rand_seed']

    newpath = f"fs_{fewshot}"
    newpath += f"_prompts_{useprompts}"
    newpath += f"_lr{lr:.0e}"
    newpath += f"_bs{batch_size}"
    newpath += f"_uf{update_freq}"
    newpath += f"_sd{seed}"

    # print(str(subdir))
    date_start_idx = str(subdir).rfind("202")
    # import pdb; pdb.set_trace()
    date_end_idx = str(subdir)[date_start_idx:].rfind(":") + 2
    date = str(subdir)[date_start_idx : date_start_idx + date_end_idx + 1]
    # print(date)
    if date.strip():
        newpath += f"_{date}"

    print(subdir.parent)
    fullpath = subdir.parent / newpath
    print(fullpath)
    subdir.rename(fullpath)
