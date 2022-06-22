import glob
from pathlib import Path
import os
import shlex
from subprocess import run

dir_list = glob.glob("results/multiwoz*")

# get full paths
dir_list = [os.path.abspath(p) for p in dir_list]

for dir_ in dir_list:
    # for aug in ["SD", "TP", "NEI"]:
    for aug in ["None"]:
        run(
            shlex.split(
                f"sbatch /data/home/justincho/trippy-public-master/DO.example.laug {dir_} {aug}"
            )
        )
