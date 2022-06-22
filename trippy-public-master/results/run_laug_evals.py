from subprocess import run
import glob
import os
import shlex

main_dir = "of_interest/*"
main_dir = "/data/home/justincho/dialoglue/trippy/results/multiwoz*False*"

dirs = glob.glob(main_dir)
# aug = "NEI"
aug = "None"
ct = 0
for sd in dirs:
    sd = os.path.abspath(sd)
    print(sd)
    command = (
        f"sbatch /data/home/justincho/trippy-public-master/DO.example.laug {sd} {aug}"
    )
    print(command)
    ct += 1
    run(shlex.split(command))
    # break

print(ct)
