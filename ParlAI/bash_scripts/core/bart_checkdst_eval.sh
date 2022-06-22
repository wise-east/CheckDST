#!/bin/bash
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id


source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/CheckDST
source /data/home/justincho/CheckDST/set_envs_mine.sh
conda activate parlai_checkdst

# CMD=$1
# echo $CMD 
# eval $CMD