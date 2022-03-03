#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --time=24:00:00 # run for one day
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/jupyter-%j.log

source /data/home/justincho/miniconda/etc/profile.d/conda.sh

conda activate parlai_internal
cat /etc/hosts
jupyter-lab --ip=0.0.0.0 --port=${1:-8888} # use your desired port 