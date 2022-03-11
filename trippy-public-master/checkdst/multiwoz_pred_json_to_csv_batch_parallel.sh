#!/bin/bash

#SBATCH --time=0:30:00 # run for one day
#SBATCH --cpus-per-task=10
#SBATCH --job-name=cpu_json2csv

source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/trippy-public-master
conda activate trippy

TASK=$1 
FP=$2 

python3 metric_bert_dst.py $TASK "dataset_config/${TASK}.json" $FP
