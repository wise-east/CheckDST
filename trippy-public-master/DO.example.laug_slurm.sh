#!/bin/bash

#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00 # run for one day
#SBATCH --cpus-per-task=10
#SBATCH --job-name=trippy_laug


source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/trippy-public-master
conda activate trippy

# Parameters ------------------------------------------------------

#TASK="sim-m"
#DATA_DIR="data/simulated-dialogue/sim-M"
#TASK="sim-r"
#DATA_DIR="data/simulated-dialogue/sim-R"
#TASK="woz2"
#DATA_DIR="data/woz2"
TASK="multiwoz21"
# DATA_DIR="data/MULTIWOZ2.1"
# TASK="multiwoz21"
# INV=SD
INV=$2
# INV=orig
DATA_DIR="data/laug_dst/${INV}"
if [ "$INV" = "None" ]; then
	DATA_DIR="data/MULTIWOZ2.3"
fi 
NOW=$(date +"%Y-%m-%d_%T")
LR="1e-4"

# Project paths etc. ----------------------------------------------

# OUT_DIR="results/multiwoz21_lr${LR}_${NOW}"
# OUT_DIR="results/multiwoz23_lr${LR}_${NOW}"
# OUT_DIR="results/convbert_multiwoz23_lr${LR}_${NOW}"
# OUT_DIR="results/convbert_multiwoz21_lr${LR}_${NOW}"
# mkdir -p ${OUT_DIR}
# OUT_DIR=$2
# OUT_DIR="results/multiwoz23_lr1e-4_gpu2/"



# dialoglue path
# OUT_DIR="/data/home/justincho/dialoglue/trippy/results/multiwoz_trippy_2021-12-08_01:45:08"
OUT_DIR=$1
# OUT_DIR="/data/home/justincho/trippy-public-master/results/multiwoz23_lr1e-4_2021-12-05_07:59:48"

# Main ------------------------------------------------------------

# MODEL_PATH="bert-base-uncased" 
# MODEL_PATH="convbert-dg"
MODEL_PATH=$OUT_DIR 
# MODEL_PATH="/data/home/justincho/trippy-public-master/results/multiwoz23_lr1e-4_gpu2/checkpoint-5920"

# for step in dev; do
# for step in train dev test; do
for step in test; do
    args_add=""
    if [ "$step" = "train" ]; then
	args_add="--do_train --predict_type=dummy"
    elif [ "$step" = "dev" ] || [ "$step" = "test" ]; then
	args_add="--do_eval --predict_type=${step}"
    fi

    python3 run_dst.py \
	    --task_name=${TASK} \
	    --data_dir=${DATA_DIR} \
	    --dataset_config=dataset_config/${TASK}.json \
	    --model_type="bert" \
	    --model_name_or_path=${MODEL_PATH} \
	    --do_lower_case \
	    --learning_rate=$LR \
	    --num_train_epochs=10 \
	    --max_seq_length=512 \
	    --per_gpu_train_batch_size=48 \
	    --per_gpu_eval_batch_size=1 \
	    --output_dir=${OUT_DIR} \
	    --save_epochs=2 \
	    --logging_steps=10 \
	    --warmup_proportion=0.1 \
	    --eval_all_checkpoints \
	    --adam_epsilon=1e-6 \
	    --label_value_repetitions \
        --swap_utterances \
	    --append_history \
	    --use_history_labels \
	    --delexicalize_sys_utts \
	    --class_aux_feats_inform \
	    --class_aux_feats_ds \
		--laug_inv ${INV} \
	    ${args_add} \
	    2>&1 | tee ${OUT_DIR}/${step}_${INV}.log
    
    if [ "$step" = "dev" ] || [ "$step" = "test" ]; then
    	python3 metric_bert_dst.py \
    		${TASK} \
		dataset_config/${TASK}.json \
    		"${OUT_DIR}/pred_res.${step}*${INV}.json" \
    		2>&1 | tee ${OUT_DIR}/eval_pred_${step}_${INV}.log
    fi
done
