#!/bin/bash

#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00 # run for one day
#SBATCH --cpus-per-task=10
#SBATCH --job-name=trippy

source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/CheckDST/trippy-public-master
conda activate trippy

# Parameters ------------------------------------------------------

#TASK="sim-m"
#DATA_DIR="data/simulated-dialogue/sim-M"
#TASK="sim-r"
#DATA_DIR="data/simulated-dialogue/sim-R"
#TASK="woz2"
#DATA_DIR="data/woz2"
# TASK="multiwoz21"
# DATA_DIR="data/MULTIWOZ2.1"

# Shared parameters 
TASK="multiwoz21"
# DATA_DIR="data/MULTIWOZ2.1"
DATA_DIR="data/MULTIWOZ2.3"
NOW=$(date +"%Y-%m-%d_%T")
LR="1e-4"

Help()
{
   # Display Help
   echo "Run training or evaluation for trippy-based models"
   echo
   echo "Syntax: scriptTemplate [-h|h|m|i]"
   echo "options:"
   echo "h     Print this Help."
   echo "d     directory to save for training or evaluate from"
   echo "f	   Use fewshot setting"
   echo "m     Whether to evaluate only (json to csv)"
   echo "s 	   Seed value"
   echo
}

SEED=""
FEWSHOT=false
N_EPOCHS=10
PER_GPU_TRAIN_BATCH_SIZE=24
SAVE_EPOCHS=0.25
METRIC_BERT_DST_ONLY=false # change to true if predictions are already generated and want to transform them from csv to json 

# OUT_DIR="results/multiwoz21_lr${LR}_${NOW}_${SEED}"
# OUT_DIR="results/convbert_multiwoz23_lr${LR}_${NOW}_${SEED}"
# OUT_DIR="results/convbert_multiwoz21_lr${LR}_${NOW}_${SEED}"

while getopts ":hd:s:m:f:" option; do
   case $option in
      h) # display Help
        Help
        exit;;
      d)
        OUT_DIR=$OPTARG;;
      s)
        SEED=$OPTARG;;
      m) 
        METRIC_BERT_DST_ONLY=$OPTARG;;
      f) 
        FEWSHOT=$OPTARG;;
     \?) # Invalid option
        echo "Error: Invalid option"
        exit;;
   esac
done

if [[ $SEED != "" ]] ; then 
	OUT_DIR="results/multiwoz23_lr${LR}_${NOW}_fewshot_${FEWSHOT}_${SEED}"
	splits=(train dev test)
else 
	echo "No seed is given. We assume that this is intentional and that only evaluation will be done for checkpoints at ${OUT_DIR}"
	splits=(test dev)
	# splits=(test)
fi 

echo $SEED
echo $OUT_DIR

# Main ------------------------------------------------------------

MODEL_PATH="bert-base-uncased" 
# MODEL_PATH="convbert-dg"

if [[ $METRIC_BERT_DST_ONLY == true ]] ; then 
	splits=(test)
fi

for step in ${splits[@]}; do
	echo $step 

    args_add=""
    if [[ "$step" = "train" ]] ; then
		args_add="--do_train --predict_type=dummy --seed ${SEED}"
	elif [[ "$step" = "dev" ]] || [[ "$step" = "test" ]] ; then
		args_add="--do_eval --predict_type=${step}"
    fi

	if [[ $METRIC_BERT_DST_ONLY != true ]] ; then 
		echo "Run training / evaluation inference" 

		mkdir -p ${OUT_DIR}

		python3 run_dst.py \
			--task_name=${TASK} \
			--data_dir=${DATA_DIR} \
			--dataset_config=dataset_config/${TASK}.json \
			--model_type="bert" \
			--model_name_or_path=${MODEL_PATH} \
			--do_lower_case \
			--learning_rate=$LR \
			--num_train_epochs=$N_EPOCHS \
			--max_seq_length=512 \
			--per_gpu_train_batch_size=$PER_GPU_TRAIN_BATCH_SIZE \
			--per_gpu_eval_batch_size=1 \
			--output_dir=${OUT_DIR} \
			--save_epochs=$SAVE_EPOCHS \
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
			${args_add} \
			2>&1 | tee ${OUT_DIR}/${step}.log
    fi 


    if [[ "$step" == "dev" ]] || [[ "$step" == "test" ]]; then

		echo "Run output reformatting"
    	python3 metric_bert_dst.py \
    		${TASK} \
		dataset_config/${TASK}.json \
    		"${OUT_DIR}/pred_res.${step}*json" \
    		2>&1 | tee ${OUT_DIR}/eval_pred_${step}.log
    fi

	# if [[ "$step" = "test" ]] && [[ $METRIC_BERT_DST_ONLY == false ]] ; then
	# 	echo "Submit LAUG evaluation jobs"
	# 	# evaluate on invariances 
	# 	sbatch DO.example.laug_slurm.sh ${OUT_DIR} NEI
	# 	sbatch DO.example.laug_slurm.sh ${OUT_DIR} TP
	# 	sbatch DO.example.laug_slurm.sh ${OUT_DIR} SD
	# fi 
done
