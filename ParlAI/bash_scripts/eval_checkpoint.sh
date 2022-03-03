#!/bin/bash
#SBATCH --job-name=eval_multiwoz2.3
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00 
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/eval_checkpoint-%j.log

source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/ParlAI
conda activate parlai_internal

Help()
{
   # Display Help
   echo "Run evaluation script for a parlai model checkpoint"
   echo
   echo "Syntax: scriptTemplate [-h|h|m|i]"
   echo "options:"
   echo "h     Print this Help."
   echo "c     checkpoint"
   echo "d     eval target (test or valid)"
   echo "t     test on small subset"
   echo
}
# Default values
DO_TEST=False
DATA_TYPE="test"

while getopts ":hc:d:t:f:" option; do
   case $option in
      h) # display Help
        Help
        exit;;
      c)
        CHECKPOINT=$OPTARG;;
      d)
        DATA_TYPE=$OPTARG;;
      t) 
        DO_TEST=$OPTARG;;
      f) 
        FEWSHOT=$OPTARG;;
     \?) # Invalid option
        echo "Error: Invalid option"
        exit;;
   esac
done

echo "Evaluating ${CHECKPOINT} on data type: ${DATA_TYPE}"

MODEL="bart"
ADDITIONAL="--init-fairseq-model None"
if [[ $CHECKPOINT =~ "gpt2" ]]; then
  MODEL="hugging_face/gpt2"
  ADDITIONAL="--add-special-tokens True"
fi 

if [[ $CHECKPOINT =~ "fewshot_True" ]]; then
  FEWSHOT=True 
fi
if [[ $CHECKPOINT =~ "fewshot_False" ]]; then 
  FEWSHOT=False 
fi

parlai eval_model \
    -dt ${DATA_TYPE} \
    -m $MODEL -t multiwoz_dst -bs 1 \
    -mf $CHECKPOINT \
    --report-filename "${CHECKPOINT}.${DATA_TYPE}_report" \
    --world-logs "${CHECKPOINT}.${DATA_TYPE}_world_logs.jsonl" \
    --just_test $DO_TEST \
    --skip-generation False \
    --few_shot $FEWSHOT \
    $ADDITIONAL \
    # --val_reduced_size -1 \
