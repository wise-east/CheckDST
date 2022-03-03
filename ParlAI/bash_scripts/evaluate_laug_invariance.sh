#!/bin/bash
#SBATCH --job-name=eval_laug_inv
#SBATCH --partition=a100 
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00 
#SBATCH --cpus-per-task=10
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/justincho/ParlAI/bash_scripts/slurm_logs/eval_laug_inv-%j.log


source /data/home/justincho/miniconda/etc/profile.d/conda.sh
cd /data/home/justincho/ParlAI
conda activate parlai_internal

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Run evaluation script for a parlai model checkpoint"
   echo
   echo "Syntax: scriptTemplate [-h|h|m|i]"
   echo "options:"
   echo "f     Use few shot test set"
   echo "h     Print this Help."
   echo "m     pass model path"
   echo "i     Type of invariance to use. Should be one of SD, TP, NEI"
   echo
}

############################################################
# Main Program                                             #
############################################################

MODEL=bart

while getopts ":hm:i:f:d:" option; do
   case $option in
      f)
        FEWSHOT=$OPTARG;;
      h) # display Help
        Help
        exit;;
      i)
        INVARIANCE=$OPTARG;;
      d)
        MODEL=$OPTARG;;
      m)
        CHECKPOINT=$OPTARG;;
     \?) # Invalid option
        echo "Error: Invalid option"
        exit;;
   esac
done

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


echo $CHECKPOINT
echo $FEWSHOT
echo $INVARIANCE
# parlai multiprocessing_eval \
# if [[ ! -e "${CHECKPOINT}.${INVARIANCE}_report_fs_${FEWSHOT}.json" ]]; then
parlai eval_model \
    -m $MODEL \
    --model-file $CHECKPOINT \
    --datatype test \
    --batchsize 1 \
    --task multiwoz_dst_laug \
    --skip-generation False \
    --report-filename "${CHECKPOINT}.${INVARIANCE}_report_fs_${FEWSHOT}.json" \
    --world-logs "${CHECKPOINT}.${INVARIANCE}_world_logs_fs_${FEWSHOT}.jsonl" \
    -aug $INVARIANCE \
    -fs $FEWSHOT \
    $ADDITIONAL
      
# fi
# looking at the data 

# parlai dd --task multiwoz_dst_laug -dt test -aug TP
# if [[ ! -e "${MF}_report_fs_${FEWSHOT}.json" ]]; then 
#   parlai eval_model \
#       -m $MODEL \
#       --model-file $MF \
#       --datatype test \
#       --batchsize 1 \
#       --task multiwoz_dst \
#       --report-filename "${MF}.report_fs_${FEWSHOT}.json" \
#       --world-logs "${MF}_world_logs_fs_${FEWSHOT}.jsonl" \
#       -fs $FEWSHOT \

# fi