# laug evaluations for dialoglue models 

submit_laug_evals () {
    out_dirs=("$@")
    for out_dir in ${out_dirs[@]}; do
        for aug in NEI TP SD ; do
            base="${out_dir##*/}"
            cmd="sbatch -J checkdst_$base /data/home/justincho/CheckDST/trippy-public-master/DO.example.laug_slurm.sh ${out_dir} ${aug}"
            echo $cmd 
            eval $cmd
        done 
    done 
}

submit_evals () {
    out_dirs=("$@")
    for out_dir in ${out_dirs[@]}; do
        base="${out_dir##*/}"
        # if only formatting files
        cmd="sbatch -J test_$base /data/home/justincho/CheckDST/trippy-public-master/DO.example.advanced_slurm.sh -d ${out_dir} -m true "
        # if inference is also needed
        # cmd="sbatch -J test_$base /data/home/justincho/CheckDST/trippy-public-master/DO.example.advanced_slurm.sh -d ${out_dir} "
        echo $cmd 
        eval $cmd
    done 
}


RESULT_DIR=/data/home/justincho/CheckDST/dialoglue/trippy/results/emnlp/
OUT_DIRS=(
    ${RESULT_DIR}multiwoz_trippy_convbert_sd42_fewshot_False_2022-06-24_07:30:52
    ${RESULT_DIR}multiwoz_trippy_convbert_sd43_fewshot_False_2022-06-24_07:30:52
    ${RESULT_DIR}multiwoz_trippy_convbert_sd44_fewshot_False_2022-06-24_07:30:52
    ${RESULT_DIR}multiwoz_trippy_convbert_sd45_fewshot_False_2022-06-24_07:30:53
    ${RESULT_DIR}multiwoz_trippy_convbert_sd46_fewshot_False_2022-06-24_07:30:53
)

echo ${OUT_DIRS[@]}
# submit_laug_evals ${OUT_DIRS[@]}
# submit_evals ${OUT_DIRS[@]}


RESULT_DIR=/data/home/justincho/CheckDST/trippy-public-master/results/emnlp/
OUT_DIRS=(
    # ${RESULT_DIR}multiwoz23_lr1e-4_2022-06-24_07:06:50_fewshot_false_42
    ${RESULT_DIR}multiwoz23_lr1e-4_2022-06-24_17:54:23_fewshot_false_43
    ${RESULT_DIR}multiwoz23_lr1e-4_2022-06-24_17:54:40_fewshot_false_44
    ${RESULT_DIR}multiwoz23_lr1e-4_2022-06-24_17:57:48_fewshot_false_45
    ${RESULT_DIR}multiwoz23_lr1e-4_2022-06-24_17:57:50_fewshot_false_46
)

echo ${OUT_DIRS[@]}
# submit_laug_evals ${OUT_DIRS[@]}
# submit_evals ${OUT_DIRS[@]}


