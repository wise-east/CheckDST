### Overview 

- Scripts are in `bash_scripts/`. 
    - logs from slurm jobs are in `bash_scripts/slurm_logs`
- Trained models are in `models`
- Data is in `data` 

### Finetuning on MultiWOZ

- `bart_finetune_multiwoz.sh` has the template for submitting a slurm job for finetuning a BART model on MultiWOZ. 
- `bart_submit_ft_multiwoz.py` contains the python script for submitting multiple jobs with various configurations. 


What's missing: 
- Automatically keeping track of all training runs and their time stamps
    - ideally automatically be able to group/filter training runs based on configurations to easily organize results. (take advantage of .opt files?)

### Other script descriptions

- `cancel_jobs.sh`
- `eval_all_checkpoints.py` 
- `eval_checkpoint.sh`
- `evaluate_all_invariance_metrics.py` 
    - template script for submitting slurm job for running laug invariance type 
    - runs all augmentations: TP, SD, NEI. 
    - usage: `python evaluate_all_invariance_metrics.py -fs <few shot: true/false> -f <force: true/false> --no_execute <whether to actually submit job: true/false>`
- `evaluate_laug_invariance.sh`

- `examine_incorrect_predictions.py` 
- `exp_commands.sh` 
- `gpt2_finetune_multiwoz.sh` 
- `muppet_finetune_multiwoz.sh` 
- `organize_results.py` 
    - 
- `remote_jupyter.sh` 
- `test_commands.sh` 
- `visualize_results.ipynb` 

