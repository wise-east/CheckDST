# Getting started (CheckDST)

1. Create a folder for data inside `ParlAI`: i.e. `mkdir ParlAI/data` 
1. Place the data from the [data section](#data) into `ParlAI/data` 
1. Create an environment: `conda create -n parlai python=3.8` 
1. Follow steps in [ParlAI](ParlAI/README.md) to install `ParlAI` locally 
    ```
    cd ParlAI 
    python setup.py develop 
    ```
    1. (optional) Before installing `ParlAI`, it may be necessary to replace torch version in `requirements.txt` with one with CUDA support that is compatible with available GPUs, e.g. for a100 gpus: `torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html`  
    1. If this results in dependency errors when running parlai commands, run `python setup.py develop` again. 
1. (optional) Train a ParlAI model: e.g. `parlai train_model --task multiwoz_dst` 
1. Evaluate on all CheckDST: `parlai eval_model --task multiwoz_checkdst:aug=orig,multiwoz_checkdst:aug=SD,multiwoz_checkdst:aug=TP,multiwoz_checkdst:aug=NED --model-file $MODELDIR -dt test` 
    1. Evaluate on individiual augmentation: `parlai eval_model --task multiwoz_checkdst --model-file $MODELDIR --aug $AUGTYPE -dt test` where `$AUGTYPE` is one of `[SD, TP, NEI]`
    1. Evaluate on original validation set: `parlai eval_model --task multiwoz_checkdst --model-file $MODELDIR -dt valid`
1. CheckDST results are available as `model.train_report_$AUGTYPE.json` files in the model directory. 
1. Metrics can be computed with a separate script that is universal for both TripPy and ParlAI models. (TBD)


Refer to the ParlAI [docs](https://www.parl.ai/docs/) for additional customization. 


### Overview 

- Scripts are in `bash_scripts/`. 
    - logs from slurm jobs are in `bash_scripts/slurm_logs`
- Trained models are in `models`
- Data is in `data` 

### Finetuning on MultiWOZ

- `bart_multiwoz_finetune.sh` has the template for submitting a slurm job for finetuning a BART model on MultiWOZ. 
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

