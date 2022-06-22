# calculate checkdst metrics with single file

import json 
from pathlib import Path 
# load gold reference 
# load aug prediction file 
# load orig prediction file 
from utils import load_jsonl, CheckDST


gold_data_fn = "/data/home/justincho/CheckDST/data/checkdst/NED/data_reformat_official_v2.3_slots_test.json"
    
orig_pred_fn = "/data/home/justincho/CheckDST/ParlAI/models/pre_emnlp/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1_sd0/model.checkpoint_step3839.test_world_logs.checkdst_prediction.jsonl"
aug_pred_fn = "/data/home/justincho/CheckDST/ParlAI/models/pre_emnlp/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1_sd0/model.checkpoint_step3839.NEI_world_logs_fs_False.checkdst_prediction.jsonl"

checkdst = CheckDST(gold_data_fn = gold_data_fn)
checkdst.add_preds(orig_pred_fn, aug_type="orig")  
checkdst.add_preds(aug_pred_fn, aug_type="NED")  

# results = checkdst.compute_metrics(orig_col="orig", aug_col="NED")
# print(results) # results should contain 

# process 
# store orig predictions & metrics
# store aug predictions & metrics
# calculate cjga 