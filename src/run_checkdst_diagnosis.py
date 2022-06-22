# calculate checkdst metrics with single file

import json 
from pathlib import Path 
from utils import load_jsonl, CheckDST
from pprint import pprint 





gold_data_fn = "/data/home/justincho/CheckDST/data/checkdst/NED/data_reformat_official_v2.3_slots_test.json"
    
pred_fn = "/data/home/justincho/CheckDST/ParlAI/models/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-5_bs4_uf1_sd0/model.checkpoint_step928.test_world_logs_multiwoz_checkdst:augmentation_method={}.checkdst_prediction.jsonl"

checkdst = CheckDST(gold_data_fn = gold_data_fn)
for aug_type in ["orig", "NED", "SDI", "PI"]: 
    checkdst.add_preds(pred_fn.format(aug_type), aug_type=aug_type)  

# calculate cjga 
for aug_type in ["NED", "SDI", "PI"]: 
    checkdst.compute_cjga(orig="orig", aug=aug_type)
    
    
print(checkdst)


