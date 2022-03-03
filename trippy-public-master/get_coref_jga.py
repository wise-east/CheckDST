# usage python get_coref_jga.py <pred_res.test.X.json> 

import json

from pandas.core.frame import DataFrame
from metric_bert_dst import load_dataset_config, tokenize, check_slot_inform
import sys 
from tqdm import tqdm 
import pandas as pd
import numpy as np
import math
import os


# prediction_fn= "/data/home/justincho/trippy-public-master/results/multiwoz23_lr1e-4_2021-12-05_07:59:48/pred_res.test.11830.json"
# prediction_fn= "/data/home/justincho/dialoglue/trippy/results/multiwoz_trippy_2021-12-08_01:45:08/pred_res.test.final.json"

dataset_config = "dataset_config/multiwoz21.json"
class_types, slots, label_maps = load_dataset_config(dataset_config)

label_maps_tmp = {}
for v in label_maps:
    label_maps_tmp[tokenize(v)] = [tokenize(nv) for nv in label_maps[v]]
label_maps = label_maps_tmp

# modified from metric_bert_dst.py
def get_slot_vals(slot, pred, joint_pd_slot, swap_with_map=True): 
    guid = pred['guid']  # List: set_type, dialogue_idx, turn_idx

    key_class_label_id = 'class_label_id_%s'%slot
    key_class_prediction = 'class_prediction_%s'%slot
    key_start_pos = 'start_pos_%s'%slot
    key_start_prediction = 'start_prediction_%s'%slot
    key_end_pos = 'end_pos_%s'%slot
    key_end_prediction = 'end_prediction_%s'%slot
    key_refer_id = 'refer_id_%s'%slot
    key_refer_prediction = 'refer_prediction_%s'%slot
    key_slot_groundtruth = 'slot_groundtruth_%s'%slot
    key_slot_prediction = 'slot_prediction_%s'%slot

    turn_gt_class = pred[key_class_label_id]
    turn_pd_class = pred[key_class_prediction]
    gt_start_pos = pred[key_start_pos]
    pd_start_pos = pred[key_start_prediction]
    gt_end_pos = pred[key_end_pos]
    pd_end_pos = pred[key_end_prediction]
    gt_refer = pred[key_refer_id]
    pd_refer = pred[key_refer_prediction]
    gt_slot = pred[key_slot_groundtruth]
    pd_slot = pred[key_slot_prediction]

    if swap_with_map: 
        gt_slot = tokenize(gt_slot)
        pd_slot = tokenize(pd_slot)

    # Make sure the true turn labels are contained in the prediction json file!
    joint_gt_slot = gt_slot

    if guid[-1] == '0': # First turn, reset the slots
        joint_pd_slot = 'none'

    # If turn_pd_class or a value to be copied is "none", do not update the dialog state.
    if turn_pd_class == class_types.index('none'):
        pass
    elif turn_pd_class == class_types.index('dontcare'):
        joint_pd_slot = 'dontcare'
    elif turn_pd_class == class_types.index('copy_value'):
        joint_pd_slot = pd_slot
    elif 'true' in class_types and turn_pd_class == class_types.index('true'):
        joint_pd_slot = 'true'
    elif 'false' in class_types and turn_pd_class == class_types.index('false'):
        joint_pd_slot = 'false'
    elif 'refer' in class_types and turn_pd_class == class_types.index('refer'):
        if pd_slot[0:3] == "§§ ":
            if pd_slot[3:] != 'none':
                joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[3:], label_maps, swap_with_map)
        elif pd_slot[0:2] == "§§":
            if pd_slot[2:] != 'none':
                joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:], label_maps, swap_with_map)
        elif pd_slot != 'none':
            joint_pd_slot = pd_slot
    elif 'inform' in class_types and turn_pd_class == class_types.index('inform'):
        if pd_slot[0:3] == "§§ ":
            if pd_slot[3:] != 'none':
                joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[3:], label_maps, swap_with_map)
        elif pd_slot[0:2] == "§§":
            if pd_slot[2:] != 'none':
                joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:], label_maps, swap_with_map)
        else:
            print("ERROR: Unexpected slot value format. Aborting.")
            exit()
    else:
        print("ERROR: Unexpected class_type. Aborting.")
        exit()


    # Check the joint slot correctness.
    # If the value label is not none, then we need to have a value prediction.
    # Even if the class_type is 'none', there can still be a value label,
    # it might just not be pointable in the current turn. It might however
    # be referrable and thus predicted correctly.
    if joint_gt_slot != 'none' and joint_gt_slot != 'dontcare' and joint_gt_slot != 'true' and joint_gt_slot != 'false' and joint_gt_slot in label_maps:
        for variant in label_maps[joint_gt_slot]:
            if variant == joint_pd_slot and swap_with_map:
                joint_pd_slot = joint_gt_slot 

    return joint_gt_slot, joint_pd_slot 

def format_data_as_df(data): 
    correct_ct = 0 
    pred_dst = {}
    gt_dst = {} 
    dict_results = [] 
    for pred in data: 
        guid = pred['guid']
        # reset dsts for new conversation 
        if guid[-1] == '0': 
            pred_dst = {} 
            gt_dst ={} 

        # guid: ['test/valid/train', 'dialog id', turn_idx]
        dialog_id, turn_idx = guid[1], guid[2]
        dict_result = {
            "id": f"{dialog_id.lower()}-{turn_idx}"
        } 

        for slot in slots: 
            joint_pd_slot = pred_dst.get(slot, None)

            joint_gt_slot, joint_pd_slot = get_slot_vals(slot, pred, joint_pd_slot)
            pred_dst[slot] = joint_pd_slot 
            gt_dst[slot] = joint_gt_slot  

            dict_result[f"{slot}_gold"]= joint_gt_slot
            dict_result[f"{slot}_pred"] = joint_pd_slot 

        dict_results.append(dict_result)

        # just need to compare these dictionaries, no need for fancy string formation 
        # if pred_dst == gt_dst: 
        #     correct_ct += 1 

    # print(f"Full JGA: {correct_ct / len(data):.4f}")

    df = pd.DataFrame(dict_results)
    return df 


def get_jgas_from_dataframe(df): 
    # calculate jga on csv loaded from metric_bert_dst.py 
    jgas =[] 
    for idx, row in df.iterrows(): 
        jga = 1 
        for slot in slots: 
            if row[f"{slot}_gold"] != row[f"{slot}_pred"]: 
                jga = 0 
        jgas.append(jga)
    return jgas


def get_coref_jga(dir): 

    with open(os.path.join(dir, "pred_res.test.final.json"), "r") as f: 
        data = json.load(f)

    # Get conversations that require coreference 
    with open("data/MULTIWOZ2.3/data.json", "r") as f: 
        multiwoz23 = json.load(f)

    need_corefs =[] 
    for dial_id, dial in tqdm(multiwoz23.items()):

        need_coref = False
        for turn_num in range(math.ceil(len(dial["log"]) / 2)):
            # # # turn number
            turn = {"turn_num": turn_num, "dial_id": dial_id}
            # any turn that comes after requiring coreference resolution will also need coref resolution
            need_coref = "coreference" in dial['log'][turn_num * 2] or need_coref
            turn['need_coref'] = need_coref

            if need_coref: 
                need_corefs.append(f"{dial_id.lower()}-{turn_num}")

    # Get JGA line by line 


    df = format_data_as_df(data)
    full_jga = np.mean(get_jgas_from_dataframe(df))

    print(f"Full JGA: {full_jga:.4f}")

    # calculate coreference jga on csv loaded from metric_bert_dst.py 
    coref_jgas =[] 
    correct_list =[] 
    for idx, row in df.iterrows(): 
        if row['id'] not in need_corefs: 
            continue 
        jga = 1 
        for slot in slots: 
            if row[f"{slot}_gold"] != row[f"{slot}_pred"]: 
                jga = 0 
        if jga == 1: 
            correct_list.append(row['id'])
        coref_jgas.append(jga)

    print(f"Coref JGA: {np.mean(coref_jgas):.4f}, number of samples: {len(coref_jgas)}")
    return np.mean(coref_jgas)

if __name__ == "__main__": 
    dir_ = sys.argv[1]
    get_coref_jga(dir_)