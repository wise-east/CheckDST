# usage python get_no_hallucination_frequency.py <pred_res.test.X.json> 


from get_coref_jga import format_data_as_df, get_slot_vals
import json

from pandas.core.frame import DataFrame
from metric_bert_dst import load_dataset_config, tokenize, check_slot_inform
import sys 
from tqdm import tqdm 
import pandas as pd
import numpy as np
import math
from transformers import (WEIGHTS_NAME, 
    BertConfig, 
    BertTokenizer,
    RobertaConfig, 
    RobertaTokenizer
)
import os

NAMED_ENTITY_SLOTS_FOR_TRIPPY ={
    "attraction-name",
    "restaurant-name",
    "hotel-name",
    "taxi-departure",
    "taxi-destination",
    "train-departure",
    "train-destination",
}


def preprocess(slot): 
    slot = slot.replace(" ' s", "'s")
    slot = slot.replace(" '", "'")
    slot = slot.replace("guest house", "guesthouse")
    slot = slot.replace(" ", "")
    slot = slot.replace("[SEP]", "")
    return slot 


def get_no_hall_freqs(dir): 


    prediction_fn = os.path.join(dir, "pred_res.test.final.json")

    # Get original context from multiwoz2.3 
    with open("data/MULTIWOZ2.3/data.json", "r") as f: 
        multiwoz23 = json.load(f)

    # get decoded conversational context 
    data_version = "2.3"
    dials_form = {} 
    for dial_id, dial in tqdm(multiwoz23.items()):
        context = []
        need_coref = False 
        for turn_num in range(math.ceil(len(dial["log"]) / 2)):
            # # # turn number
            turn = {"turn_num": turn_num, "dial_id": dial_id}

            # # # user utterance
            user_utt = dial["log"][turn_num * 2]["text"]
            sys_resp = dial["log"][turn_num * 2 + 1]["text"]
            # any turn that comes after requiring coreference resolution will also need coref resolution
            need_coref = "coreference" in dial['log'][turn_num * 2] or need_coref
            turn['need_coref'] = need_coref

            # # # dialog states, extracted based on "metadata", only in system side (turn_num * 2 + 1)
            slots_inf = []
            for domain, slot in dial["log"][turn_num * 2 + 1]["metadata"].items():
                for slot_type, slot_val in slot["book"].items():
                    if data_version == "2.3":
                        slot_val = [] if slot_val == "" else [slot_val]
                    if (
                        slot_val != []
                        and slot_type != "booked"
                        and slot_val[0] != "not mentioned"
                    ):
                        slots_inf += [domain, slot_type, slot_val[0] + ","]

                for slot_type, slot_val in slot["semi"].items():
                    # 2.3 doesn't have a list of possible values. just a single value. wrap as a list
                    if data_version == "2.3":
                        slot_val = [] if slot_val == "" else [slot_val]
                    if slot_val != [] and slot_val[0] != "not mentioned":
                        slots_inf += [domain, slot_type, slot_val[0] + ","]

            turn["slots_inf"] = " ".join(slots_inf)
            # turn["slots_err"] = self.create_err(slots_inf[:])
            # turn["slots_err"] = ""
            # # adding current turn to dialog history
            context.append("<user> " + user_utt)
            # # # dialog history
            turn["context"] = " ".join(context)
            # adding system response to next turn
            context.append("<system> " + sys_resp)
            dials_form[dial_id.lower() + "-" + str(turn_num)] = turn


    with open(prediction_fn, "r") as f: 
        data = json.load(f)

    dataset_config = "dataset_config/multiwoz21.json"
    class_types, slots, label_maps = load_dataset_config(dataset_config)

    label_maps_tmp = {}
    for v in label_maps:
        label_maps_tmp[tokenize(v)] = [tokenize(nv) for nv in label_maps[v]]
    label_maps = label_maps_tmp

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Get JGA line by line 
    correct_ct = 0 
    pred_dst = {}
    gt_dst = {} 

    dict_results = [] 
    no_hall_freqs = [] 
    count = 0 

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

        input_context = tokenizer.decode(pred['input_ids_restaurant-food'])

        no_hall_freq = 1
        context = dials_form[dict_result["id"]]['context'].lower().replace(" '", "'")
        for slot in NAMED_ENTITY_SLOTS_FOR_TRIPPY: 
            joint_pd_slot = pred_dst.get(slot, None)

            joint_gt_slot, joint_pd_slot = get_slot_vals(slot, pred, joint_pd_slot, swap_with_map=False)
            pred_dst[slot] = joint_pd_slot 
            gt_dst[slot] = joint_gt_slot  

            dict_result[f"{slot}_gold"]= joint_gt_slot
            dict_result[f"{slot}_pred"] = joint_pd_slot


            if joint_pd_slot != "none" \
                and joint_pd_slot not in context \
                and preprocess(joint_pd_slot) not in context \
                and preprocess(joint_pd_slot) not in preprocess(context) \
                and preprocess(joint_pd_slot) not in input_context \
                and preprocess(joint_pd_slot) not in preprocess(input_context): 
                no_hall_freq = 0 
                print(joint_pd_slot)
                print(preprocess(joint_pd_slot))
                print(context)
                print(preprocess(input_context))
                # print(context)

                count += 1 
            # elif joint_pd_slot != "none" and joint_pd_slot in context: 
            #    print(joint_pd_slot)
            #    print(context)
            #    no_hall_freq = 0 
            
            if joint_pd_slot in label_maps: 
                for alternative in label_maps[joint_pd_slot]: 
                    # print(alternative)
                    if alternative in context: 
                        no_hall_freq = 1 

            # if no_hall_freq == 0 : 
            #    print(joint_pd_slot)
            #    print(context)
        no_hall_freqs.append(no_hall_freq)    

    # if no_hall_freq == 0 : 
    #    break 

    # 100% is expected
    print(f"no hallucination frequency: {np.mean(no_hall_freqs):.4f} (expected 100%) \t hallucination count: {count}")

    return np.mean(no_hall_freqs), count 

    