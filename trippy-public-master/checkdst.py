import glob
import os 
import sys
from get_no_hallucination_frequency import NAMED_ENTITY_SLOTS_FOR_TRIPPY

from get_no_hallucination_frequency import get_no_hall_freqs
from get_coref_jga import get_coref_jga
from get_conditional_jga import get_conditional_jgas, load_pred_as_df

from pathlib import Path
import json 
import pandas as pd 
from metric_bert_dst import load_dataset_config, tokenize, check_slot_inform
import numpy as np 
from tqdm import tqdm 
from loguru import logger
import math 
import re 
from collections import defaultdict
import seaborn as sns 


# TARGET_METRIC = [
#     "test_jga", 
#     # "valid_jga", 
#     "coref_jga", 
#     "NEI cJGA", 
#     # "NEI JGA", 
#     "SD cJGA", 
#     # "SD JGA",
#     "TP cJGA",
#     # "TP JGA"
# ]

TARGET_METRIC= [
    "NoHF Orig",
    "TP cJGA",
    "NoHF Swap",
    "SD cJGA",
    "test_jga",
    "coref_jga",
    "NEI cJGA",
]


class CheckDST_Trippy: 

    def __init__(self, dir_name): 

        self.dir_name = dir_name
        # load and save path 
        self.save_path = os.path.join(self.dir_name, "checkdst_results.json")

        self.dataset_config = "dataset_config/multiwoz21.json"
        self.class_types, self.slots, self.label_maps = load_dataset_config(self.dataset_config)
        self.check_dst_results = defaultdict(dict) 
        self.invariance_types = ["NEI", "TP", "SD"] 
        self.orig_jgas = defaultdict(dict)
        self.named_entity_slots = NAMED_ENTITY_SLOTS_FOR_TRIPPY

        # needed for any conversions 
        label_maps_tmp = {}
        for v in self.label_maps:
            label_maps_tmp[tokenize(v)] = [tokenize(nv) for nv in self.label_maps[v]]
        self.label_maps = label_maps_tmp

        # load all pred
        self.get_all_pred_files()

    def load_tp_skip_ids(self, tp_skip_fn="data/laug_dst/TP/no_change_ids.txt"): 
        """
        load dialog ids that had no change between original and paraphrased version
        Use these dialog ids to exclude them in calculating conditional jgas
        """
        with open(tp_skip_fn, "r") as f: 
            self.tp_skip_ids = [t.lower().replace(".json", "") for t in f.read().splitlines()]
        
    def load_need_coref_dial_ids(self, multiwoz23_fp="data/MULTIWOZ2.3/data.json"): 
        """Get conversations that require coreference 
        """

        with open(multiwoz23_fp, "r") as f: 
            multiwoz23 = json.load(f)

        self.need_corefs =[] 
        for dial_id, dial in tqdm(multiwoz23.items()):

            need_coref = False
            for turn_num in range(math.ceil(len(dial["log"]) / 2)):
                # # # turn number
                turn = {"turn_num": turn_num, "dial_id": dial_id}
                # any turn that comes after requiring coreference resolution will also need coref resolution
                need_coref = "coreference" in dial['log'][turn_num * 2] or need_coref
                turn['need_coref'] = need_coref

                if need_coref: 
                    self.need_corefs.append(f"{dial_id.lower()}-{turn_num}".replace(".json", ""))

    def get_all_pred_files(self, file_pattern="pred_res.*.csv"): 
        """
        CSV files are already processed JSON files. Using the processed files will make this script run much faster. 
        """

        all_fns = sorted(glob.glob(os.path.join(self.dir_name, file_pattern)))

        self.aug_test_fns = {} 
        for inv in self.invariance_types: 
            self.aug_test_fns[inv] = [fn for fn in all_fns if inv in fn and "test" in fn]

        self.orig_test_fns = [fn for fn in all_fns if all([inv not in fn for inv in self.invariance_types]) and "test" in fn] 
        self.orig_dev_fns = [fn for fn in all_fns if all([inv not in fn for inv in self.invariance_types]) and "dev" in fn] 

    def load_pred_file_as_df(self, fn): 
        if ".csv" in fn: 
            df = pd.read_csv(fn)
            return df 

        elif ".json" in fn: 
            with open(fn, "r") as f: 
                data = json.load(f)
            
            df = self.format_data_as_df(data)
            return df 
        else: 
            logger.error("Found a file that is neither a csv or a json file. Not sure how to process this file")

    # modified from metric_bert_dst.py
    def get_slot_vals(self, slot, pred, joint_pd_slot, swap_with_map=True): 
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
        if turn_pd_class == self.class_types.index('none'):
            pass
        elif turn_pd_class == self.class_types.index('dontcare'):
            joint_pd_slot = 'dontcare'
        elif turn_pd_class == self.class_types.index('copy_value'):
            joint_pd_slot = pd_slot
        elif 'true' in self.class_types and turn_pd_class == self.class_types.index('true'):
            joint_pd_slot = 'true'
        elif 'false' in self.class_types and turn_pd_class == self.class_types.index('false'):
            joint_pd_slot = 'false'
        elif 'refer' in self.class_types and turn_pd_class == self.class_types.index('refer'):
            if pd_slot[0:3] == "§§ ":
                if pd_slot[3:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[3:], self.label_maps, swap_with_map)
            elif pd_slot[0:2] == "§§":
                if pd_slot[2:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:], self.label_maps, swap_with_map)
            elif pd_slot != 'none':
                joint_pd_slot = pd_slot
        elif 'inform' in self.class_types and turn_pd_class == self.class_types.index('inform'):
            if pd_slot[0:3] == "§§ ":
                if pd_slot[3:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[3:], self.label_maps, swap_with_map)
            elif pd_slot[0:2] == "§§":
                if pd_slot[2:] != 'none':
                    joint_pd_slot = check_slot_inform(joint_gt_slot, pd_slot[2:], self.label_maps, swap_with_map)
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
        if joint_gt_slot != 'none' and joint_gt_slot != 'dontcare' and joint_gt_slot != 'true' and joint_gt_slot != 'false' and joint_gt_slot in self.label_maps:
            for variant in self.label_maps[joint_gt_slot]:
                if variant == joint_pd_slot and swap_with_map:
                    joint_pd_slot = joint_gt_slot 

        return joint_gt_slot, joint_pd_slot 

    def format_data_as_df(self, data): 
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

            for slot in self.slots: 
                joint_pd_slot = pred_dst.get(slot, None)

                joint_gt_slot, joint_pd_slot = self.get_slot_vals(slot, pred, joint_pd_slot)
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

    def get_dial_id_to_jga_from_pred_file(self, fn):
        df = self.load_pred_file_as_df(fn) 
        dialid2jga = {}
        correct_list =[] 
        for idx, row in df.iterrows(): 
            jga = int(all([row[f"{slot}_gold"] == row[f"{slot}_pred"] for slot in self.slots]))
            if jga == 1: 
                correct_list.append(row['id'])

            # if NEI is the augmentation, keep only those with named entity slots 
            if "NEI" in fn: 
                for k in row.keys(): 
                    if "gold" in k and k.split("_")[0] in NAMED_ENTITY_SLOTS_FOR_TRIPPY and row[k] != "none": 
                        dialid2jga[row["id"].replace(".json", "")] = jga 
                        break
            else: 
                dialid2jga[row["id"].replace(".json", "")] = jga 
            # dialid2jga[row["id"].replace(".json", "")] = jga 

        return dialid2jga 

    
    def load_conditional_jgas(self): 
        """ get and store cjgas for each checkpoint
        """
        self.load_tp_skip_ids()
        
        for inv in self.invariance_types: 
        # for inv in ["NEI"]: 
            
            logger.info(f"Loading jgas for invariance type: {inv}")
            # print(self.aug_test_fns[inv])
            for inv_fn in tqdm(self.aug_test_fns[inv]): 
                inv_jgas = self.get_dial_id_to_jga_from_pred_file(inv_fn)
                step_ct = self.get_step_count_from_fp(inv_fn)
                orig_jgas_copy = self.orig_jgas[step_ct].copy()

                is_fewshot = "fewshot_True" in inv_fn 
                if inv == "TP": 
                    inv_jgas = {k:v for k,v in inv_jgas.items() if k not in self.tp_skip_ids}
                    orig_jgas_copy = {k:v for k,v in orig_jgas_copy.items() if k not in self.tp_skip_ids}

                if inv == "NEI": 
                    # only compare those with named entity slots 
                    with open(f"/data/home/justincho/trippy-public-master/data/laug_dst/NEI/changed_dialids_test_{is_fewshot}.txt", "r") as f: 
                        swapped_dialogs = [s.replace("\n", "") for s in f.readlines()]
                    orig_jgas_copy = {k:v for k,v in orig_jgas_copy.items() if k in swapped_dialogs}
                    inv_jgas = {k:v for k,v in inv_jgas.items() if k in swapped_dialogs}
                    
                if inv_jgas.keys() != orig_jgas_copy.keys(): 
                    # continue
                    import pdb; pdb.set_trace()

                conditional_jgas = {}
                for dial_id, inv_jga in inv_jgas.items(): 
                    orig_jga = orig_jgas_copy[dial_id]
                    if inv_jga or orig_jga: 
                        conditional_jga = 1 if inv_jga and orig_jga else 0 
                        conditional_jgas[dial_id] = conditional_jga 
                self.check_dst_results[step_ct][f"{inv} cJGA"] = np.mean(list(conditional_jgas.values()))
                self.check_dst_results[step_ct][f"{inv} JGA"] = np.mean(list(inv_jgas.values()))
    
    def load_orig_jgas(self, split="test"): 
        logger.info(f"Loading original {split} set jgas...")
        fns = self.orig_test_fns if split=="test" else self.orig_dev_fns
        for fp in tqdm(fns): 
            dialid2jga = self.get_dial_id_to_jga_from_pred_file(fp)
            jga = np.mean(list(dialid2jga.values()))
            step_ct = self.get_step_count_from_fp(fp)
            self.check_dst_results[step_ct][f"{split}_jga"] = jga  
            if split == "test": 
                self.orig_jgas[step_ct] = dialid2jga 

    def get_step_count_from_fp(self, fp:str): 
        """
        """
        step_ct = re.sub("[^0-9]", "", Path(fp).name)
        if step_ct:
            return step_ct 
        else:
            return "final"

    def load_coref_jgas(self): 
        """load the coref jga value from each prediction file 
        """
        logger.info("Loading original test set coref jgas...")
        for step_ct, dialid2jga in tqdm(self.orig_jgas.items()): 
            coref_jgas =[jga for dialid, jga in dialid2jga.items() if dialid in self.need_corefs]
            self.check_dst_results[step_ct]["coref_jga"] = np.mean(coref_jgas)

    def get_check_dst_result_df(self): 
        """
        Return a dataframe of checkdst results 
        """

        return pd.DataFrame(self.check_dst_results)

    def load_no_hall_freqs(self): 
        """
        optional: will be 100% or very close to it. 
        """

        raise NotImplementedError

    def load_all(self): 
        if os.path.isfile(self.save_path): 
            logger.info(f"Found previously loaded results. Loading it from {self.save_path}")
            self.load_results(self.save_path) 

        else: 
            logger.info("Didn't find previous loaded results. Loading from csv files")
            self.load_need_coref_dial_ids()
            self.load_orig_jgas(split="valid")
            self.load_orig_jgas()
            self.load_conditional_jgas() 
            self.load_coref_jgas()
            self.save_results() 

    def load_results(self, save_path=""): 
        """
        """
        
        if not save_path: 
            save_path = self.save_path 
        with open(self.save_path, "r") as f: 
            self.check_dst_results = json.load(f)

    def save_results(self, save_path=""): 
        """
        """
        if not save_path: 
            save_path = self.save_path 

        with open(self.save_path, "w") as f: 
            json.dump(self.check_dst_results, f, indent=4, sort_keys=True)


def get_check_dst_objs(main_dir):

    subdirs = glob.glob(os.path.join(main_dir, "*")) 
    dst_results ={}
    for sd in subdirs: 
        name = Path(sd).name
        if "multiwoz" not in name: 
            continue 
        print(name)
        checkdst = CheckDST_Trippy(sd)

        checkdst.load_all() 

        dst_results[name] = checkdst

    return dst_results 


def return_grouped_dfs(dst_results): 
    fewshot_dfs =[] 
    fullshot_dfs =[] 
    for fp, checkdst_res in dst_results.items(): 
        df = checkdst_res.get_check_dst_result_df()
        if "True" in fp: 
            fewshot_dfs.append(df)
            # print(df)
        else: 
            fullshot_dfs.append(df)
    return fullshot_dfs, fewshot_dfs


def format_dfs_for_plotting(dfs):
    grouped = pd.concat([d.T for d in dfs])
    # grouped["variable"] = grouped.index
    grouped = grouped.reset_index()
    grouped = grouped[grouped["index"]!="final"] # final is the same as the last checkpoint 
    grouped["index"] = grouped["index"].apply(lambda x: int(x)) # transform for corret ordering in plot

    grouped["NoHF Orig"] = [100]* len(grouped)
    grouped["NoHF Swap"] = [100]* len(grouped)

    for tm in TARGET_METRIC: 
        grouped[tm] = grouped[tm].apply(lambda x : x*100 if x <=1 else x)

    plot_metrics = [tm for tm in TARGET_METRIC if "valid" not in tm]
    molten_df = grouped.melt(["index"], plot_metrics)
    molten_df.head()

    return molten_df


def plot_cjga_trends(molten_df, no_band=True, title=""): 

    sns.set_theme()
    sns.set(font_scale = 1.5)
    sns.set_style("whitegrid")
    ci = None if no_band else "sd"

    molten_df["%"] = molten_df["value"].tolist()
    molten_df["steps"] = molten_df["index"].tolist()
    
    if "full" in title.lower(): 
        molten_df["epochs"] = molten_df["steps"].apply(lambda x: int(x/2366))
    if "few" in title.lower(): 
        molten_df["epochs"] = molten_df["steps"].apply(lambda x: int(x/277))

    molten_df["CheckDST"] = molten_df["variable"].tolist()

    if 'epochs' in molten_df.columns: 
        x_col = "epochs"
    else: 
        x_col = "steps"

    rel = sns.relplot(
        data=molten_df, kind="line",
        x=x_col, y="%",
        hue="CheckDST", 
        # style="CheckDST",
        estimator=np.median,
        aspect=1.2, 
        ci=ci,
        linewidth=7
    )

    rel._legend.remove()
    leg = rel._legend
    for line in leg.get_lines():
        line.set_linewidth(1.0)

    rel.fig.suptitle(title)

def get_paper_results(dfs, final=True): 
    df = pd.concat([d.T for d in dfs]).reset_index()


    df["NoHF Orig"] = [100]* len(df)
    df["NoHF Swap"] = [100]* len(df)

    for tm in TARGET_METRIC: 
        df[tm] = df[tm].apply(lambda x : x*100 if x<=1 else x)

    # agg_func = "mean"
    agg_func = "median"
    stat_ = df.groupby("index").agg([agg_func, "sem"])
    stat_ = stat_[stat_.index != "final"]
    stat_.index = pd.Series(stat_.index).apply(lambda x: int(x))
    stat_ = stat_.sort_values(by="index")

    if not final: 
        return stat_

    final = stat_[stat_.index == stat_["valid_jga"].idxmax()[agg_func]]

    return final 


if __name__ == "__main__": 

    use_new = True 
    # use_new = False

    if use_new: 
        # main_dir = sys.argv[1]
        # main_dir = "/data/home/justincho/trippy-public-master/results/of_interest/multiwoz23_lr1e-4_2022-01-05_09:41:24_fewshot_False_46"
        main_dir = "/data/home/justincho/trippy-public-master/results/arxiv_results/multiwoz23_lr1e-4_2021-12-05_07:59:37"
        main_dir = "/data/home/justincho/trippy-public-master/results/multiwoz23_lr1e-4_2022-01-08_07:38:54_fewshot_False_53"

        checkdst = CheckDST_Trippy(main_dir)
        checkdst.load_need_coref_dial_ids()
        checkdst.load_orig_jgas(split="valid")
        checkdst.load_orig_jgas()
        checkdst.load_conditional_jgas() 
        checkdst.load_coref_jgas()
        print(checkdst.get_check_dst_result_df())

    else:
        # main_dir = sys.argv[1]
        # main_dir = "/data/home/justincho/trippy-public-master/results/of_interest"
        main_dir = "/data/home/justincho/trippy-public-master/results/paper_results"

        dirs = glob.glob(os.path.join(f"{main_dir}", "*12-05*"))
        print(dirs)

        all_results = [] 
        for dir in dirs: 
            name = Path(dir).name
            print(name)
            results = [name]
            results += get_conditional_jgas(dir)
            results.append(get_no_hall_freqs(dir)[0]) # returns no_hall_freq & count 
            results.append(get_coref_jga(dir))

            results = [str(r) for r in results]
            print(results)
            with open(os.path.join(dir, "checkdst_results_new.csv"), "w") as f: 
                f.write(",".join(results))

    #     all_results.append(results)


    # with open(os.path.join(main_dir, "checkdst_all_results_new.csv"), "w") as f: 
    #     for r in all_results: 
#     for r in all_results: 
    #     for r in all_results: 
    #         f.write(",".join(r) + "\n")