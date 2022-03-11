# usage: python get_conditional_jga file1 file2 
import os 
import json 
import pandas as pd
from get_coref_jga import get_jgas_from_dataframe, format_data_as_df
from get_no_hallucination_frequency import NAMED_ENTITY_SLOTS_FOR_TRIPPY
import numpy as np 
import sys
from pathlib import Path

def load_pred_as_df(fn): 
    with open(fn, "r") as f: 
        data = json.load(f)
        
    df = format_data_as_df(data)
    return df 

def get_conditional_jgas(dir, fn="pred_res.test.final.json"): 

    orig_fn = os.path.join(dir, fn)
    # orig_df = pd.read_csv(orig_fn)
    orig_df = load_pred_as_df(orig_fn).sort_values(by="id")
    orig_df_jgas = get_jgas_from_dataframe(orig_df)
    print(f"orig JGA: {np.mean(orig_df_jgas):.4f}")

    tp_skip_fn = "data/laug_dst/TP/no_change_ids.txt"
    with open(tp_skip_fn, "r") as f: 
        tp_skips = [t.lower().replace(".json", "") for t in f.read().splitlines()]

    results =[np.mean(orig_df_jgas)] 

    # for inv in ["NEI"]: 
    # for inv in ["TP", "SD", "NEI"]: 
    for inv in ["NEI", "TP", "SD"]: 
        orig_copy = orig_df.copy() 

        inv_fn = str(Path(orig_fn).with_suffix("")) + f"{inv}.json"
        # inv_fn = os.path.join(dir, f"pred_res.test.final{inv}.json")

        if not os.path.isfile(inv_fn): 
            print(f"File not found for {inv}: {inv_fn}")
            results.append(-1)
            continue 

        # inv_df = pd.read_csv(inv_fn)
        inv_df = load_pred_as_df(inv_fn).sort_values(by="id")

        inv_df['id'] = inv_df['id'].apply(lambda x: x.replace(".json", ""))
        orig_copy['id'] = orig_copy['id'].apply(lambda x: x.replace(".json", ""))

        # make sure we skip samples that weren't actually paraphrased. 
        if inv == "TP": 
            print(len(inv_df))
            inv_df = inv_df.loc[~inv_df['id'].isin(tp_skips)]
            print(len(inv_df))
            orig_copy = orig_copy.loc[~orig_copy['id'].isin(tp_skips)]
            
        # # TODO: make sure we skip samples that didn't have named entities replaced
        # # TODO: make sure that named entities are actually replaced for the testfile 
        if inv == "NEI": 
            nei_keeps = [] 
            for idx, row in inv_df.iterrows(): 
                for k in row.keys(): 
                    if "gold" in k and k.split("_")[0] in NAMED_ENTITY_SLOTS_FOR_TRIPPY and row[k] != "none": 
                        # if idx < 100: 
                        #     print(row[k])
                        nei_keeps.append(row['id'])
            print(len(inv_df))
            inv_df = inv_df.loc[inv_df['id'].isin(nei_keeps)]
            print(len(inv_df))
            orig_copy = orig_copy.loc[orig_copy['id'].isin(nei_keeps)]

        assert inv_df['id'].tolist() == orig_copy['id'].tolist()

        inv_df_jgas = get_jgas_from_dataframe(inv_df)
        orig_copy_df_jgas = get_jgas_from_dataframe(orig_copy)

        assert len(inv_df_jgas) == len(orig_copy_df_jgas)

        conditional_jgas =[] 
        
        for idx in range(len(inv_df_jgas)): 
            inv_jga = inv_df_jgas[idx]
            orig_jga = orig_copy_df_jgas[idx]
            if inv_jga or orig_jga: 
                
                conditional_jga = 1 if inv_jga and orig_jga else 0 
                conditional_jgas.append(conditional_jga)

        # print(f"orig JGA: {np.mean(orig_df_jgas):.4f}")
        print(f"\n{inv} JGA: {np.mean(inv_df_jgas):.4f}")
        print(f"{inv} cJGA: {np.mean(conditional_jgas):.4f}")
        results.append(np.mean(inv_df_jgas))
        results.append(np.mean(conditional_jgas))

    return results


if __name__ == "__main__": 
    dir_ = sys.argv[1]
    get_conditional_jgas(dir_)