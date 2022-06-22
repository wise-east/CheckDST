import json 
import os 

DATAPATH = os.environ.get("DATAPATH", "")

for key in ["train", "valid", "test"]: 
    with open(os.path.join(DATAPATH, f"multiwoz_dst/MULTIWOZ2.3/data_reformat_{key}.json"), "r") as f: 
        v23 = json.load(f)

    v22_path= os.path.join(DATAPATH, f"multiwoz_dst/MULTIWOZ2.2/data_reformat_{key}.json")
    v21_path= os.path.join(DATAPATH, f"multiwoz_dst/MULTIWOZ2.1/data_reformat_{key}.json")

    def add_coref_labels(target_file): 
        with open(target_file, "r") as f: 
            target_data = json.load(f)
        
        for k in target_data.keys(): 
            if k not in v23: 
                target_data[k]['need_coref'] = False 
            else: 
                target_data[k]['need_coref'] = v23[k]['need_coref']

        with open(target_file, "w") as f: 
            json.dump(target_data, f, indent=2, sort_keys=True)

    # add_coref_labels(v22_path)
    add_coref_labels(v21_path)