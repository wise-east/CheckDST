import json 
from collections import defaultdict
import sys 
from pathlib import Path 

data_path = sys.argv[1]
data_path = Path(data_path).absolute()
cur_dir = data_path.parent
# data_path="data.json"

with data_path.open("r") as f: 
    data = json.load(f)

split_keys = {} 
for key in ["val", "test"]: 
    with open(f"{key}ListFile.json", "r") as f: 
        split_keys[key] = f.read().splitlines()

data_splits = {
    "test": {},
    "val": {},
    "train": {} 
}

for k, dial in data.items(): 
    if k in split_keys["test"]: 
        data_splits["test"][k] = dial
    elif k in split_keys["val"]: 
        data_splits["val"][k] = dial 
    else: 
        data_splits["train"][k] = dial 

assert len(data) == len(data_splits["test"]) + len(data_splits["val"]) + len(data_splits["train"])

for key in data_splits.keys(): 
    save_path = cur_dir / f"{key}_dials.json"
    with save_path.open("w") as f: 
        json.dump(obj=data_splits[key], fp=f, indent=4, sort_keys=True)
 
