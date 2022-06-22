# This is necessary to make sure that we make a fair comparison with the same kind of slot values seen during training and testing
import json
import os
from tqdm import tqdm
from loguru import logger



# for version in ["2.1", "2.2", "2.3"]: 
for version in ["2.1", "2.3"]: 
    for inv in  ["SD", "TP", "orig"]:
        keys = ["train", "valid", "test"]
        # keys = ["test"]
        for key in keys:
            fn = os.path.join("laug_dst", inv, f"data_reformat_{key}.json")
            with open(fn, "r") as f:
                laug_data = json.load(f)

            gold_path = os.path.join(
                f"multiwoz_dst/MULTIWOZ{version}", f"data_reformat_{key}.json"
            )
            with open(gold_path, "r") as f:
                gold_data = json.load(f)

            more_laug = len(set(laug_data.keys()) - set(gold_data.keys()))
            more_official = len(set(gold_data.keys()) - set(laug_data.keys()))

            logger.info(
                f"For split '{key}': # keys more in laug: {more_laug}, # keys more in official_v23: {more_official}"
            )

            new_data = {}
            fn = os.path.join(
                "laug_dst", inv, f"data_reformat_official_v{version}_slots_{key}.json"
            )
            for k, v in tqdm(laug_data.items()):
                if k not in gold_data:
                    continue
                new_data[k] = laug_data[k].copy()

                # replace dst predictions and orig_context with gold data's content
                new_data[k]["slots_inf"] = gold_data[k]["slots_inf"]
                new_data[k]["orig_context"] = gold_data[k]["context"]
                new_data[k]["need_coref"] = gold_data[k]["need_coref"]

                # replace the context for 'orig' from LAUG
                if "orig" in inv:
                    new_data[k]["context"] = gold_data[k]["context"]
                    new_data[k].pop("orig_context")

            with open(fn, "w") as f:
                json.dump(obj=new_data, fp=f, indent=2)
            logger.info(len(new_data))
