# Format a world logs file to use for calculating CheckDST metrics

# output should simply be a list of dictionaries with keys: "context", "dial_id", "aug_type", "gold", "pred", "requires_coref"

import json
from pathlib import Path
import pandas as pd
import sys

sample_fn = Path(sys.argv[1])
# sample_fn = Path("/data/home/justincho/CheckDST/ParlAI/models/pre_emnlp/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1_sd0/model.checkpoint_step3839.NEI_world_logs_fs_False.jsonl")


def load_jsonl(fn):
    with open(fn, "r") as f:
        data = [json.loads(l) for l in f.readlines()]
    return data


def extract_target_components_from_worldlogs_line(line_item):

    # import pdb; pdb.set_trace()
    turn_input = line_item['dialog'][0][0]
    turn_output = line_item['dialog'][0][1]
    extracted = {
        "context": turn_input['text'],
        "dial_id": f"{turn_input['dial_id'].replace('.json', '')}-{turn_input['turn_num']}",
        "aug_type": turn_input.get("aug_type", "Not found"),
        "gold": turn_input['eval_labels'][0],
        "pred": turn_output['text'],
        "requires_coref": "coref_jga" in turn_output['metrics'],
    }
    # context may not be necessary as it can be mapped back to the original data file

    return extracted


data = load_jsonl(sample_fn)

checkdst_formatted = [extract_target_components_from_worldlogs_line(d) for d in data]
df = pd.DataFrame(checkdst_formatted)

train_stats_fn = sample_fn.with_suffix("").with_suffix(".trainstats")
# print(train_stats_fn)
target_fn = sample_fn.with_suffix(".checkdst_prediction.jsonl")
print(target_fn)
df.to_json(target_fn, orient="records", lines=True)
