# get 50 single domain conversations from each of the attraction, train, restaurant, hotel domain for training
# another 50 for validation
# and 200 for testing
import json
import os
from argparse import ArgumentParser
from utils import seq2dict
from collections import defaultdict
import random

LAUG_FN_PREFIX = "data_reformat_official_v2.3_slots_"
OFF_FN_PREFIX = "data_reformat_"


def get_single_domain_conversations(path):

    dial_by_domain = defaultdict(dict)
    keys = ["test", "train", "valid"]

    # domains to do few shot with
    target_domains = ["attraction", "hotel", "restaurant", "train", "taxi"]

    dialid2turns = defaultdict(dict)
    # load all data
    for key in keys:
        # load laug data
        if "laug" in path:
            with open(os.path.join(path, f"{LAUG_FN_PREFIX}{key}.json"), "r") as f:
                data = json.load(f)

        # load official data
        else:
            with open(os.path.join(path, f"{OFF_FN_PREFIX}{key}.json"), "r") as f:
                data = json.load(f)

        # approach: group by dialids, check last turns dst length, keep only those with single target domain
        # only add conversations that have a single domain throughout the entire conversation

        # group by dial id
        for id, turn_item in data.items():
            dial_id = id.split(".json")[0]
            dialid2turns[dial_id][id] = turn_item

    for dialid, turns in dialid2turns.items():
        max_idx = -1
        for id, turn in turns.items():
            max_idx = max(turn["turn_num"], max_idx)

        last_dst = seq2dict(turns[f"{dialid}.json-{max_idx}"]["slots_inf"])
        # format of last_dst: {domain: {slot key: slot value}}

        # check if last dst only contains one domain.
        num_domains = len(last_dst)
        if num_domains != 1:
            continue

        else:
            domain = list(last_dst.keys())[0]
            # we only care about conversations in the target domains
            if domain not in target_domains:
                continue
            dial_by_domain[domain][dialid] = turns

    print(f"Single domain counts:")
    for k, v in dial_by_domain.items():
        print(f"\t{k}: {len(v)}")
        # for id, turns in v.items():
        #     print(turns)
        #     break

    return dial_by_domain


parser = ArgumentParser()

parser.add_argument(
    "-p",
    "--path",
    required=True,
    help="directory of data that has reformatted multiwoz data (should have data_reformat_X.json name)",
)

args = parser.parse_args()

# example paths
# path = "/data/home/justincho/project/ParlAI/data/multiwoz_dst/MULTIWOZ2.3"
# path = "/data/home/justincho/project/ParlAI/data/laug_dst/TP"

path = args.path
dial_by_domain = get_single_domain_conversations(path)

# goal: {domain: {dial_id: {dial_turn_id: dial_turn_item}}}


few_shot_sizes = {"train": 50, "valid": 50, "test": 200}
few_shot_train = 50
few_shot_valid = 50

few_shot_data = defaultdict(dict)

random.seed(0)
for domain, convs in dial_by_domain.items():

    # import pdb; pdb.set_trace()
    if domain == "attraction":
        few_shot_test = 78
    else:
        few_shot_test = 200

    conv_list = random.sample(
        list(convs.keys()), k=few_shot_train + few_shot_valid + few_shot_test
    )

    for conv_id in conv_list[:few_shot_train]:
        few_shot_data['train'].update(convs[conv_id])

    for conv_id in conv_list[few_shot_train : few_shot_train + few_shot_valid]:
        few_shot_data['valid'].update(convs[conv_id])

    for conv_id in conv_list[
        few_shot_train
        + few_shot_valid : few_shot_train
        + few_shot_valid
        + few_shot_test
    ]:
        few_shot_data['test'].update(convs[conv_id])


for k, v in few_shot_data.items():
    if "laug" in path:
        save_path = os.path.join(path, f"{LAUG_FN_PREFIX}{k}_fewshot.json")
    else:
        save_path = os.path.join(path, f"{OFF_FN_PREFIX}{k}_fewshot.json")
    with open(save_path, "w") as f:
        json.dump(obj=v, fp=f, indent=2, sort_keys=True)
