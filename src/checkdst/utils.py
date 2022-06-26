import json
import re
from typing import List, Dict, Tuple, Set
from loguru import logger
from pathlib import Path, PosixPath
from collections import defaultdict

import os
import seaborn as sns
import pandas as pd
import numpy as np

DOMAINS = {
    "attraction",
    "hotel",
    "hospital",
    "restaurant",
    "police",
    "taxi",
    "train",
}

ALL_SLOTS = {
    "attraction--area",
    "attraction--name",
    "attraction--type",
    "hospital--department",
    "hotel--area",
    "hotel--day",
    "hotel--internet",
    "hotel--name",
    "hotel--parking",
    "hotel--people",
    "hotel--pricerange",
    "hotel--stars",
    "hotel--stay",
    "hotel--type",
    "restaurant--area",
    "restaurant--day",
    "restaurant--food",
    "restaurant--name",
    "restaurant--people",
    "restaurant--pricerange",
    "restaurant--time",
    "taxi--arriveby",
    "taxi--departure",
    "taxi--destination",
    "taxi--leaveat",
    "train--arriveby",
    "train--day",
    "train--departure",
    "train--destination",
    "train--leaveat",
    "train--people",
}


NAMED_ENTITY_SLOTS = {
    "attraction--name",
    "restaurant--name",
    "hotel--name",
    "bus--departure",
    "bus--destination",
    "taxi--departure",
    "taxi--destination",
    "train--departure",
    "train--destination",
}

SLOT_VAL_CONVERSION = {
    "centre": "center",
    "3-star": "3",
    "2-star": "2",
    "1-star": "1",
    "0-star": "0",
    "4-star": "4",
    "5-star": "5",
}

METRIC_MAPPING = {
    "jga_orig": "JGA",
    "coref_jga_orig": "CorefJGA",
    "factuality_orig": "F Orig",
    "factuality_NED": "F Swap",
    "jga_NED": "NED pJGA",
    "cjga_orig_NED": "NED cJGA",
    "conditional_jga_orig_NED": "NED old cJGA",
    "jga_PI": "PI pJGA",
    "cjga_orig_PI": "PI cJGA",
    "conditional_jga_orig_PI": "PI old cJGA",
    "jga_SDI": "SDI pJGA",
    "cjga_orig_SDI": "SDI cJGA",
    "conditional_jga_orig_SDI": "SDI old cJGA",
}


def find_seed(fn: PosixPath) -> int:
    """Find the seed value in the filename

    Args:
        fn (PosixPath): filename of interest

    Returns:
        int: found seed value. None if not found.
    """
    seed = re.search("sd([0-9]+)", str(fn))
    if seed:
        return seed[1]
    else:
        return None


def find_step(fn: PosixPath) -> int:
    """Find the training step value in the filename

    Args:
        fn (PosixPath): filename of interest

    Returns:
        int: found step value. 'inf' if not found.
    """
    if isinstance(fn, str):
        fn = Path(fn)
    step = re.sub("[^0-9]", "", fn.name)

    step = int(step) if step else float("inf")
    return step


def sort_by_steps(fns: List[PosixPath]) -> List[PosixPath]:
    """Return a list of filenames in ascending order of there steps

    Args:
        fns (List[PosixPath]): list of filenames

    Returns:
        List[PosixPath]: sorted list of filenames
    """
    return sorted(fns, key=lambda x: find_step(x))


def load_jsonl(fn: str) -> List[Dict]:
    """Load jsonl file"""
    with open(fn, "r") as f:
        data = [json.loads(l) for l in f.readlines()]
    return data


def normalize_dial_ids(dial_id: str) -> str:
    """Normalize dial id to take the form such as 'mul003-6' without any suffixes (.json), all lower case

    Args:
        dial_id (str): dial id

    Returns:
        str: normalized dial id
    """
    return dial_id.replace(".json", "").lower()


def extract_slot_from_string(slots_string: str) -> Tuple[List[str]]:
    """
    Either ground truth or generated result should be in the format:
    "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
    and this function would reformat the string into list:
    ["dom--slot_type--slot_val", ... ]
    """
    slots_list = []

    if slots_string is None:
        return [], [], [], []

    per_domain_slot_lists = {}
    named_entity_slot_lists = []

    # # # remove start and ending token if any
    try:
        str_split = slots_string.strip().split()
    except Exception as e:
        logger.error(str(e))
        import pdb

        pdb.set_trace()

    if str_split != [] and str_split[0] in ["<bs>", "</bs>"]:
        str_split = str_split[1:]
    if "</bs>" in str_split:
        str_split = str_split[: str_split.index("</bs>")]

    # split according to ";"
    # str_split = slots_string.split(self.BELIEF_STATE_DELIM)
    str_split = " ".join(str_split).split(",")
    if str_split[-1] == "":
        str_split = str_split[:-1]
    str_split = [slot.strip() for slot in str_split]

    for slot_ in str_split:
        slot = slot_.split()
        # ignore cases without proper format and incorrect domains
        if len(slot) > 2 and slot[0] in DOMAINS:
            domain = slot[0]
            # handle cases where slot key contains "book"
            if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                slot_type = slot[1] + " " + slot[2]
                slot_val = " ".join(slot[3:])
            else:
                slot_type = slot[1]
                slot_val = " ".join(slot[2:])

            # any normalizations
            slot_val = SLOT_VAL_CONVERSION.get(slot_val, slot_val)

            # may be problematic to skip these cases
            # if not slot_val == "dontcare":
            slots_list.append(domain + "--" + slot_type + "--" + slot_val)

            # divide by domains and categorize as named entities
            if domain in per_domain_slot_lists:
                per_domain_slot_lists[domain].add(slot_type + "--" + slot_val)
            else:
                per_domain_slot_lists[domain] = {slot_type + "--" + slot_val}
            if domain + "--" + slot_type in NAMED_ENTITY_SLOTS:
                named_entity_slot_lists.append(
                    domain + "--" + slot_type + "--" + slot_val
                )

    for slot in slots_list:
        assert is_proper_slot_format(slot), f"Slot: {slot} is not in proper format."

    return (slots_list, per_domain_slot_lists, named_entity_slot_lists)


def is_proper_slot_format(slot):
    return len(slot.split("--")) == 3


def is_valid_slot_key(slot_key: str) -> bool:
    split = slot_key.split("--")
    if len(split) == 3:
        slot_key = "--".join(split[:2])
    elif len(split) == 2:
        pass
    else:
        logger.error(
            f"Given slot_key is not in correct format: {slot_key}. It must be either <domain>--<slot_key>--<slot_value> or <domain>--<slot_key>"
        )
        return False

    return slot_key in ALL_SLOTS


def find_all_slot_keys(fn: str = None) -> Set:
    """Utility function to find all valid slot values used for ALL_SLOTS

    Args:
        fn (str, optional): data file to extract slot keys from. It should be a json file with {dial id: item} pairs
        where the item contains reference slot predictions as a single string Defaults to None.

    Returns:
        Set: a set of slot key values in the format "<domain>--<slot_key>"
    """

    if fn is None:
        all_slot_keys = set()
        for split in ["train", "valid", "test"]:
            fn = (
                Path(os.environ["DATAPATH"])
                / f"multiwoz_dst/MULTIWOZ2.3/data_reformat_{split}.json"
            )
            slot_keys = find_all_slot_keys(fn)
            all_slot_keys = all_slot_keys.union(slot_keys)

    else:
        all_slot_keys = set()
        with open(fn, "r") as f:
            data = json.load(f)
            for dial_id, item in data.items():
                slots_inf = item["slots_inf"]
                slot_keys, _, _ = extract_slot_from_string(slots_inf)
                for sk in slot_keys:
                    slot_key = "--".join(sk.split("--")[:2])
                    all_slot_keys.add(slot_key)

    return all_slot_keys


def plot_cjga_trends(df, no_band=True, title="", log_scale=False, no_legend=False):

    sns.set_theme()
    # ci = None if no_band else "sd"
    ci = None if no_band else 98

    rename_columns = {"variable": "CheckDST"}

    df["%"] = df["value"].apply(lambda x: x * 100 if x <= 1 else x)

    for idx, row in df.iterrows():
        df.at[idx, "variable"] = METRIC_MAPPING.get(row["variable"], row["variable"])
    df.rename(rename_columns, axis=1, inplace=True)
    # sns.axes_style("white")
    # sns.set_style("white")
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    # sns.set_theme("white")

    rel = sns.relplot(
        data=df,
        kind="line",
        x="epoch",
        y="%",
        hue=rename_columns.get("variable", "variable"),
        # style=rename_columns.get("variable", "variable"),
        estimator=np.median,
        ci=ci,
        aspect=0.8,
        linewidth=4,
    )
    # plt.legend([],[], frameon=False)

    if log_scale:
        rel.ax.set(xscale="log")
        # f = lambda x:
        # rel.ax.set(xscale="function", functions=())
        # rel.ax.set_xlim(-1,10)
        # rel.ax.set_xticks([0.02*i for i in range(0,50)]+list(range(1,11)))

    rel.ax.lines[0].set_linestyle("--")

    # rel._legend.remove()
    # rel.legend(fontsize=5)
    # leg = rel.ax.legend()
    if not no_legend:
        leg = rel._legend
        leg_lines = leg.get_lines()
        for line in leg_lines:
            line.set_linewidth(4.0)
        leg_lines[0].set_linestyle("--")

    # leg = ax.legend()
    # leg_lines = leg.get_lines()
    # leg_lines[5].set_linestyle(":")

    rel.fig.suptitle(title)


def get_full_df_with_all_results(all_results, epochs):
    # key level of input:  {epoch: {seed}}
    df_list = []
    for epoch in epochs:
        df_by_epoch = pd.DataFrame(all_results[epoch]).T
        df_by_epoch["epoch"] = [epoch] * len(df_by_epoch)
        # df_by_epoch = df_by_epoch.groupby('epoch').agg(["mean", "std"])
        # df_by_epoch = df_by_epoch.groupby('epoch').agg("median")
        df_list.append(df_by_epoch)

    full_df = pd.concat(df_list)
    return full_df


def fix_layers(original_dict):
    # temporary solution to fix old code
    # key level: {run: {epoch: {seed}}}  -> {epoch: {seed}}
    new_dict = defaultdict(dict)
    for run_key, run in original_dict.items():
        for epoch, seed in run.items():

            # new_dict[epoch][k] = v
            for k, v in seed.items():
                new_dict[epoch][k] = v
    return new_dict


def get_interested_cols(prefix: str):
    # prefix should be one of cja_orig, jga, conditional_jga_orig
    interested_cols = [
        # "seed",
        "jga_orig",
        # "jga_valid",
        "coref_jga_orig",
        f"{prefix}_PI",
        f"{prefix}_SDI",
        f"{prefix}_NED",
        "factuality_orig",
        "factuality_NED",
    ]
    return interested_cols


def plot_checkdst_results(
    checkdst_results, epochs, model_name="", prefix="cjga_orig", no_legend=False
):
    # prefix should be one of cja_orig, jga, conditional_jga_orig, all

    full_df = get_full_df_with_all_results(checkdst_results, epochs)
    # full_df = full_df[full_df['jga_orig']>0.5]
    if "convbert" in model_name.lower() or "trippy" in model_name.lower():
        for key in full_df.keys():
            if "factuality" in key:
                # import pdb; pdb.set_trace()
                full_df[key] = [1.0] * len(full_df[key])

    if prefix != "all":
        interested_cols = [(get_interested_cols(prefix), prefix)]
    else:
        interested_cols = [
            (get_interested_cols(pfx), pfx)
            for pfx in ["cjga_orig", "jga", "conditional_jga_orig"]
        ]

    for cols, pfx in interested_cols:
        if "seed" not in full_df:
            full_df = full_df.reset_index()
        full_df.rename(columns={"index": "seed"}, inplace=True)
        # full_df[interested_cols]
        molten_df = full_df.melt("epoch", cols)
        # title=f"{model_name} CheckDST ({pfx})"
        title = f"{model_name}"
        plot_cjga_trends(
            molten_df, no_band=True, log_scale=True, title=title, no_legend=False
        )
    return molten_df


def print_out_paper_results(checkdst_results, epochs):
    full_df = get_full_df_with_all_results(checkdst_results, epochs)
    full_df = full_df[full_df["jga_orig"] > 0.3]
    full_df = full_df.groupby("epoch").agg(["median", "sem", "count"])
    # full_df = full_df.groupby('epoch').agg("median")
    interested_cols = get_interested_cols(prefix="cjga_orig")

    df = full_df[interested_cols]

    return df
