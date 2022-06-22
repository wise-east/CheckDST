import glob
import pandas as pd
from typing import List, Dict
from loguru import logger


def get_pred_dfs(main_dir: str, inv: str, step: str) -> pd.DataFrame:
    """
    load prediction cvs files ad dataframe
    """

    pred_files = glob.glob(f"{main_dir}/**/pred_res.test.{step}{inv}.csv")

    all_results = [pd.read_csv(f) for f in pred_files]

    return all_results


def format_belief_state(row):
    """
    return belief states in domain slot_type slot_value format
    """

    gold_bs = ""
    pred_bs = ""
    for k in sorted(row.keys()):
        if "-" not in k and "_" not in k:
            continue
        # print(k, row[k])
        if row[k] != "none":
            domain, slot_type = k.split("_")[0].split("-")
            bs = f"{domain} {slot_type} {row[k]}, "
            if "gold" in k:
                gold_bs += bs
            if "pred" in k:
                pred_bs += bs

    return gold_bs, pred_bs


def get_common_rows(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    all the dfs should be the same structure
    """

    assert df_list != []

    if len(df_list) == 1:
        logger.warning("Asked to merge but was given a single dataframe in list")
        return df_list[0]

    for idx in range(len(df_list) - 1):
        if idx == 0:
            merged = df_list[idx].merge(df_list[idx + 1]).dropna(axis=0)
        else:
            merged = merged.merge(df_list[idx + 1]).dropna(axis=0)

    return merged
