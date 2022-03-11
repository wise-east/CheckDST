import json
from loguru import logger
import re
from tqdm import tqdm
import pandas as pd


def find_step(fn):
    step = re.sub("[^0-9]", "", fn.name)
    if step:
        step = int(step)
    else:
        step = float('inf')
    return step


def sort_by_steps(fns):
    return sorted(fns, key=lambda x: find_step(x))


def find_checkpoint_with_best_cjga(logs, test_reports):
    """
    """
    max_cjga = -1
    max_cjga_log = ""
    for c, t in zip(logs, test_reports):
        assert find_step(c) == find_step(t), (c, t)

        with t.open("r") as f:
            test_report = json.load(f)

        cjga = test_report["report"]["jga_conditional"]
        if cjga > max_cjga:
            max_cjga = cjga
            max_cjga_log = c

    return max_cjga_log, max_cjga


def compare_final_with_largest_step():
    # last logs is not necessarily the same as the final report
    with open(
        '/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1_sd0/model.logs_step34904.NEI_world_logs_fs_False.jsonl',
        "r",
    ) as f:
        last_logs_data = [json.loads(l) for l in f.readlines()]

    with open(
        '/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1_sd0/model.NEI_world_logs_fs_False.jsonl',
        "r",
    ) as f:
        final_data = [json.loads(l) for l in f.readlines()]

    with open(
        '/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1_sd0/model.NEI_report_fs_False.json',
        "r",
    ) as f:
        final_report = json.load(f)

    with open(
        '/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1_sd0/model.logs_step34904.NEI_report_fs_False.json',
        "r",
    ) as f:
        last_logs_report = json.load(f)

    logger.info(last_logs_data == final_data)
    logger.info(final_report == last_logs_report)


def load_jsonl_files(jsonl_fn):
    with open(jsonl_fn, "r") as f:
        data = f.readlines()
        json_data = [json.loads(l) for l in data]

    return json_data


def get_dial_id_from_log_line(log_line):
    curr = log_line['dialog'][0][0]

    dial_id = f"{curr['dial_id']}-{curr['turn_num']}"

    return dial_id


def format_logs_as_dict_list(json_data):
    """
    Transform logs file into a list of dicts that can be used to form a DataFrame    
    """
    df_list = []

    for idx in tqdm(range(0, len(json_data), 2)):
        original = json_data[idx]
        perturbed = json_data[idx + 1]

        orig_label = original['dialog'][0][0]
        orig_pred = original['dialog'][0][1]
        aug_label = perturbed['dialog'][0][0]
        aug_pred = perturbed['dialog'][0][1]

        original_dial_id = get_dial_id_from_log_line(original)
        perturbed_dial_id = get_dial_id_from_log_line(perturbed)
        assert original_dial_id == perturbed_dial_id

        orig_text = orig_label['text']
        aug_text = aug_label['text']

        orig_metrics = orig_pred['metrics']
        aug_metrics = aug_pred['metrics']

        orig_gt_dst = orig_label['eval_labels']
        orig_pred_dst = orig_pred['text']

        aug_gt_dst = aug_label['eval_labels']
        aug_pred_dst = aug_pred['text']

        jga_original = orig_metrics['jga_original']
        jga_perturbed = aug_metrics['jga_perturbed']
        jga_conditional = aug_metrics.get('jga_conditional', -1)

        df_list.append(
            {
                "dialid": original_dial_id,
                "orig_text": orig_text,
                "perturbed_text": aug_text,
                "orig_label": orig_gt_dst[0],
                "perturbed_label": aug_gt_dst[0],
                "orig_pred": orig_pred_dst,
                "perturbed_pred": aug_pred_dst,
                "jga_orig": jga_original,
                "jga_perturbed": jga_perturbed,
                "jga_conditional": jga_conditional,
            }
        )

    return df_list


def get_dialogue_state():

    return


def get_dict_diff():

    return


def compare_two_logs(log1_fn, log2_fn):
    """
    main function for finding examples that get incorrectly predicted in one checkpoint
    while correctly predicted in another 
    """

    log1 = load_jsonl_files(log1_fn)
    log2 = load_jsonl_files(log2_fn)

    log1_df = pd.DataFrame(format_logs_as_dict_list(log1))
    log2_df = pd.DataFrame(format_logs_as_dict_list(log2))

    # find dialogue ids that have different cJGA

    # extract what the differences are

    # generate statistics

    # return the statistics

    return log1_df, log2_df
