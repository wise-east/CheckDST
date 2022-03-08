import json
from pprint import pprint
from loguru import logger
from tqdm import tqdm
from subprocess import run
import shlex

# caveat: test must be done with SD augmentation

# full shot runs

sample_model_path = "/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.3/lr1e-4_ngpu1_bs4_fewshot_False_prompts_True_sd1_2021-11-13_10:31:28/model"

cmd_laug = f"""
parlai eval_model -m bart --model-file {sample_model_path} --datatype test --batchsize 1 --task multiwoz_dst_laug -aug SD  --world-logs dst_laug_test_full.jsonl --test_reduced True
"""

cmd_orig = f"""
parlai eval_model -m bart --model-file {sample_model_path} --datatype test --batchsize 1 --task multiwoz_dst --world-logs dst_test_full.jsonl --test_reduced True
"""

run(shlex.split(cmd_laug))
run(shlex.split(cmd_orig))

# few shot runs
sample_fs_model_path = "/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.1/lr1e-4_ngpu1_bs4_fewshot_True_prompts_True_sd0_2021-11-13_10:25:44/model"

cmd_laug = f"""
parlai eval_model -m bart --model-file {sample_fs_model_path} --datatype test --batchsize 1 --task multiwoz_dst_laug -aug SD  --world-logs dst_laug_test_fs.jsonl --test_reduced True
"""

cmd_orig = f"""
parlai eval_model -m bart --model-file {sample_fs_model_path} --datatype test --batchsize 1 --task multiwoz_dst --world-logs dst_test_fs.jsonl --test_reduced True
"""

run(shlex.split(cmd_laug))
run(shlex.split(cmd_orig))

# template = "/data/home/justincho/ParlAI/dst_{}test_fs_include_empty.jsonl"
# template = "/data/home/justincho/ParlAI/dst_{}test_fs.jsonl"
templates = [
    "/data/home/justincho/ParlAI/data/dst_{}test_full.jsonl",
    "/data/home/justincho/ParlAI/data/dst_{}test_fs.jsonl",
]


def compare(template):
    test_file = template.format("")
    aug_file = template.format("laug_")

    with open(aug_file, "r") as f:
        data_txt = f.read().splitlines()
        aug_result = [json.loads(l) for l in data_txt]

    with open(test_file, "r") as f:
        data_txt = f.read().splitlines()
        test_result = [json.loads(l) for l in data_txt]

    assert len(test_result) * 2 == len(
        aug_result
    ), "Length is not matched. 2X test pred != laug pred"
    for idx, tr in tqdm(enumerate(test_result)):
        aug = aug_result[idx * 2]
        x = tr['dialog'][0][0]
        y = aug['dialog'][0][0]

        # keep only overlapping keys
        x.pop('need_coref')
        x.pop('id')
        y.pop('id')

        # if they are not the same, break
        if x != y:
            for k in x.keys():
                if k not in y:
                    logger.info((k, x[k]))
                if x[k] != y[k]:
                    logger.info((x[k], y[k], k))
            # pprint(x)
            # pprint(y)
            break

        # compare beam text
        x_beam = tr['dialog'][0][1]['beam_texts'][0][0]
        y_beam = aug['dialog'][0][1]['beam_texts'][0][0]

        if x_beam != y_beam:
            logger.info(f"\n\tx: {x_beam}\n\ty: {y_beam}")
            break

        try:
            x_jga = tr['dialog'][0][1]['metrics']['joint goal acc']
        except Exception as e:
            # print(e)
            # pprint(tr['dialog'][0][1]['metrics'])
            # pprint(tr)
            x_jga = -1

        try:
            y_jga = aug['dialog'][0][1]['metrics']['jga_original']
        except Exception as e:
            # print(e)
            # pprint(aug)
            y_jga = -1

        if x_jga != y_jga:
            logger.info(f"\n\tx: {x_jga}\n\ty: {y_jga}")
            pprint(tr)
            pprint(aug)
            break

    logger.info("Clear")


for template in templates:
    compare(template)
