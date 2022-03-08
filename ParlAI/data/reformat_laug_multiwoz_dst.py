import json
from utils import read_zipped_json, my_strip, normalize_text
import os
from tqdm import tqdm
from loguru import logger
from pprint import pprint

keys = ["train", "val", "test"]

invs = ["SD", "TP", "orig"]

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laug_dst")

# load official data to use for original context if it's not given in the laug file due to some bug
official_data = {}
for k in keys:
    name = k if k != "val" else "valid"
    with open(f"multiwoz_dst/MULTIWOZ2.3/data_reformat_{name}.json", "r") as f:
        official = json.load(f)
    official_data[k] = official

for inv in invs:
    sub_dir = inv

    for key in keys:
        data = read_zipped_json(
            os.path.join(data_dir, sub_dir, key + '.json.zip'), key + '.json'
        )
        print('load {}, size {}'.format(key, len(data)))

        results = {}
        skipped_error = 0
        for title, sess in tqdm(data.items()):
            logs = sess['log']
            context = ""
            title = title.replace(".json", "")
            # this is for keeping track of augmented conversation if there is any augmentation for 'text' field such that is different from the 'originalText' field
            context_ = ""

            slots = {}
            for i, diag in enumerate(logs):
                # format utterance
                text = normalize_text(diag['text'])

                # odd turns are user turns. add DST example
                if i % 2 == 0:
                    context += "<user> " + text + " "
                    turn_num = int(i / 2)
                    # ignore slots. will get it from v23 anyways
                    turn = {
                        'turn_num': turn_num,
                        'dial_id': title.lower() + ".json",
                        "context": my_strip(context),
                    }
                    sample_name = turn['dial_id'] + f"-{turn_num}"

                    # if we are reformatting a datset with augmentation, keep the original text as well
                    if sub_dir != "orig":
                        orig_text = diag.get("originalText", "")
                        if orig_text is None:
                            orig_text = ""
                        orig_text = normalize_text(orig_text)
                        if orig_text == "":
                            # logger.info("Instance without 'originalText' field found: ")
                            # import pdb; pdb.set_trace()
                            # pprint(diag)
                            # load original context from official data and use it.
                            context_ = official_data[key][
                                turn['dial_id'] + f"-{turn_num}"
                            ]['context']
                            orig_text = context_.split("<user>")[-1]
                        else:
                            context_ += "<user> " + orig_text + " "

                        # make sure that the original and the augmented version are not the same
                        # if they are the same, exclude
                        if orig_text.replace(" ", "") == text.replace(" ", ""):
                            # import pdb; pdb.set_trace()
                            # pprint(diag)
                            skipped_error += 1
                            continue
                        turn["orig_context"] = my_strip(context_)

                    # if it's the original, there is no difference
                    else:
                        turn["orig_context"] = turn["context"]

                    results[sample_name] = turn

                else:
                    context += "<system> " + text + " "
                    context_ += "<system> " + text + " "

        if key == "val":
            key = "valid"
        logger.info(f"# {key} examples: {len(results)}")
        logger.info(f"# skipped because of no augmentation: {skipped_error}")
        with open(
            os.path.join(data_dir, sub_dir, f"data_reformat_{key}.json"), "w"
        ) as f:
            json.dump(fp=f, obj=results, indent=2, sort_keys=True)
