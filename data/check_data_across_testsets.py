# for comparing test set used for invariance metrics and the one used for training (in original multiwoz_dst/MULTIWOZ2.3)
# for all test sets of the invariances, check if all slot predictions and original context is actually the same as those in the original test set.

import json
from loguru import logger


# load official labels from reformatted data
with open("multiwoz_dst/MULTIWOZ2.3/data_reformat_test.json", "r") as f:
    official = json.load(f)


augs = ["TP", "SD", "orig"]

for aug in augs:
    logger.info(f"Invariance being checked: {aug}")
    with open(f"laug_dst/{aug}/data_reformat_official_v2.3_slots_test.json", "r") as f:
        # with open(f"laug_dst/{aug}/data_reformat_test.json", "r") as f:
        laug_orig = json.load(f)

    off_keys = set(official.keys())
    laug_orig_keys = set(laug_orig.keys())

    extra_in_off = off_keys - laug_orig_keys
    extra_in_laug = laug_orig_keys - off_keys

    if extra_in_off:
        logger.info(f"Extra in official: {len(extra_in_off)}")
        # for k in extra_in_off:
        #     logger.info(k)
        #     logger.info(official[k])

    if extra_in_laug:
        logger.info(f"Extra in LAUG: {len(extra_in_laug)}")
        # for k in extra_in_laug:
        #     logger.info(k)
        #     logger.info(laug_orig[k])

    # for those that intersect, check that the labels are the same
    intersection = off_keys.intersection(laug_orig_keys)
    count = 0
    for k in sorted(intersection):
        for key in official[k]:
            # if not original, comparison of the context key should be with "orig_context"
            if aug != "orig" and key == "context":
                laug_key = "orig_context"
            # for other keys, compare with same key
            else:
                laug_key = key
            # check for differences
            if official[k][key] != laug_orig[k][laug_key]:
                count += 1
                if key == "context":
                    logger.info(f"ID: {k}")
                    logger.info(f'Official labels: \n\t {official[k]["slots_inf"]}')
                    logger.info(f'LAUG labels: \n\t {laug_orig[k]["slots_inf"]}')
                    for idx in range(len(official[k][key])):
                        if official[k][key][idx] != laug_orig[k][laug_key][idx]:
                            logger.info(official[k][key])
                            logger.info(laug_orig[k][laug_key])
                            # logger.info(official[k][key][max(idx - 3, 0) :])
                            # logger.info(laug_orig[k][laug_key][max(idx - 3, 0) :])
                            break
                else:
                    logger.info(f"{k} {key}")
                    logger.info(f"Official {key}: \n\t{official[k][key]}")
                    logger.info(f"LAUG {key}: \n\t{laug_orig[k][laug_key]}")
        # if count == 1:
        #     break

    if count == 0:
        logger.info(f"All clear for {aug}")
