import json
from pathlib import Path
from utils import (
    read_zipped_json,
    normalize_dialogue_ids,
    test_match,
    replace_dict_values_str,
)
from loguru import logger
import os

from parlai.tasks.multiwoz_checkdst.mutator import EntityMutator
import numpy as np
from argparse import ArgumentParser
from pprint import pprint
from tqdm import tqdm
from loguru import logger
from collections import Counter

SEED = 42


DATADIR = os.environ.get("DATADIR", "")


def check_key_match(sd_test, orig_data):
    # check whether keys match for each item
    sd_sample = sd_test["PMUL4648.json"]["log"][5]
    orig_sample = orig_data["PMUL4648.json"]["log"][5]
    if "turn_id" in orig_sample:
        orig_sample.pop("turn_id")
    if "originalText" in sd_sample:
        sd_sample.pop("originalText")
    for k in orig_sample:
        if k not in sd_sample:
            print(k)

    for k in sd_sample:
        if k not in orig_sample:
            print(k)


def find_unchanged_dialogue_ids(augmented_data, orig_data):
    """Find dialogue IDs where there were no changes to the utterance

    Args:
        augmented_data (_type_): _description_
        orig_data (_type_): _description_

    Returns:
        List[str]: list of dialogue ids with no changes to the user utterance
    """
    count = 0
    total = 0
    same_dialog_ids = []
    for k, v in augmented_data.items():
        tp_sample = v
        orig_sample = orig_data[k]
        len_tp = len(tp_sample["log"])
        len_orig = len(orig_sample["log"])
        assert len_tp == len_orig, (k, len_tp, len_orig)

        for idx, log in enumerate(tp_sample["log"]):
            assert log["metadata"] == orig_sample["log"][idx]["metadata"]
            if idx % 2 == 0 and log["text"] != orig_sample["log"][idx]["text"]:
                count += 1
            if idx % 2 == 0 and log["text"] == orig_sample["log"][idx]["text"]:
                same_dialog_ids.append(f"{k}-{idx}")
            if idx % 2 == 0:
                total += 1

            # assert log['span_info'] == orig_sample['log'][idx]['span_info']

    logger.info(f"Fraction of properly paraphrased turns: {count / total*100:.2f}%")
    logger.info(f"Total number of unchanged turns: {len(same_dialog_ids)}")

    return same_dialog_ids


def store_unchanged_dialogue_ids():

    orig_fn = os.path.join(DATADIR, "multiwoz_dst/MULTIWOZ2.3/test_dials.json")
    tp_fn = os.path.join(DATADIR, "checkdst/PI/test.json.zip")
    sd_fn = os.path.join(DATADIR, "checkdst/SDI/test.json.zip")

    with open(orig_fn) as f:
        orig = json.load(f)

    # load zipped data directly
    tp_test = read_zipped_json(tp_fn, str(Path(Path(tp_fn).name).with_suffix("")))
    sd_test = read_zipped_json(sd_fn, str(Path(Path(sd_fn).name).with_suffix("")))

    # check whether all dialogue ids match
    test_match(orig, tp_test)
    test_match(orig, sd_test)

    orig_data = normalize_dialogue_ids(orig)
    sd_test = normalize_dialogue_ids(sd_test)
    tp_test = normalize_dialogue_ids(tp_test)

    tp_same_dialog_ids = find_unchanged_dialogue_ids(tp_test, orig_data)
    sd_same_dialog_ids = find_unchanged_dialogue_ids(sd_test, orig_data)

    # keep track of these: they are excluded from cJGA calculations
    with open("checkdst/PI/no_change_ids.txt", "w") as f:
        for id in tp_same_dialog_ids:
            f.write(id + "\n")

    with open("checkdst/SDI/no_change_ids.txt", "w") as f:
        for id in sd_same_dialog_ids:
            f.write(id + "\n")


class NED_processor:
    def __init__(self, original_datapath, testlist_filepath):

        self.named_entity_slots_for_trippy = {
            "attraction-name",
            "restaurant-name",
            "hotel-name",
            "taxi-departure",
            "taxi-destination",
            "train-departure",
            "train-destination",
        }
        # enforce same mutation as ParlAI agent

        rng = np.random.RandomState(SEED)
        self.mutator = EntityMutator(opt=None, rng=rng)

        with open(original_datapath, "r") as f:
            self.orig = json.load(f)

        with open(testlist_filepath, "r") as f:
            self.test_convs = f.readlines()
            self.test_convs = [t.replace("\n", "") for t in self.test_convs]
            assert (
                len(self.test_convs) == 1000
            ), f"Number of test conversations should be 1000, but got {len(self.test_convs)}"

    def replace_named_entities(self):
        swapped_ct = 0
        change = False
        self.ned_conversion_map = {}
        self.ned_orig = self.orig.copy()
        slot_vals = set()
        self.change_list = []
        n_text_examples = 0
        print_ = False
        orig_values = set()
        train_destination = []
        turn_count = 0
        for dial_id, dial in tqdm(self.ned_orig.items()):

            # if dial_id not in test_convs:
            #     continue
            def replace_with_scrambled_entity(
                domain, slot_type, slot_value, print_=False
            ):
                slot_value = slot_value.lower()
                changed_inner = False
                if f"{domain}-{slot_type}" in self.named_entity_slots_for_trippy:
                    changed_inner = True
                    # if f"{domain}-{slot_type}" == "train-destination":
                    if f"{domain}-{slot_type}" == "restaurant-name":
                        train_destination.append(slot_value)

                    # print(f"{domain}-{slot_type}-{slot_value}")
                    orig_values.add(slot_value)

                    if "".join(sorted(slot_value)) not in self.ned_conversion_map:
                        new_v = self.mutator.scramble(slot_value)
                        self.ned_conversion_map["".join(sorted(slot_value))] = new_v
                    else:
                        new_v = self.ned_conversion_map["".join(sorted(slot_value))]

                    # go through all turns and replace if found
                    for idx2 in range(len(dial["log"])):
                        # update only if the golden slot value is actually found in the text
                        if slot_value.lower() in dial["log"][idx2]["text"].lower():
                            # update text
                            dial["log"][idx2]["text"] = (
                                dial["log"][idx2]["text"]
                                .lower()
                                .replace(slot_value, new_v)
                            )
                            # update metadata
                            replace_dict_values_str(
                                dial["log"][idx2]["metadata"], slot_value, new_v
                            )
                            changed_inner = True

                            if print_:
                                print(dial["log"][idx2]["text"])
                                print(f"{domain} {slot_type} {slot_value}")
                                print(f"{domain} {slot_type} {new_v}")

                return changed_inner

            for idx in reversed(range(0, len(dial["log"]), 2)):
                turn_count += 1
                user_utt = dial["log"][idx]["text"]
                sys_utt = dial["log"][idx + 1]["text"]
                metadata = dial["log"][idx + 1]["metadata"]

                changed_outer = False
                for domain, slots in metadata.items():
                    # import pdb; pdb.set_trace()
                    for k, v in slots["book"].items():
                        if k == "booked":
                            if v == []:
                                continue
                            else:
                                for booked_places in slots["book"]["booked"]:
                                    for slot_type, slot_value in booked_places.items():
                                        slot_vals.add(slot_value)
                                        changed_outer = (
                                            changed_outer
                                            or replace_with_scrambled_entity(
                                                domain,
                                                slot_type,
                                                slot_value,
                                                print_=print_,
                                            )
                                        )

                        elif v and v != "not mentioned":
                            # print(f"{domain} {k} {v}")
                            # pass
                            changed_outer = (
                                changed_outer
                                or replace_with_scrambled_entity(
                                    domain, k, v, print_=print_
                                )
                            )

                    for k, v in slots["semi"].items():
                        if v and v != "not mentioned":
                            # pass
                            changed_outer = (
                                changed_outer
                                or replace_with_scrambled_entity(
                                    domain, k, v, print_=print_
                                )
                            )
                n_text_examples += dial_id in self.test_convs
                if changed_outer and dial_id in self.test_convs:
                    dial_full_id = f"{dial_id.lower().replace('.json', '')}-{idx//2}"
                    self.change_list.append(dial_full_id)

        len(orig_values), len(slot_vals), len(train_destination), turn_count
        Counter(train_destination).most_common(10)
        len(self.ned_conversion_map), n_text_examples, len(self.change_list)

        assert n_text_examples == 7372


def test_replacing_entities():
    sample = {
        "dialog_act": {"Booking-Request": [["Stay", "?"], ["Day", "?"]]},
        "metadata": {
            "hotel": {
                "book": {"booked": [], "day": "monday", "people": "", "stay": ""},
                "semi": {
                    "area": "centre",
                    "internet": "dontcare",
                    "name": "alexander bed and breakfast",
                    "parking": "not mentioned",
                    "pricerange": "not mentioned",
                    "stars": "not mentioned",
                    "type": "guesthouse",
                },
            },
            "train": {
                "book": {"booked": [], "people": ""},
                "semi": {
                    "arriveBy": "",
                    "day": "",
                    "departure": "",
                    "destination": "",
                    "leaveAt": "",
                },
            },
        },
        "span_info": [],
        "text": "what day would you like to book on and for how long ?",
        "turn_id": 5,
    }

    pprint(sample)
    replace_dict_values_str(sample, "alexander bed and breakfast", "whatever")
    pprint(sample)


# for testing purposes
def compare_with_reformatted_data():
    with open("multiwoz_dst/MULTIWOZ2.3/data_reformat.json", "r") as f:
        rdata = json.load(f)
        logger.info(f"Total number of turns in reformatted dataset: {len(rdata)}")

    tc = []
    for k, v in rdata.items():
        logger.debug(f"{k, v}")
        for slot in v["slots_inf"].split(","):
            slot_type = "train_destination"
            # slot_type = "restaurant-name"
            if slot.strip == "":
                continue
            elif "-".join(slot.split()[:2]) == slot_type:
                tc.append(" ".join(slot.split()[2:]))

    logger.info(len(tc))
    Counter(tc).most_common(10)


def check_in_dict(ned_conversion_map):

    ne_samples = [
        "birmingham new street",
        "golden house",
        "express by holiday inn cambridge",
        "the cambridge belfry",
        "Nandos City",
        "alexander bed and breakfast",
        "the hotspot",
        "loch fine",
    ]

    for ne_sample in ne_samples:
        print("".join(sorted(ne_sample.lower())) in ned_conversion_map)


def examine_dialogue_acts():
    """
    Legacy code for examining whether dialogue acts should be updated
    Verdict: does not have to be updated at inference time (for test set)
    Will need to be done if anyone wants to train a TripPy model with replaced named entities
    """

    ACTS_DICT = {
        "taxi-depart": "taxi-departure",
        "taxi-dest": "taxi-destination",
        "taxi-leave": "taxi-leaveAt",
        "taxi-arrive": "taxi-arriveBy",
        "train-depart": "train-departure",
        "train-dest": "train-destination",
        "train-leave": "train-leaveAt",
        "train-arrive": "train-arriveBy",
        "train-people": "train-book_people",
        "restaurant-price": "restaurant-pricerange",
        "restaurant-people": "restaurant-book_people",
        "restaurant-day": "restaurant-book_day",
        "restaurant-time": "restaurant-book_time",
        "hotel-price": "hotel-pricerange",
        "hotel-people": "hotel-book_people",
        "hotel-day": "hotel-book_day",
        "hotel-stay": "hotel-book_stay",
        "booking-people": "booking-book_people",
        "booking-day": "booking-book_day",
        "booking-stay": "booking-book_stay",
        "booking-time": "booking-book_time",
    }

    # need to update the dialogue_acts.json file as well?? for inference, no.
    with open(
        os.path.join(DATADIR, "multiwoz_dst/MULTIWOZ2.3/dialogue_acts.json"), "r"
    ) as f:
        dacts = json.load(f)

    # load_acts can be found in trippy-public-master/dataset_multiwoz21.py
    # s_dict = load_acts("multiwoz_dst/MULTIWOZ2.3/dialogue_acts.json")

    slot_counter = Counter()
    slotname_counter = Counter()
    slotvalue_counter = Counter()
    count = 0
    not_found_count = 0
    found_count = 0
    for dial_id, turn_acts in tqdm(dacts.items()):
        # print(dial_id + ".json")
        # break

        if dial_id + ".json" not in test_convs:
            continue
        else:
            count += 1

        for turn, acts in turn_acts.items():

            if isinstance(acts, dict):
                for slot, slot_vals in acts.items():
                    for sv in slot_vals:
                        # print(slot, sv)
                        slot_counter[f"{slot}-{sv[0]}-{sv[1]}"] += 1
                        slotname_counter[f"{slot}-{sv[0]}"] += 1

                        if (
                            f"{slot}-{sv[0]}" == "Attraction-Inform-Name"
                            and sv[1] != "None"
                        ):
                            slotvalue_counter[sv[1]] += 1
                            if "".join(sorted(sv[1].lower())) not in ned_conversion_map:
                                logger.info((sv[1], "".join(sorted(sv[1].lower()))))
                                not_found_count += 1
                            else:
                                found_count += 1
                            # assert "".join(sorted(sv[1].lower())) in nei_conversion_map, (sv[1], "".join(sorted(sv[1].lower())))
            else:
                if acts == "No Annotation":
                    continue
                print(f"Not a dict: {acts}")

        # print(acts)
        # break
    count, found_count, not_found_count


if __name__ == "__main__":
    # store_unchanged_dialogue_ids()
    original_datapath = os.path.join(DATADIR, "multiwoz_dst/MULTIWOZ2.3/data.json")
    testlist_filepath = os.path.join(
        DATADIR, "multiwoz_dst/MULTIWOZ2.1/testListFile.json"
    )
    ned_processor = NED_processor((original_datapath), testlist_filepath)
    ned_processor.replace_named_entities()

    # compare with dialids that were changed for ParlAI
    with open(
        os.path.join(
            DATADIR, "checkdst/NED/parlai_changed_dialids_test_fewshot_False.txt"
        ),
        "r",
    ) as f:
        changed_parlai_ids = f.readlines()
        changed_parlai_ids = [c.replace("\n", "") for c in changed_parlai_ids]

    logger.info(
        f"Number of dial ids changed for original data.json: {len(ned_processor.change_list)}"
    )
    logger.info(
        f"Number of dial ids changed for reformatted ParlAI data: {len(changed_parlai_ids)}"
    )

    in_both = set(ned_processor.change_list).intersection(changed_parlai_ids)
    logger.info(
        f"Only their intersection will be used to evaluate the models: {len(in_both)}"
    )

    with open(
        os.path.join(DATADIR, "checkdst/NED/both_changed_dial_ids.txt"), "w"
    ) as f:
        for l in list(in_both):
            f.write(l + "\n")
