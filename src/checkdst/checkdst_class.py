import pandas as pd
import numpy as np

from loguru import logger
import json
from checkdst.utils import normalize_dial_ids, extract_slot_from_string
from typing import Dict, List
import math


class CheckDST:
    def __init__(self, gold_data_fn: str):
        """Initialize dataframe with original data (reformatted MultiWOZ data)
        TODO: directly load from MultiWOZ 2.3 data to remove dependency with fomratting code

        Args:
            gold_data_fn (str): filepath to original MultiWOZ data
        """
        with open(gold_data_fn, "r") as f:
            self.gold_data = json.load(f)

        # normalize dial_id names
        self.gold_data = {normalize_dial_ids(k): v for k, v in self.gold_data.items()}
        self.df = pd.DataFrame(self.gold_data).T
        self.df.rename(columns={"dial_id": "dial_idx"}, inplace=True)
        self.df.index.name = "dial_id"

        # dictionary for holding key results
        self.checkdst_results = {}

        # required keys for each prediction row to contain
        self.required_keys = {
            "dial_id" "context",
            "pred",
            "gold",
        }

    # @classmethod
    # def aggregate_by_seeds(cls, checkdst_list:List[CheckDST])->CheckDST:

    #     return

    def get_results(self) -> Dict:
        """Return checkdst results

        Returns:
            Dict: checkdst result output
        """
        return self.checkdst_results

    def __str__(self):
        output_str = ""
        for k, v in sorted(self.checkdst_results.items(), key=lambda x: x[0]):
            output_str += f"{k}: {v}\n"
        return output_str

    def add_preds(self, pred_fn: str, aug_type: str, compute: bool = True) -> None:
        """Read CheckDST formatted jsonl files and add them to dataframe

        Args:
            pred_fn (str): filepath to CheckDST formatted jsonl file
            aug_type (str): type of augmentation used
        """
        # load data
        try:
            preds = pd.read_json(pred_fn, orient="records", lines=True)
        except Exception as e:
            logger.error(e)
            logger.error(f"Failed to load {str(pred_fn)}")
            return

        # format columns
        preds.set_index("dial_id", inplace=True)
        preds.rename(columns={k: f"{k}_{aug_type}" for k in preds.keys()}, inplace=True)
        prev_len = len(self.df)

        # add to dataframe
        # import pdb; pdb.set_trace()
        self.df = self.df.join(preds, on="dial_id")

        # make sure that the total number of rows doesn't change after merge
        if prev_len != len(self.df):
            logger.error("The dataframe's length should not have increased in size.")

        # compute metrics
        if compute:
            self._compute_metrics(aug_type)

    def _compute_metrics(self, aug_type: str) -> Dict:
        """Key function for computing CheckDST metrics (except for cJGA)
        This function is not for direct invocation.

        Args:
            aug_type (str): type of augmentation. carried over from add_preds

        Returns:
            Dict: returns the updated checkdst_results dictionary
        """

        # format predictions for computing metrics
        pred_col_key = f"pred_{aug_type}"
        gold_col_key = f"gold_{aug_type}"
        self.df[pred_col_key] = self.df[pred_col_key].apply(
            lambda x: extract_slot_from_string(x) if isinstance(x, str) else np.NaN
        )
        self.df[gold_col_key] = self.df[gold_col_key].apply(
            lambda x: extract_slot_from_string(x) if isinstance(x, str) else np.NaN
        )

        jgas = []
        coref_jgas = []
        hallucinate = []
        pred_named_entity_cts = []
        for idx, row in self.df.iterrows():

            # ignore any rows without a prediction for the given dial_id. this is the case if there were no augmentations in the original dataset
            if not isinstance(row[pred_col_key], tuple) and (
                math.isnan(row[pred_col_key]) or math.isnan(row[gold_col_key])
            ):
                jgas.append(np.NaN)
                coref_jgas.append(np.NaN)
                hallucinate.append(0)
                pred_named_entity_cts.append(0)
                continue
            slots_pred = row[pred_col_key][0]
            slots_truth = row[gold_col_key][0]

            # calculate jga and coref jga
            jga = set(slots_pred) == set(slots_truth)
            jgas.append(jga)
            if row["need_coref"]:
                coref_jgas.append(jga)
            else:
                # filler to match number of rows
                coref_jgas.append(np.NaN)

            # calculate hallucination
            slots_pred_named_entity = row[pred_col_key][2]
            context = row[f"context_{aug_type}"]

            hallucinate_ct = 0
            # iterate through named entity slots
            for predicted_slot in slots_pred_named_entity:
                curr_domain = predicted_slot.split("--")[0]
                ne = predicted_slot.split("--")[-1]

                # keep only the slot values
                for tmp_slot in slots_truth:
                    slot_name = tmp_slot.split("--")[0] + " " + tmp_slot.split("--")[1]
                    ne = ne.replace(slot_name, "")
                for tmp_slot in slots_pred:
                    slot_name = tmp_slot.split("--")[0] + " " + tmp_slot.split("--")[1]
                    ne = ne.replace(slot_name, "")

                # get combined hallucination
                hallucinate_ct += not (ne in context)
            hallucinate.append(hallucinate_ct)
            pred_named_entity_cts.append(len(slots_pred_named_entity))

        jga_key = f"jga_{aug_type}"
        coref_jga_key = f"coref_jga_{aug_type}"
        hallucinate_cts_key = f"hallucinate_cts_{aug_type}"
        pred_named_entity_cts_key = f"pred_named_entity_cts_{aug_type}"

        # assign new columns
        self.df[jga_key] = jgas
        self.df[coref_jga_key] = coref_jgas
        self.df[hallucinate_cts_key] = hallucinate
        self.df[pred_named_entity_cts_key] = pred_named_entity_cts

        # compute metrics
        total_jga = self.df[jga_key].mean()
        total_coref_jga = self.df[coref_jga_key].mean()
        total_times_hallucinated = self.df[hallucinate_cts_key].sum()
        total_named_entities_predicted = self.df[pred_named_entity_cts_key].sum()
        hallucination_frequency = (
            total_times_hallucinated / total_named_entities_predicted
        )

        # store results in results dict
        self.checkdst_results[jga_key] = round(total_jga, 4)
        self.checkdst_results[coref_jga_key] = round(total_coref_jga, 4)
        self.checkdst_results[f"factuality_{aug_type}"] = round(
            1 - hallucination_frequency, 4
        )
        self.checkdst_results[
            f"hallucination_cts_{aug_type}"
        ] = total_times_hallucinated
        self.checkdst_results[
            f"pred_named_entity_cts_{aug_type}"
        ] = total_named_entities_predicted
        self.checkdst_results[f"coref_ct_{aug_type}"] = len(
            self.df[~self.df[coref_jga_key].isna()]
        )  # track number of cases that required coreference resolution

        # import pdb; pdb.set_trace()

        return self.checkdst_results

    def compute_cjga(self, orig: str, aug: str) -> Dict:
        """Compute cJGA (consistent JGA), which is the frequency of getting both original and augmented version correct

        Args:
            orig (str): column tag for original input JGA
            aug (str): column tag for augmented JGA

        Returns:
            Dict: _description_
        """
        orig_jga_key = f"jga_{orig}"
        aug_jga_key = f"jga_{aug}"

        # only compute cjga if both keys are present. notify which keys are not present.
        if orig_jga_key not in self.df.keys() or aug_jga_key not in self.df.keys():
            additional_info = ""
            if orig_jga_key not in self.df.keys():
                additional_info += f"{orig_jga_key} is not present."
            if aug_jga_key not in self.df.keys():
                additional_info += f" {aug_jga_key} is not present."
            logger.error(
                f"Both keys {orig_jga_key} and {aug_jga_key} must be present. {additional_info}"
            )
            return

        cjgas = []
        conditional_jgas = []
        for idx, row in self.df.iterrows():

            # ignore any rows without a jga for the given dial_id. this is the case if there were no augmentations in the original dataset
            if math.isnan(row[orig_jga_key]) or math.isnan(row[aug_jga_key]):
                cjgas.append(np.NaN)
                conditional_jgas.append(np.NaN)
                continue

            # consistent JGA: 1 if orig_jga = 1 & aug_jga=1 (and automatically orig_jga = aug_jga)
            cjga = row[orig_jga_key] and row[aug_jga_key]
            cjgas.append(cjga)

            # ---OLD--- conditional JGA: 1 if orig_jga =1 or aug_jga =1 and (orig_jga = aug_jga)
            if row[orig_jga_key] or row[aug_jga_key]:
                conditional_jga = row[orig_jga_key] and row[aug_jga_key]
                conditional_jgas.append(conditional_jga)
            else:
                conditional_jgas.append(np.NaN)

        cjga_key_name = f"cjga_{orig}_{aug}"
        conditional_jga_key_name = f"conditional_jga_{orig}_{aug}"
        self.df[cjga_key_name] = cjgas
        self.df[conditional_jga_key_name] = conditional_jgas

        self.checkdst_results[cjga_key_name] = round(self.df[cjga_key_name].mean(), 4)
        self.checkdst_results[conditional_jga_key_name] = round(
            self.df[conditional_jga_key_name].mean(), 4
        )

        # import pdb; pdb.set_trace()
        return self.checkdst_results

    def _check_fields(self, item: Dict) -> bool:
        """Check if the fields in each jsonl line has the necessary fields

        Args:
            item (Dict): dictionary from each line of jsonl file

        Returns:
            bool: True if all conditions are met
        """
        for k in self.required_keys():
            if k not in item.keys():
                return False
        raise True

    def find_cjga_examples(self):
        """Find cases where jga_aug = 1 but jga_orig = 0 to argue for cjga"""

        raise NotImplementedError
