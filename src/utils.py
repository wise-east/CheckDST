import json 
import pandas as pd 
import re 
import tqdm 
from loguru import logger 

DOMAINS = [
    "attraction",
    "hotel",
    "hospital",
    "restaurant",
    "police",
    "taxi",
    "train",
]
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


def find_step(fn):
    step = re.sub("[^0-9]", "", fn.name)
    if step:
        step = int(step)
    else:
        step = float('inf')
    return step


def sort_by_steps(fns):
    return sorted(fns, key=lambda x: find_step(x))


# load jsonl files 
def load_jsonl(fn):
    with open(fn, "r") as f: 
        data = [json.loads(l) for l in f.readlines()] 
    return data 


def normalize_dial_ids(dial_id): 
    return dial_id.replace(".json", "").lower() 

class CheckDST: 
    
    def __init__(self, gold_data_fn:str): 
        
        with open(gold_data_fn, "r") as f: 
            self.gold_data = json.load(f)    

        # normalize dial_id names 
        """Format of dataframe
        index: normalized dial_id 
        columns: items 
        """
        self.gold_data = {normalize_dial_ids(k) :v for k, v in self.gold_data.items()}
        self.df = pd.DataFrame(self.gold_data).T
        self.df.rename(columns={"dial_id": "dial_idx"}, inplace=True)
        self.df.index.name = "dial_id"
        
    def add_preds(self, pred_fn:str, aug_type:str=None)->None: 
        preds = pd.read_json(pred_fn, orient='records', lines=True)
        # add to df 
        preds.set_index("dial_id", inplace=True)
        preds.rename(columns={k: f"{k}_{aug_type}" for k in preds.keys()}, inplace=True)
        preds.head()
        prev_len = len(self.df)
        self.df = self.df.join(preds, on="dial_id")
        import pdb; pdb.set_trace()
    
        # make sure that the total number of rows doesn't change
        if prev_len != len(self.df): 
            logger.error("The dataframe's length should not have increased in size.")
            import pdb; pdb.set_trace() 

        # compute metrics 
        self._compute_metrics()
        
        
    def _compute_metrics(): 
        
        return 

    def _check_fields(): 
        """
        Check if the prediction files' input fields have the necessary values 
        """
        return 

    def _extract_slot_from_string(self, slots_string):
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
        str_split = slots_string.strip().split()
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
            if len(slot) > 2 and slot[0] in DOMAINS:
                domain = slot[0]
                if slot[1] == "book" and slot[2] in ["day", "time", "people", "stay"]:
                    slot_type = slot[1] + " " + slot[2]
                    slot_val = " ".join(slot[3:])
                else:
                    slot_type = slot[1]
                    slot_val = " ".join(slot[2:])
                slot_val = SLOT_VAL_CONVERSION.get(slot_val, slot_val)
                # may be problematic to skip these cases
                if not slot_val == "dontcare":
                    slots_list.append(domain + "--" + slot_type + "--" + slot_val)
                    
                if domain in per_domain_slot_lists:
                    per_domain_slot_lists[domain].add(slot_type + "--" + slot_val)
                else:
                    per_domain_slot_lists[domain] = {slot_type + "--" + slot_val}
                if domain + "--" + slot_type in NAMED_ENTITY_SLOTS:
                    named_entity_slot_lists.append(
                        domain + "--" + slot_type + "--" + slot_val
                    )


        return (
            slots_list,
            per_domain_slot_lists,
            named_entity_slot_lists
        )


def custom_evaluation(
    self
):

    # to switch between computing paraphrase invariance and regular scores
    if self.keep_original: 
        self.flag_compute = 1 - self.flag_compute

    resp = model_response.get('text', "")
    # if not resp:
    #     return
    # import pdb; pdb.set_trace()
    # extract ground truth from labels
    (
        slots_truth,
        slots_truth_per_domain,
        slots_truth_named_entity,
        slots_truth_named_entity_interested,
    ) = self._extract_slot_from_string(labels[0])
    # extract generated slots from model_response
    (
        slots_pred,
        slots_pred_per_domain,
        slots_pred_named_entity,
        slots_pred_named_entity_interested,
    ) = self._extract_slot_from_string(resp)

    def add_slot_p_r(type: str):
        for gt_slot in slots_truth:
            self.metrics.add(f"slot_r_{type}", AverageMetric(gt_slot in slots_pred))
            curr_domain = gt_slot.split("--")[0]
            self.metrics.add(
                f"{curr_domain}/slot_r_{type}", AverageMetric(gt_slot in slots_pred)
            )
        for predicted_slot in slots_pred:
            self.metrics.add(
                f"slot_p_{type}", AverageMetric(predicted_slot in slots_truth)
            )
            curr_domain = predicted_slot.split("--")[0]
            self.metrics.add(
                f"{curr_domain}/slot_p_{type}",
                AverageMetric(predicted_slot in slots_truth),
            )

    def add_hallucination(type: str):
        for gt_slot in slots_truth_named_entity:
            self.metrics.add(
                f"all_ne/slot_r_{type}", AverageMetric(gt_slot in slots_pred)
            )
            curr_domain = gt_slot.split("--")[0]
            self.metrics.add(
                f"{curr_domain}_ne/slot_r_{type}",
                AverageMetric(gt_slot in slots_pred),
            )
        for predicted_slot in slots_pred_named_entity:
            self.metrics.add(
                f"all_ne/slot_p_{type}",
                AverageMetric(predicted_slot in slots_truth),
            )
            curr_domain = predicted_slot.split("--")[0]
            ne = predicted_slot.split("--")[-1]
            for tmp_slot in slots_truth:
                slot_name = tmp_slot.split("--")[0] + " " + tmp_slot.split("--")[1]
                ne = ne.replace(slot_name, "")

            for tmp_slot in slots_pred:
                slot_name = tmp_slot.split("--")[0] + " " + tmp_slot.split("--")[1]
                ne = ne.replace(slot_name, "")

            # print(",,,,,", ne, ",,,,,")
            self.metrics.add(
                f"{curr_domain}_ne/slot_p_{type}",
                AverageMetric(predicted_slot in slots_truth),
            )
            self.metrics.add(
                f"{curr_domain}_ne/hallucination_{type}",
                AverageMetric(not (ne.strip() in teacher_action.get("text"))),
            )

            # get combined hallucination
            self.metrics.add(
                f"all_ne/hallucination_{type}",
                AverageMetric(not (ne.strip() in teacher_action.get("text"))),
            )

    def add_jga(type: str):
        self.metrics.add(f'jga_{type}', AverageMetric(jga_curr))
        self.metrics.add(
            f"named_entities/jga_{type}",
            AverageMetric(
                set(slots_truth_named_entity) == set(slots_pred_named_entity)
            ),
        )
        for domain in slots_truth_per_domain:
            if domain in slots_pred_per_domain:
                self.metrics.add(
                    f"{domain}/jga_{type}",
                    AverageMetric(
                        slots_truth_per_domain[domain]
                        == slots_pred_per_domain[domain]
                    ),
                )

    jga_curr = set(slots_truth) == set(slots_pred)
    # print out when predictions are wrong
    if jga_curr == False:
        tag = "perturbed" if self.flag_compute else "orig"
        # logging.info(f"{tag}\n\tteacher_action: {teacher_action}")
        logging.info(f"\tslots_truth: {slots_truth}\n\tslots_pred: {slots_pred}")

    # import pdb

    # pdb.set_trace()

    # metrics on original test set
    if self.flag_compute == 0:
        add_jga("original")
        add_slot_p_r("original")
        add_hallucination("original")
        self.metrics.add("ct_original", SumMetric(1))

    # # no need to calculate any other metrics for regular test set
    # if self.data_aug == "orig":
    #     return

    # # now calculated externally in order to prevent redundant predictions on the original test set 
    # # metrics on the perturbed version of the test set
    # if self.flag_compute:
    #     self.metrics.add("ct_augment", SumMetric(1))
    #     self.metrics.add(
    #         f'consistency', AverageMetric(slots_pred == self.slots_pred_prev)
    #     )
    #     # the slots should be the same
    #     # assert slots_truth == self.slots_truth_prev, (slots_truth, self.slots_truth_prev)
    #     add_jga(type="perturbed")
    #     add_slot_p_r(type="perturbed")
    #     add_hallucination("perturbed")

    #     # conditional metrics (conditioned on the original prediction being correct)
    #     if self.jga_prev:
    #         add_jga(type="conditional")
    #         add_slot_p_r(type="conditional")
    #         add_hallucination("conditional")

    #     # conditional metrics (conditioned on any of the perturbed or original being correct)
    #     if jga_curr or self.jga_prev:
    #         add_jga(type="new_conditional")

    # # combined metrics (original + perturbed)
    # add_slot_p_r(type="all")
    # add_jga(type="all")
    # add_hallucination("all")

    # self.jga_prev = jga_curr
    # self.slots_truth_prev = slots_truth
    # self.slots_pred_prev = slots_pred
    # self.teacher_action_prev = teacher_action

    # for reference to verify for external calculation 
    # keep track of coreference JGA 
    if teacher_action.get('need_coref', False):
        self.metrics.add(
            'coref_jga', AverageMetric(set(slots_truth) == set(slots_pred))
        )
        self.metrics.add('coref_ct', SumMetric(1))
