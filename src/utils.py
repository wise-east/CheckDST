from calendar import c
import json 
import pandas as pd 
import re 
import tqdm 
from typing import List, Dict 
from loguru import logger 
import numpy as np
import math 
from pprint import pprint 

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

def extract_slot_from_string(slots_string:str):
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
        import pdb; pdb.set_trace()
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


class CheckDST: 
    
    def __init__(self, gold_data_fn:str): 
        """Initialize dataframe with original data (reformatted MultiWOZ data)
        TODO: directly load from MultiWOZ 2.3 data to remove dependency with fomratting code 

        Args:
            gold_data_fn (str): filepath to original MultiWOZ data 
        """
        with open(gold_data_fn, "r") as f: 
            self.gold_data = json.load(f)    

        # normalize dial_id names 
        self.gold_data = {normalize_dial_ids(k) :v for k, v in self.gold_data.items()}
        self.df = pd.DataFrame(self.gold_data).T
        self.df.rename(columns={"dial_id": "dial_idx"}, inplace=True)
        self.df.index.name = "dial_id"
        
        # dictionary for holding key results 
        self.checkdst_results = {} 

    def get_results(self)->Dict: 
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

    def add_preds(self, pred_fn:str, aug_type:str, compute:bool = True)->None: 
        """Read CheckDST formatted jsonl files and add them to dataframe 

        Args:
            pred_fn (str): filepath to CheckDST formatted jsonl file 
            aug_type (str): type of augmentation used 
        """
        # load data 
        preds = pd.read_json(pred_fn, orient='records', lines=True)
        
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
        
        
    def _compute_metrics(self, aug_type: str)->Dict: 
        """Key function for computing CheckDST metrics (except for cJGA)
        This function is not for direct invocation. 

        Args:
            aug_type (str): type of augmentation. carried over from add_preds

        Returns:
            Dict: returns the updated checkdst_results dictionary  
        """

        # format predictions for computing metrics 
        pred_col_key = f'pred_{aug_type}'
        gold_col_key = f'gold_{aug_type}'
        self.df[pred_col_key] = self.df[pred_col_key].apply(lambda x: extract_slot_from_string(x) if isinstance(x, str) else np.NaN)
        self.df[gold_col_key] = self.df[gold_col_key].apply(lambda x: extract_slot_from_string(x) if isinstance(x, str) else np.NaN)
        
        
        jgas = [] 
        coref_jgas = []  
        hallucinate = [] 
        pred_named_entity_cts = [] 
        for idx, row in self.df.iterrows(): 
            
            # ignore any rows without a prediction for the given dial_id. this is the case if there were no augmentations in the original dataset 
            if not isinstance(row[pred_col_key], tuple) and (math.isnan(row[pred_col_key]) or math.isnan(row[gold_col_key])): 
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
            if row['need_coref']: 
                coref_jgas.append(jga)
            else: 
                # filler to match number of rows 
                coref_jgas.append(np.NaN)

            # calculate hallucination 
            slots_pred_named_entity = row[pred_col_key][2]
            context = row[f'context_{aug_type}']
            
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
                hallucinate_ct += not(ne in context)
            hallucinate.append(hallucinate_ct)
            pred_named_entity_cts.append(len(slots_pred_named_entity))
                
                
        jga_key = f'jga_{aug_type}'
        coref_jga_key = f'coref_jga_{aug_type}'
        hallucinate_cts_key = f'hallucinate_cts_{aug_type}'
        pred_named_entity_cts_key = f'pred_named_entity_cts_{aug_type}'
        
        # assign new columns 
        self.df[jga_key] = jgas         
        self.df[coref_jga_key] = coref_jgas       
        self.df[hallucinate_cts_key] = hallucinate       
        self.df[pred_named_entity_cts_key] = pred_named_entity_cts       

        # compute metrics 
        total_jga = self.df[jga_key].mean()
        total_coref_jga = self.df[coref_jga_key].mean() 
        total_times_hallucinated =  self.df[hallucinate_cts_key].sum()
        total_named_entities_predicted = self.df[pred_named_entity_cts_key].sum()
        hallucination_frequency = total_times_hallucinated / total_named_entities_predicted

        # store results in results dict         
        self.checkdst_results[jga_key]  = round(total_jga, 4) 
        self.checkdst_results[coref_jga_key]  = round(total_coref_jga, 4) 
        self.checkdst_results[f'hallucination_freq_{aug_type}']  = round(hallucination_frequency, 4) 
        self.checkdst_results[f'hallucination_cts_{aug_type}']  = total_times_hallucinated
        self.checkdst_results[f'pred_named_entity_cts_{aug_type}']  =  total_named_entities_predicted
        self.checkdst_results[f'coref_ct_{aug_type}']  = len(self.df[~self.df[coref_jga_key].isna()]) # track number of cases that required coreference resolution 
                
        # import pdb; pdb.set_trace()

        return  self.checkdst_results


    def compute_cjga(self, orig:str, aug: str) -> Dict: 
        """Compute cJGA (consistent JGA), which is the frequency of getting both original and augmented version correct 

        Args:
            orig (str): column tag for original input JGA 
            aug (str): column tag for augmented JGA

        Returns:
            Dict: _description_
        """
        
        orig_jga_key = f"jga_{orig}"
        aug_jga_key = f"jga_{aug}"
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
            
        cjga_key_name = f'cjga_{orig}_{aug}'
        conditional_jga_key_name = f'conditional_jga_{orig}_{aug}'
        self.df[cjga_key_name] = cjgas 
        self.df[conditional_jga_key_name] = conditional_jgas 
        
        self.checkdst_results[cjga_key_name] = round(self.df[cjga_key_name].mean(), 4)
        self.checkdst_results[conditional_jga_key_name] = round(self.df[conditional_jga_key_name].mean(), 4)

        # import pdb; pdb.set_trace()
        return self.checkdst_results

    def _check_fields(self, item: Dict)->bool: 
        """Check if the fields in each jsonl line has the necessary fields

        Args:
            item (Dict): dictionary from each line of jsonl file 

        Returns:
            bool: True if all conditions are met 
        """
        raise NotImplementedError

    def find_cjga_examples(self):
        """Find cases where jga_aug = 1 but jga_orig = 0 to argue for cjga 
        """
        
        raise NotImplementedError