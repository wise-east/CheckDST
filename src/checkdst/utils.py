import json 
import re 
from typing import List, Dict, Tuple
from loguru import logger 
from pathlib import Path, PosixPath

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


def find_step(fn:PosixPath)-> int:
    """Find the training step value in the filename

    Args:
        fn (PosixPath): filename of interest

    Returns:
        int: found step value. 'inf' if not found. 
    """
    if isinstance(fn, str): 
        fn = Path(fn)
    step = re.sub("[^0-9]", "", fn.name)
    
    step = int(step) if step else float('inf')
    return step


def sort_by_steps(fns: List[PosixPath])-> List[PosixPath]:
    """Return a list of filenames in ascending order of there steps

    Args:
        fns (List[PosixPath]): list of filenames

    Returns:
        List[PosixPath]: sorted list of filenames 
    """
    return sorted(fns, key=lambda x: find_step(x))


def load_jsonl(fn:str)-> List[Dict]:
    """Load jsonl file 
    """
    with open(fn, "r") as f: 
        data = [json.loads(l) for l in f.readlines()] 
    return data 


def normalize_dial_ids(dial_id:str)-> str: 
    """Normalize dial id to take the form such as 'mul003-6' without any suffixes (.json), all lower case

    Args:
        dial_id (str): dial id 

    Returns:
        str: normalized dial id 
    """
    return dial_id.replace(".json", "").lower() 

def extract_slot_from_string(slots_string:str)-> Tuple[List[str]]:
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

