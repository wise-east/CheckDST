import json
from pathlib import Path
from glob import glob

DOMAINS = ["attraction", "hotel", "hospital", "restaurant", "police", "taxi", "train"]

# from DST agents
def _extract_slot_from_string(slots_string):
    """
    Either ground truth or generated result should be in the format:
    "dom slot_type slot_val, dom slot_type slot_val, ..., dom slot_type slot_val,"
    and this function would reformat the string into list:
    ["dom--slot_type--slot_val", ... ]
    """
    slots_list = []

    slot_val_conversion = {
        "centre": "center",
        "3-star": "3",
        "2-star": "2",
        "1-star": "1",
        "0-star": "0",
        "4-star": "4",
        "5-star": "5",
    }

    per_domain_slot_lists = {}
    named_entity_slot_lists = []
    named_entity_slot_interested_lists = []

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
            slot_val = slot_val_conversion.get(slot_val, slot_val)
            if not slot_val == "dontcare":
                slots_list.append(domain + "--" + slot_type + "--" + slot_val)
            if domain in per_domain_slot_lists:
                per_domain_slot_lists[domain].add(slot_type + "--" + slot_val)
            else:
                per_domain_slot_lists[domain] = {slot_type + "--" + slot_val}
        # else:
        #     print("Slot prediction without proper comma splits: ")
        #     print(str_split)

    return (
        slots_list,
        per_domain_slot_lists,
        named_entity_slot_lists,
        named_entity_slot_interested_lists,
    )


def get_jga_values(data):

    jga_original = 0
    jga_perturbed = 0
    jga_new_conditional = 0
    jga_conditional = 0
    new_conditional_ct = 0
    conditional_ct = 0
    prev_jga = -1

    p_ct = 0
    o_ct = 0
    for idx, d in enumerate(data):
        eval_str = d['dialog'][0][0]['eval_labels'][0]
        pred_str = d['dialog'][0][1]['text']

        eval_labels = set(_extract_slot_from_string(eval_str)[0])
        pred_labels = set(_extract_slot_from_string(pred_str)[0])

        jga = float(eval_labels == pred_labels)

        # perturbed examples
        if idx % 2:
            metric = "jga_perturbed"
            jga_perturbed += jga
            p_ct += 1

            if jga == 1 or prev_jga == 1:
                jga_new_conditional += jga and prev_jga
                new_conditional_ct += 1
            if prev_jga == 1:
                jga_conditional += jga
                conditional_ct += 1
        else:
            metric = "jga_original"
            jga_original += jga
            o_ct += 1

        prev_jga = jga

        target_value = d['dialog'][0][1]['metrics'][metric]

        # if "nandos" in eval_labels or "nandos ," in eval_str:
        #     print(d['dialog'])
        #     break

        assert jga == target_value, (eval_labels, pred_labels, d['dialog'])

    assert p_ct == o_ct

    return (
        jga_original / (len(data) / 2),
        jga_perturbed / (len(data) / 2),
        jga_conditional / conditional_ct,
        jga_new_conditional / new_conditional_ct,
    )


fps = []
# fps += glob("/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1*")
fps += glob(
    "/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.3/fs_True_prompts_True_lr5e-05_bs4_uf1*"
)
# fps += glob("/data/home/justincho/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.3/fs_False_prompts_True_lr5e-05_bs4_uf1*")
fps += glob(
    "/data/home/justincho/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.3/fs_True_prompts_True_lr1e-05_bs4_uf2*"
)
fps += glob(
    "/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.3/fs_True_prompts_True_lr5e-05_bs4_uf1*"
)
# fps += glob("/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.3/fs_False_prompts_True_lr1e-04_bs4_uf1*")
# fps += glob("/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.1/fs_False_prompts_True_lr5e-05_bs8_uf1*")
fps += glob(
    "/data/home/justincho/ParlAI/models/bart_scratch_multiwoz2.1/fs_True_prompts_True_lr5e-05_bs4_uf1*"
)
# fps += glob("/data/home/justincho/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.1/fs_False_prompts_True_lr1e-05_bs8_uf2*")
fps += glob(
    "/data/home/justincho/ParlAI/models/bart_all_pft_lr5e-6_eps10_ngpu8_bs8_2021-11-11_multiwoz2.1/fs_True_prompts_True_lr5e-05_bs4_uf2*"
)
# fps += glob("/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.1/fs_False_prompts_True_lr1e-04_bs8_uf1*")
fps += glob(
    "/data/home/justincho/ParlAI/models/bart_muppet_multiwoz2.1/fs_True_prompts_True_lr1e-04_bs4_uf1*"
)


for dir_ in fps:

    report_li = sorted(list(Path(dir_).glob("*_report*.json")))
    world_logs_li = sorted(list(Path(dir_).glob("model.*_world_logs*.jsonl")))

    for idx, wl_fn in enumerate(world_logs_li):
        if "test" in str(wl_fn):
            continue
        print(wl_fn)
        with open(wl_fn, "r") as f:
            data = [json.loads(l) for l in f.read().splitlines()]

        jga_original, jga_perturbed, jga_conditional, jga_new_conditional = get_jga_values(
            data
        )

        try:
            with open(report_li[idx], "r") as f:
                report_data = json.load(f)
        except Exception as e:
            print(e)
            print(report_li[idx])
            continue

        assert report_data['report']['jga_original'] == jga_original, (
            report_data['report']['jga_original'],
            jga_original,
        )
        assert report_data['report']['jga_perturbed'] == jga_perturbed, (
            report_data['report']['jga_perturbed'],
            jga_perturbed,
        )
        assert report_data['report']['jga_conditional'] == jga_conditional, (
            report_data['report']['jga_conditional'],
            jga_conditional,
        )
        if 'jga_new_conditional' in report_data:
            report_data.pop('jga_new_conditional')
        report_data['report']['jga_new_conditional'] = jga_new_conditional

        with open(report_li[idx], "w") as f:
            json.dump(report_data, f, indent=4, sort_keys=True)
