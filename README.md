# CheckDST
### Official repo for CheckDST from <em>Know Thy Strengths: Comprehensive Dialogue State Tracking Diagnostics</em>

CheckDST conducts a comprehensive diagnosis for dialogue state tracking (DST) models. 
It is model- and data-agnostic: only the prediction files for the original test set and the augmented test set has to be provided in the specified format. 
CheckDST receives predictions in `jsonl` format, where each line is a valid `json` with the following fields: 
```json
{
    "dial_id": "mul0003-5", 
    "context": "<user> i 'm looking for a place to stay . it needs to be a guesthouse and include free wifi .",  
    "aug_type": "orig",
    "pred": "hotel type guesthouse, hotel internet yes,",   
    "gold": "hotel type guesthouse, hotel internet yes,", 
    "requires_coref": false, 
 }
```

**Notes on the json fields** 
- `dial_id`: unique id that identifies the current user turn. There is no format requirements, but needs to be the same for samples that will be compared
- `context`: (optional) The dialogue context and the final user utterance. For computing CheckDST metrics, this is optional, but useful to include for analysis. 
- `aug_type`: Any descriptor for the used augmentation. This will be used as the suffix for the dataframe columns. There is no format requirements. 
- `pred` & `gold`: the predicted and reference dialogue states, repectively. `pred` and `gold` **must** be in `<domain> <slot key> <slot value>` format and be separated with commas. Trailing commas are allowed.
- `requires_coref`: (optional) Required only if CorefJGA needs to be calculated. Either True or False. 
- Optional fields are required to be present, but can be empty as ""



**Example**

Given these two files: 
- `orig_test_prediction.jsonl`: contains predictions of the original test file
- `NED_test_prediction.jsonl`: contains predictions with named entities replaced with unseen ones

CheckDST can be computed with the following: 

```python
from pprint import pprint
import checkdst

# initialize checkdst object 
checkdst = CheckDST()

# add predictions on the original test set. this computes JGA, CorefJGA if applicable, and hallucination 
aug_type, orig_pred_fn = "orig", "orig_test_prediction.jsonl" 
checkdst.add_preds(orig_pred_fn, aug_type=aug_type)

# add predictions on the augmented test set. this computes JGA, CorefJGA if applicable, and hallucination 
aug_type, aug_pred_fn = "NED", "NED_test_prediction.jsonl" 
checkdst.add_preds(aug_pred_fn, aug_type=aug_type)

# compute consistent JGA metrics by comparing JGAs on the original test set and the augmented test set 
checkdst.compute_cjga(orig="orig", aug=aug_type)

# show results 
pprint(checkdst.results(), indent=4)
```

A more detailed example can be found in `src/checkdst/run_checkdst_diagnosis.py`

- This repo provides prediction formatting scripts for the following packages or code bases: 
    - [ParlAI generation models](#generation-models): `src/checkdst/format_parlai_world_logs.py`
    - [TripPy](#classification-models)-based classification models: `src/checkdst/format_trippy_predictions.py`
        - e.g. [ConvBERT-DG](https://github.com/alexa/dialoglue), [TripPy-COCO](https://arxiv.org/pdf/2010.12850.pdf), etc.



## Minimal Setup 

```bash
conda create -n checkdst python=3.8 # create virtual environment 
conda activate checkdst # activate env
pip install -e . # install checkdst package
pip install -r requirements.txt
cd ParlAI  
python setup.py develop # install the version of ParlAI included in this repo
```

If any message appears such as `error: protobuf 4.21.2 is installed but protobuf<3.20,>=3.9.2 is required by {'tensorboard'}` in the last step, install a supported version, e.g. `pip install protobuf==3.9.2`, and rerun `python setup.py develop`. 

## Pre-trained / Pre-finetuned model weights 

These models can be loaded via ParlAI and be fine-tuned with our ParlAI script. 

The pre-trained weights for BART (which we use for SimpleTOD) can be found readily in ParlAI and TripPy / ConvBERT-DG models use the basic BERT model that is automatically loaded from the training script. 

We will share the model weights of the following models soon: SOLOIST and PrefineDST. 
- SOLOIST: TBD
- PrefineDST: TBD 

## Preparing the data 

To replicate results from the paper, the MultiWOZ2.3 data must be prepared first. There are a handful of scripts that needs to be run in order to format and place the data in order to accommodate both generation models for ParlAI and TripPy models. 

Take the following steps to download the original/raw data files provided by all of MultiWOZ2.1, 2.2, and 2.3 and format the data properly for training/inference.

1. Request for LAUG data: 
    - The augmented [MultiWOZ 2.3](https://github.com/lexmen318/MultiWOZ-coref) for speech disfluencies and paraphrases can be found in the [LAUG repo](https://github.com/thu-coai/LAUG#supported-datasets). The URL for this dataset must be requested first before proceeding.

1. Set the URL you received for downloading the data as the environment variable `LAUG_DOWNLOAD_LINK` in [set_envs.sh](set_envs.sh) 
1. Create a new virtual environment (e.g. conda), install ParlAI (the version stored in this repo), and set environment variables
    * Make sure that `which python` and `which pip` are properly pointing to your virtual environment's  python and pip. 
```bash
source set_envs.sh 
conda create -n checkdst python=3.8
conda activate checkdst
cd ParlAI
pip install -e . 
cd .. 
```

4. Execute the data preparation script `.data/prepare_multiwoz_dst.sh`
    - You can learn more about the data preparation details [here](data/README.md)

**Note** 
- There are some strange inconsistencies between dialogues in MultiWOZ 2.3 and LAUG: some of the context and slot labels in the LAUG dialogues have typos. Therefore, we use the original MultiWOZ 2.3 dataset. 


## Generation Models 

For generation models, we used [ParlAI](https://parl.ai). `ParlAI/` contains a copy of the ParlAI project with compatible modifications to enable evaluating generation models with CheckDSTs. 

Detailed instructions can be found in the [ParlAI CheckDST README](ParlAI/CHECKDST_README.md)
- Instructions on replicating results of generation models, including PrefineDST, can be found here. 


## Classification Models 

`trippy-public-master/` contains a copy of the [official TripPy repo](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public) with compatible modifications to enable evaluating existing TripPy models with CheckDST. 

`dialoglue/` contains a copy of the [DialoGLUE repo](https://github.com/alexa/dialoglue) for running the ConvBERT-DG variant of the TripPy model. 

Detailed instructions can be found in the [TripPy CheckDST README](trippy-public-master/CHECKDST_README.md)
