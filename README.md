# CheckDST
Official repo for CheckDST 

Supported models: 
- ParlAI generation models
    - `parlai eval_model --task multiwoz_dst_laug --model-file $MODEL_DIR --aug all` 
- TripPy classification models
    - `DO.example.laug $MODEL_DIR all` 

TBD: 
- evaluate HuggingFace checkpoints with CheckDST

## Data: 
The augmented MultiWOZ2.3 for speech disfluencies and paraphrases can be found in the [LAUG repo](https://github.com/thu-coai/LAUG#supported-datasets). 
After requesting for data from the LAUG repo, you will get a URL for downloading the data. 

```
wget $LAUG_DOWNLOAD_LINK laug.zip
unzip laug.zip 
```
The data will be in `laug/data/multiwoz/SD` for speech disfluencies and `laug/data/multiwoz/TP` for paraphrases. 
The dataset augmented with replaced named entities can be found in [NED_data.zip](NED_data.zip), given in this repo.

## Generation Models 

For generation models, we used [ParlAI](https://parl.ai). `ParlAI` contains a copy of the ParlAI project with compatible modifications to enable evaluate generation models with CheckDSTs. 

#### Instructions: 

1. Create a folder for data inside `ParlAI`: i.e. `mkdir ParlAI/data` 
1. Place the data from the [data section](#data) into `ParlAI/data` 
1. Create an environment: `conda create -n parlai python=3.7` 
1. Follow steps in [ParlAI](ParlAI/README.md) to install `ParlAI` 
1. (optional) Train a ParlAI model: e.g. `parlai train_model --task multiwoz_dst` 
1. Evaluate on CheckDST: `parlai eval_model --task multiwoz_dst_laug --model-file $MODELDIR --aug $AUGTYPE` where `$AUGTYPE` is one of `[SD, TP, NEI]`
1. CheckDST results are available as `model.train_report_$AUGTYPE.json` files in the model directory. 


Refer to the ParlAI [docs](https://www.parl.ai/docs/) for additional customization. 


## Classification (TripPy) Models 

`trippy-public-master` contains a copy of the [official TripPy repo](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public) with compatible modifications to enable evaluating existing TripPy models with CheckDST. 

#### Instructions: 

1. Create a folder for data inside `trippy-public-master`: i.e. `mkdir trippy-public-master/data` 
1. Place the data from the [data section](#data) into a `trippy-public-master/data` 
1. Set up your environment according to the [TripPy README](trippy-public-master/README.md)
    * It is recommended to create a separate environment than that used for ParlAI models due to dependency problems. 
1. (optional) Train a TripPy model using `DO.example.advanced`
1. `DO.example.laug $MODELDIR $AUGTYPE` where `$AUGTYPE` is one of `[SD, TP, NEI]`
1. Process the output of TripPy scripts and print out CheckDST results: `python checkdst.py $RESULTSDIR` where `$RESULTSDIR` is the parent directory of trained models.

TripPy-based models can use `trippy-public-master/DO.example.laug` to evaluate models with CheckDST.  


`dialoglue` contains a copy of the [DialoGLUE repo](https://github.com/alexa/dialoglue) for running the ConvBERT variant of the TripPy model. 

