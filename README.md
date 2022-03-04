# CheckDST
Official repo for CheckDST 

## Data: 
The augmented MultiWOZ2.3 for speech disfluencies and paraphrases can be found in the [LAUG repo](https://github.com/thu-coai/LAUG#supported-datasets). 
The dataset augmented with replaced named entities can be found in `NED_data.zip`.

## Generation Models 

For generation models, we used [ParlAI](). `ParlAI` contains a copy of the ParlAI project with compatible modifications to enable evaluate generation models with CheckDSTs. 

#### Instructions: 

1. Create a folder for data inside `ParlAI`: i.e. `mkdir ParlAI/$DATADIR` 
1. Place the data from the [data section](#data) into a `ParlAI/$DATADIR` 
1. (optional) Train a ParlAI model: e.g. `parlai train_model --task multiwoz_dst` 
1. Evaluate on CheckDST: `parlai eval_model --task multiwoz_dst_laug --model-file $MODELDIR --aug $AUGTYPE` where `$AUGTYPE` is one of `[SD, TP, NEI]`
1. CheckDST results are available as `model.train_report_$AUGTYPE.json` files in the model directory. 


Refer to the ParlAI [docs](https://www.parl.ai/docs/) for additional customization. 


## Classification (TripPy) Models 

`trippy-public-master` contains a copy of the [official TripPy repo](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public) with compatible modifications to enable evaluating existing TripPy models with CheckDST. 

#### Instructions: 

1. Create a folder for data inside `ParlAI`: i.e. `mkdir ParlAI/$DATADIR` 
1. Place the data from the [data section](#data) into a `ParlAI/$DATADIR`
1. (optional) Train a TripPy model using `DO.example.advanced`
1. `DO.example.laug $MODELDIR $AUGTYPE` where `$AUGTYPE` is one of `[SD, TP, NEI]`
1. Process and print out CheckDST results: `python checkdst.py $RESULTSDIR` where `$RESULTSDIR` is the parent directory of trained models.

TripPy-based models can use `trippy-public-master/DO.example.laug` to evaluate models with CheckDST.  

