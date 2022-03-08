# CheckDST
Official repo for [CheckDST](https://openreview.net/forum?id=I_YteLtAYsM): <em>Measuring Real-World Generalization of Dialogue State Tracking Performance</em>

- Supported models: 
    - [ParlAI generation models](#generation-models)
    - [TripPy](#classification-models)-based classification models 
        - e.g. [ConvBERT-DG](https://github.com/alexa/dialoglue), [TripPy-COCO](https://arxiv.org/pdf/2010.12850.pdf), etc.

- TBD 
    - HuggingFace checkpoints


## Preparing the data: 

Following these steps will prepare the proper data format for both ParlAI and TripPy models, as well as downloading the original/raw data files provided by all of MultiWOZ2.1, 2.2, and 2.3. 

1. Request for LAUG data: 
    - The augmented [MultiWOZ 2.3](https://github.com/lexmen318/MultiWOZ-coref) for speech disfluencies and paraphrases can be found in the [LAUG repo](https://github.com/thu-coai/LAUG#supported-datasets). 

1. Set the URL you received for downloading the data as the environment variable `LAUG_DOWNLOAD_LINK` in [set_envs.sh](set_envs.sh) 
1. Create a new environment (e.g. conda), install ParlAI, and set environment variables
```bash
source set_envs.sh 
conda create -n parlai python=3.7 
cd ParlAI
pip install -e . 
cd ../data
```

4. Execute the data preparation script `./prepare_multiwoz_dst.sh`
    - You can learn more about the data preparation details [here](data/README.md)

**Note** 
- There are some strange inconsistencies between dialogues in MultiWOZ 2.3 and LAUG: some of the context and slot labels in the LAUG dialogues have typos. Therefore, we use the original MultiWOZ 2.3 dataset. 


## Generation Models 

For generation models, we used [ParlAI](https://parl.ai). `ParlAI/` contains a copy of the ParlAI project with compatible modifications to enable evaluating generation models with CheckDSTs. 

Detailed instructions can be found in the [ParlAI CheckDST Readme](ParlAI/CHECKDST_README.md)
- Instructions on replicating PrefineDST results can also be found here. 


## Classification Models 

`trippy-public-master/` contains a copy of the [official TripPy repo](https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public) with compatible modifications to enable evaluating existing TripPy models with CheckDST. 

`dialoglue/` contains a copy of the [DialoGLUE repo](https://github.com/alexa/dialoglue) for running the ConvBERT-DG variant of the TripPy model. 

Detailed instructions can be found in the [TripPy CheckDST Readme](trippy-public-master/CHECKDST_README.md)
