#!/bin/bash 

echo "**** make sure that ParlAI is installed and that the environment variables is set with 'set_envs.sh' ****"
echo "Otherwise, this script will fail and data will be stored in incorrect locations."

# allow for the environment variables to be set if repository is cloned into home directory 
if [[ -f "~/CheckDST/set_envs.sh" ]] ; then 
    source ~/CheckDST/set_envs.sh 
fi 

# Check if the correct environment variables are set: 
if [[ $LAUG_DOWNLOAD_LINK == "" ]] || [[ $CHECKDST_DIR == "" ]]; then 
    echo "Make sure to run `set_envs.sh` to set the correct environment variables to properly prepare the data."
    exit 1 
fi

### Get data from official 2.3 repo and copy it into DST folder 
if [[ ! -e $DATAPATH/multiwoz_dst/MULTIWOZ2.3 ]]; then
    echo "Download MULTIWOZ2.3 data"
    mkdir -p $DATAPATH/multiwoz_dst/MULTIWOZ2.3/

    git clone git@github.com:lexmen318/MultiWOZ-coref.git
    cd MultiWOZ-coref 
    unzip MultiWOZ2_3.zip
    cd .. 
    cp MultiWOZ-coref/MultiWOZ2_3/* $DATAPATH/multiwoz_dst/MULTIWOZ2.3/
    rm -rf MultiWOZ-coref
fi

### Reformat the data.json file for DST in ParlAI

# if MULTIWOZ2.1 directory doesn't exist, download it and format the data using code in multiwoz_dst's agent.py. 
# This step is needed to get files that list out the train/valid/test split for the data for formatting MULTIWOZ2.3
if [[ ! -e $DATAPATH/multiwoz_dst/MULTIWOZ2.1 ]]; then
    echo "MultiWOZ2.1 data not found. Downloading MultiWOZ2.1 with parlai agent"
    parlai dd -t multiwoz_dst --version 2.1
fi

# this will do the reformating inside the multiwoz_dst agent (refer to tasks/multiwoz_dst/utils/reformat.py)
parlai dd -t multiwoz_dst --version 2.3 

# if MULTIWOZ2.2 directory doesn't exist, download it and format the data using code in multiwoz_dst's agent.py
# if [[ ! -e multiwoz_dst/MULTIWOZ2.2 ]]; then
#     mkdir -p multiwoz_dst/MULTIWOZ2.2/
#     echo "MultiWOZ2.2 data not found. Downloading MultiWOZ2.2 from github repo"
#     git clone git@github.com:budzianowski/multiwoz.git
#     mv multiwoz/data/MultiWOZ_2.2/* multiwoz_dst/MULTIWOZ2.2/
#     rm -rf multiwoz
#     echo "Formatting MultiWOZ2.2"
#     python multiwoz_dst/MULTIWOZ2.2/convert_to_multiwoz_format.py --multiwoz21_data_dir multiwoz_dst/MULTIWOZ2.1/ --output_file multiwoz_dst/MULTIWOZ2.2/data.json
#     parlai dd -t multiwoz_dst --version 2.2  
# fi

# Download LAUG data
if [[ ! -e $DATAPATH/laug_dst ]]; then
    mkdir -p $DATAPATH/laug_dst 
    if [[ ! -e $DATAPATH/laug ]]; then 
        echo "Getting LAUG data from https://github.com/thu-coai/LAUG and place it in data folder first."
        wget $LAUG_DOWNLOAD_LINK -O $DATAPATH/download
        unzip $DATAPATH/download  # creates folder `laug/`
        rm $DATAPATH/download
    fi
    cp -r $DATAPATH/laug/data/multiwoz/{SD,TP}/ laug_dst
    mkdir -p $DATAPATH/laug_dst/orig
    cp $DATAPATH/laug/data/multiwoz/{train,test,val}.json.zip $DATAPATH/laug_dst/orig/
fi

cd $DATAPATH

# OPTIONAL: add coreference labels for other versions of multiwoz (2.1 and 2.2)
echo "Add coref labels from multiwoz 2.3 to multiwoz 2.1 and 2.2"
python add_coref_labels_for_all_multiwoz_versions.py

## reformat laug dataset for multiwoz dst (without the DST labels)
echo "Reformat LAUG data"
python reformat_laug_multiwoz_dst.py

## replace labels and original context in LAUG with those from official v2.3 data
echo "Replace LAUG labels and original context with those from official v2.3 official data"
python replace_laug_context_and_slotinf_with_official.py

## make sure that the replacement is correctly done. 
echo "Check whether replacement was correctly done"
python check_data_across_testsets.py #this should say "All clear for $augmentation"

## Optional: prepare single-domain few shot data for the original dataset 
# for V in 2.1 2.2 2.3; do 
#     echo "Prepare few shot data for version ${V}"
#     python form_multiwoz_dst_few_shot_data.py -p $DATAPATH/multiwoz_dst/MULTIWOZ${V}/
# done

## Optional: prepare few shot data for the augmented dataset (only done for MultiWOZ2.3)
# for INV in SD TP orig; do
#     echo "Prepare few shot data for ${INV}"
#     python form_multiwoz_dst_few_shot_data.py -p $DATAPATH/laug_dst/${INV}
# done

## match naming convention
mv $DATAPATH/laug_dst $DATAPATH/checkdst
mv $DATAPATH/checkdst/orig $DATAPATH/checkdst/NED
mv $DATAPATH/checkdst/SD $DATAPATH/checkdst/SDI 
mv $DATAPATH/checkdst/TP $DATAPATH/checkdst/PI

# get unchanged NED dialogue ids 
parlai dd -t multiwoz_checkdst -aug NED -dt test


# format data for trippy models 
# split MultiWOZ2.3 data.json file into train/val/test splits 
python split_data.py $DATAPATH/multiwoz_dst/MULTIWOZ2.3/data.json $DATAPATH/multiwoz_dst/MULTIWOZ2.1

# python prep_data_for_trippy.py 
