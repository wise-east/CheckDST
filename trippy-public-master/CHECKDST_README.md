# Getting started (CheckDST)

1. Create a folder for data inside `trippy-public-master`: i.e. `mkdir trippy-public-master/data` 
1. Place the data from the [data section](#data) into a `trippy-public-master/data` 
1. Set up your environment according to the [TripPy README](trippy-public-master/README.md)
    * It is recommended to create a separate environment than that used for ParlAI models due to dependency problems. i.e. `conda create -n trippy python=3.7`
1. (optional) Train a TripPy model using `DO.example.advanced`
1. `DO.example.laug $MODELDIR $AUGTYPE` where `$AUGTYPE` is one of `[SD, TP, NEI]`
1. Process the output of TripPy scripts and print out CheckDST results: `python checkdst.py $RESULTSDIR` where `$RESULTSDIR` is the parent directory of trained models.

TripPy-based models can use `trippy-public-master/DO.example.laug` to evaluate models with CheckDST.  