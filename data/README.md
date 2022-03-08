# Data processing overview

`prepare_multiwoz_dst.sh` is for downloading/processing all the multiwoz and LAUG data we use and making sure that the data quality is correct. 
runs the following: 
1. reformat the v23 multiwoz data for parlai training
2. reformat LAUG data for parlai training
3. replace labels and original context in LAUG data with those from v23 multiwoz data 
4. check whether the original context and labels in LAUG invariances (SD, TP, original) are all the same with the official v23 data 
5. form fewshot data for original dataset and all invariance datasets
6. format data for TripPy models 

# Notes

### Verdict on LAUG vs official MultiWOZ2.3
- Use MultiWOZ2.3 version, not LAUG 
- LAUG labels have some noise for some unknown reason 
- TO DO: show comparison code 