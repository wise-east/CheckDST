Verdict on LAUG vs self processed MultiWOZ2.3
- Use MultiWOZ2.3 version, not LAUG 
- command for comparison: 
  1. Format LAUG data: `python reformat_laug_multiwoz_dst.py -sd orig`
  2. Format MultiWOZ2.3 data: `python reformat_v23_multiwoz_dst.py `
  3. `python compare_versions.py -fn1 LAUG/orig/data_reformat_test.json -fn2 multiwoz_dst/MULTIWOZ2.3/data_reformat_test.json`
  4. Examine samples that differ 
  5. move data into MULTIWOZ2.3 folder: `cp LAUG/orig/data*.json multiwoz_dst/MULTIWOZ2.3/`
- LAUG labels have some noise for some unknown reason 

`prepare_multiwoz_dst.sh` is for downloading/processing all the multiwoz and LAUG data we use and making sure that the data quality is correct. 
runs the following: 
1. reformat the v23 multiwoz data for parlai training
2. reformat LAUG data for parlai training
3. replace labels and original context in LAUG data with those from v23 multiwoz data 
4. check whether the original context and labels in LAUG invariances (SD, TP, original) are all the same with the official v23 data 
5. form fewshot data for original dataset and all invariance datasets

