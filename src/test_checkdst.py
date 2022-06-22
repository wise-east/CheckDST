from utils import CheckDST

gold_data_fn = "/data/home/justincho/CheckDST/data/checkdst/NED/data_reformat_official_v2.3_slots_test.json"

checkdst = CheckDST(gold_data_fn= gold_data_fn)

print(checkdst.df)