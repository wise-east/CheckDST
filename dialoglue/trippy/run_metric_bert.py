import glob 
import subprocess 
import shlex 
import os 

TASK="multiwoz21"
STEP="test"

def main(): 
    dirs = glob.glob("results/multiwoz*")
    print(dirs)

    # return 

    for out_dir in dirs: 

        out_dir = os.path.abspath(out_dir)
        # cmd = f"""

        #     python3 metric_bert_dst.py {TASK}
        #         dataset_config/{TASK}.json 
        #         {out_dir}/pred_res.{STEP}*json
        #         2>&1 | tee {out_dir}/eval_pred_{STEP}.log

        # """

        # script usage: DO.example.advanced <seed> <eval_dir> <metric_bert_dst_only>  
        # NOTE: don't use DO.example.advanced in this repo.
        cmd = f"""
            sbatch /data/home/justincho/trippy-public-master/DO.example.advanced -d {out_dir} -m true
        """
        print(cmd)

        subprocess.run(shlex.split(cmd))

    return 

if __name__ == "__main__": 
    main() 