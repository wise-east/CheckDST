from pathlib import Path
import json
import numpy as np
import sys
from pprint import pprint
from collections import defaultdict

path = sys.argv[1]
# print(sys.argv)

dirs = Path("./").glob(path)

for dir in dirs:
    print(f"{'='*100:>100}\n\n")
    subdirs = sorted(Path(dir).glob("*"))
    print(Path(dir).parts[-1])
    grouped_results = defaultdict(list)
    for sd in subdirs:
        trainstats = sd / "model.trainstats"
        trainconfig = trainstats.parts[-2][
            trainstats.parts[-2].find("fs") : trainstats.parts[-2].rfind("_sd")
        ]
        # shortened_dir = " --- ".join([dir.parts[-1], trainconfig])
        if not trainstats.is_file():
            trainstats = sd / "modelmodel.trainstats"
        shortened_dir = trainconfig

        if trainstats.is_file():
            with trainstats.open("r") as f:
                results = json.load(f)
            if "final_test_report" in results:
                jga = results['final_test_report'].get("joint goal acc", -1)
                loss = results['final_test_report'].get("loss", -1)

            if jga != -1 and jga < 0.3:
                print(f"strange JGA, check dir: \t{jga:.3f} {str(shortened_dir)}")

            elif jga != -1:
                print(f"JGA: \t{jga:.3f} {str(shortened_dir)}")
                grouped_results[trainconfig].append(jga)
            elif loss != -1:
                print(f"loss: {loss:.3f} {str(shortened_dir)}")
            else:
                print(f"training incomplete: {shortened_dir}")

    print(f"{Path(dir).parts[-1]},median jga,mean jga,count")
    for k, v in grouped_results.items():
        print(f"{k},{np.median(v):.4f},{np.mean(v):.4f},{len(v)}")
    # print(f"{Path(dir).parts[-1]}, full median jga, {np.median(full_jga):.4f}")
    # print(f"{Path(dir).parts[-1]}, few median jga, {np.median(few_jga):.4f}")
# avg_jga = all_jga/count
# print(f"avg jga: {avg_jga}")
