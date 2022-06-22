# check whether val, test list file in LAUG dataset and those in MultiWOZ2.1 is the same
from loguru import logger

keys = ["test", "val"]

for key in keys:
    # laug
    with open(f"LAUG/{key}ListFile", "r") as f:
        laug = set(f.read().splitlines())
    # multiwoz2.1
    with open(f"multiwoz_dst/MULTIWOZ2.1/{key}ListFile.json", "r") as f:
        v21 = set(f.read().splitlines())

    # logger.info(laug - v21)
    # logger.info(v21 - laug)
    if key == "test":
        assert laug == v21
    elif key == "val":
        assert laug == (v21 - {"PMUL4707"})
