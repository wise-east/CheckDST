import re
from pathlib import PosixPath
import json
from typing import List


def find_all_checkpoints(dir_: PosixPath) -> List[PosixPath]:
    """Find all model checkpoints in a model directory

    Args:
        fn (PosixPath): model directory that contains checkpoints

    Returns:
        List[PosixPath]: list of checkpoint filepaths within the directory
    """

    checkpoints = [
        fn
        for fn in dir_.glob("*")
        if re.match(".*checkpoint_step[0-9]*$", str(fn))
        or re.match(".*model$", str(fn))
        or re.match(".*checkpoint$", str(fn))
    ]

    return checkpoints


def find_epoch_for_checkpoint(checkpoint_fn: PosixPath) -> float:
    """Load the corresponding trainstats file for model checkpoint and load epoch value

    Args:
        checkpoint_fn (PosixPath): filename of model checkpoint

    Returns:
        float: epoch value
    """
    trainstat_fn = checkpoint_fn.parent / (checkpoint_fn.name + ".trainstats")
    # print(trainstat_fn)

    with trainstat_fn.open("r") as f:
        trainstat = json.load(f)
        epoch = round(trainstat["total_epochs"], 2)

    return epoch
