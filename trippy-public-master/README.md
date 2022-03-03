## *** For readers of "Out-of-Task Training for Dialog State Tracking Models" ***

The first version of the MTL code is available now. `DO.example.mtl` will train a model with MTL using an auxiliary task. As of now, pre-tokenized data is loaded for the auxiliary tasks. The next update will also include tokenization of the original data.

The paper is available here:
https://www.aclweb.org/anthology/2020.coling-main.596
https://arxiv.org/abs/2011.09379

## Introduction

TripPy is a new approach to dialogue state tracking (DST) which makes use of various copy mechanisms to fill slots with values. Our model has no need to maintain a list of candidate values. Instead, all values are extracted from the dialog context on-the-fly.
A slot is filled by one of three copy mechanisms:
1. Span prediction may extract values directly from the user input;
2. a value may be copied from a system inform memory that keeps track of the system’s inform operations;
3. a value may be copied over from a different slot that is already contained in the dialog state to resolve coreferences within and across domains.
Our approach combines the advantages of span-based slot filling methods with memory methods to avoid the use of value picklists altogether. We argue that our strategy simplifies the DST task while at the same time achieving state of the art performance on various popular evaluation sets including MultiWOZ 2.1.

## How to run

Two example scripts are provided for how to use TripPy. `DO.example.simple` will train and evaluate a simpler model, whereas `DO.example.advanced` uses the parameters that will result in performance similar to the reported ones. Best performance can be achieved by using the maximum sequence length of 512.

## Datasets

Supported datasets are:
- sim-M (https://github.com/google-research-datasets/simulated-dialogue.git)
- sim-R (https://github.com/google-research-datasets/simulated-dialogue.git)
- WOZ 2.0 (see data/)
- MultiWOZ 2.1 (see data/, https://github.com/budzianowski/multiwoz.git)

With a sequence length of 180, you should expect the following average JGA:
- 56% for MultiWOZ 2.1
- 88% for sim-M
- 90% for sim-R
- 92% for WOZ 2.0

## Requirements

- torch (tested: 1.4.0)
- transformers (tested: 2.9.1)
- tensorboardX (tested: 2.0)

## Citation

This work is published as [TripPy: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://www.aclweb.org/anthology/2020.sigdial-1.4/)

If you use TripPy in your own work, please cite our work as follows:

```
@inproceedings{heck2020trippy,
    title = "{T}rip{P}y: A Triple Copy Strategy for Value Independent Neural Dialog State Tracking",
    author = "Heck, Michael and van Niekerk, Carel and Lubis, Nurul and Geishauser, Christian and
              Lin, Hsien-Chin and Moresi, Marco and Ga{\v{s}}i{\'c}, Milica",
    booktitle = "Proceedings of the 21st Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2020",
    address = "1st virtual meeting",
    publisher = "Association for Computational Linguistics",
    pages = "35--44",
}
```


