# Compositional Attention with Intermediate Step Supervision

This repository contains source code for the paper "Compositional Attention with Intermediate Step Supervision".

## Requirements
Python 3.10.16

## Datasets
The synthetic datasets are generated in file `datasets/compositional_dataset.py`.

The SCAN, PCFG, COGS and CFQ dataset variants are external and can be downloaded from:

-   SCAN: (https://github.com/brendenlake/SCAN)
-   PCFG: (https://github.com/i-machine-think/am-i-compositional)
-   COGS: (https://github.com/najoungkim/COGS)
-   CFQ: (https://github.com/google-research/google-research/tree/master/cfq)

## Running function composition experiments
Evaluate all the cells in the `experiments/compositional_attention.ipynb` notebook. The notebook contains 3 experiments:

- Baseline with standard Self-Attention
- Compositional Attention without Intermediate Step Supervision
- Compositional Attention with Intermediate Step Supervision

Each training loop may take up to approximately 25 minutes to complete, you can modify the basic functions in `datasets/compositional_dataset.py`.

## Other experiments not mentioned in the paper
Origianlly we proposed a model to probabilistically select the best given function for the next intermediate step (reminds me of bandit problems). Due to the limited time, we are not able to finish this idea, hence leaving here as a fun future work.
