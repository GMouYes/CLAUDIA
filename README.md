# Contrastive Learning with Auxiliary User Detection for Identifying Activities (CLAUDIA)

Repository for CLAUDIA published in ICMLA 2024. 

## Source Code CLAUDIA

We are excited to share our latest work on improving HAR performance with novel network design. Down below we share the detailed runnable code and configuration template with sampled data (2 batch sliced example).

We are also sharing more details beyond the 6 page-limited paper contents, with supplementary tables, figures, and experimental result analysis details (in ICMLA_CLAUDIA_Supplement_material.pdf). 

## What is in the repo
- bash script in ``code`` folder for running the code ``run_lr1e-4_commonD6144_hidden256_0.06_0.03_0.006.sh``
- folder ``code`` containing all python code
- folder ``data`` containing sampled data (not the full data)

We showcase an example of running the given code on a sampled data slice. Extrasensory (http://extrasensory.ucsd.edu/#paper.vaizman2017a) is a public dataset and researchers should be able to download and process the full original source dataset (unfortunately, we do not own this dataset). For more details, please refer to their original paper.

## How to run the code
- Make sure the required packages are installed with compatible versions. We are aware that torch_geometrics are sensitive to different versions of PyTorch
- Unzip folders under data (even the sampled users with sampled example files are large)
- Modify the ``config_lr1e-4_commonD6144_hidden256_0.06_0.03_0.006.yml`` file hyper-parameter settings
- run the script with ``run_lr1e-4_commonD6144_hidden256_0.06_0.03_0.006.sh``
- Check printline logs in ``log`` folder and the results in ``output`` folder
