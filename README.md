# Contrastive Learning with Auxiliary User Detection for Identifying Activities (CLAUDIA)

Anonymous repository for ICMLA 2024 paper review. 

## Source Code CLAUDIA

We are excited to share our latest work progress in improving HAR performance with novel network design. Down below we share the detailed runnable code and configuration template with sampled data (2 batch sliced example).

We are also sharing more details beyond the 8 page-limited paper contents, with supplementary tables, figures, and experimental result analysis details. 

## What are in the repo
- bash script in ``code`` folder for running the code ``run_lr1e-4.sh``
- folder ``code`` containing all python code
- folder ``data`` containing sampled data (not the full data)

We showcase an example for runnning the given code on a sampled data slice. Extrasensory (http://extrasensory.ucsd.edu/#paper.vaizman2017a) is a public dataset and researchers should be able to download and process the full original source dataset (unfortunately, we do not own this dataset). For more details, please refer to their original paper.

## How to run the code
- Make sure the required packages are installed with compatible versions. We are aware that torch_geometrics are sensitive to different versions of pytorch
- Unzip folders under data (even the sampled users with sampled example files are large)
- Modify the run_lr1e-4.sh file hyper-parameter settings
- run the script with ``run_lr1e-4.sh``
- Check printline logs in ``log`` folder and the results in ``output`` folder
