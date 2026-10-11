#!/usr/bin/env bash
set -euo pipefail

pred_path=./result_g_0.8
result_dir=./
n_fold=5
labels=m6A,m1A,Am,AtoI,nonA
machines=CNNBiGRUAttention
encodes=OH+W2V1+NUCPROP
outfile=metrics_CNNBiGRUAttention.xlsx

python evaluation.py  --pred_path ${pred_path} --machine_method ${machines} --machine_encode ${encodes} --result_dir ${result_dir} --outfile ${outfile} --n_fold ${n_fold}  --labels ${labels}

