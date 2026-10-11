#!/usr/bin/env bash
set -euo pipefail

identity=0.8
dataset=./dataset_genomic_${identity}
output_dir=result_g_${identity}
w2v_path=./word2vec_hg38 #/rna_w2v_1_64_64_40_1.pt

mkdir -p ${output_dir}

step0=$(date +%s)

deeps=CNNBiGRUAttention
encodes=OH+W2V1+NUCPROP
python classify_cnnbigruattention_ovr.py \
  --test_csv test_${identity}.csv \
  --split_dir ${dataset}  \
  --deeps ${deeps} \
  --encodes ${encodes} \
  --w2v_path ${w2v_path} \
  --output_dir ${output_dir} \
  --epochs 20 \
  --batch_size 128 \
  --loss focal_bce \
  --seed 42 \

step1=$(date +%s)
echo "Elapsed: $((step1-step0)) seconds"



