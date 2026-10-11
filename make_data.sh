#!/usr/bin/env bash

identity=0.8 # cd-hit-est
input_dir=./dataset_genomic_${identity}
infile_nr=rmbase_mAs_genomic_cdhit_priority_soft_nr_${identity}.tsv

mkdir -p ${input_dir}

python split_dataset.py \
    --input_tsv ${input_dir}/${infile_nr} \
    --input_dir ${input_dir} \
    --identity ${identity} \
    --test_size 0.20 \
    --n_splits 5 \
    --nona_ratio 1.0 \
    --max_sample 50000 \
    --seed 42



