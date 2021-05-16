#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# basic settings
start_stage=1
stop_stage=100
n_gpus=1
n_workers=16
vocoder_category=wavernn
conf=conf/wavernn.yaml
raw_data_dir=downloads
expdir=exp/2_1_rnn_norm

if [ "${start_stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    # Stage0: download data
    echo =======================
    echo " Stage0: download data "
    echo =======================
    mkdir -p ${raw_data_dir}
    echo "please download kiritan dataset from https://zunko.jp/kiridev/login.php, requires a Facebook account due to licensing issues"
    echo "put kiritan_singing.zip under ${raw_data_dir}"
    unzip -o ${raw_data_dir}/kiritan_singing.zip -d ${raw_data_dir}
fi

if [ "${start_stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    # Stage1: data preprocessing & format into .npy file
    echo ============================
    echo " Stage1: data preprocessing "
    echo ============================
    python local/preprocess.py \
        --path "${raw_data_dir}" \
        --extension ".wav" \
        --num_workers "${n_workers}" \
        --hp_file "${conf}"
fi

if [ "${start_stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    # Stage1: data preprocessing & format into .npy file
    echo ============================
    echo " Stage2: training "
    echo ============================
    python local/preprocess.py \
        --path "${raw_data_dir}" \
        --extension ".wav" \
        --num_workers "${n_workers}" \
        --hp_file "${conf}"
fi