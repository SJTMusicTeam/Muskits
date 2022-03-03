#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
stage=1
stop_stage=100
fs=24000

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${KIRITAN}" ]; then
    log "Fill the value of 'KIRITAN' of db.sh"
    exit 1
fi

mkdir -p ${KIRITAN}

train_set=tr_no_dev
train_dev=dev
recog_set=eval

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    # The KIRITAN data should be downloaded from https://zunko.jp/kiridev/login.php
    # with Facebook authentication

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Dataset split "
    python local/dataset_split.py ${KIRITAN}/kiritan_singing/wav data/local 0.1 0.1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Generate data directory"
    # scp files generation
    for dataset in ${train_set} ${train_dev} ${recog_set}; do
      local/data_pre.sh data/local/${dataset}_raw data/${dataset} ${fs}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Prepare segments"
    for dataset in ${train_set} ${train_dev} ${recog_set}; do
      src_data=$1/data/${dataset}
      local/prep_segments.py --silence pau --silence sil ${src_data} 10000 # in ms
      mv ${src_data}/segments.tmp ${src_data}/segments
      mv ${src_data}/label.tmp ${src_data}/label
      mv ${src_data}/text.tmp ${src_data}/text
      cat ${src_data}/segments | awk '{printf("%s kiritan\n", $1);}' > ${src_data}/utt2spk
      utils/utt2spk_to_spk2utt.pl < ${src_data}/utt2spk > ${src_data}/spk2utt
      utils/fix_data_dir.sh --utt_extra_files label ${src_data}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"