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

if [ -z "${POPCSDB}" ]; then
    log "Fill the value of 'POPCSDB' of db.sh"
    exit 1
fi

mkdir -p ${POPCSDB}

train_set=tr_no_dev
train_dev=dev
eval_set=eval

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    # The POPCS data should be downloaded from \
    # https://drive.google.com/file/d/1qbfgZX9hG4qG3E39xndd76D6plvWaCQd/view
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparaion "

    mkdir -p data/local_raw
    python local/restruct_db.py ${POPCSDB} ${POPCS}/data/local_raw

    mkdir -p wav_dump
    # we convert the music score to midi format
    python local/data_prep.py ${POPCS}/data/local_raw \
        --wav_dumpdir wav_dump \
        --sr ${fs}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Midi extraction"
    mkdir -p midi_dump
    for x in ${train_set} ${train_dev} ${eval_set}; do
        src_data=data/${x}
        pyscripts/audio/midi_extraction.py ${src_data}/wav.scp ${src_data} \
            --midi_dump midi_dump \
            --name midi --fs ${fs}
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Prepare segments"
    for x in ${train_set} ${train_dev} ${eval_set}; do
        src_data=data/${x}
        python local/prep_segments.py ${src_data}
        mv ${src_data}/segments.tmp ${src_data}/segments
        mv ${src_data}/label.tmp ${src_data}/label
        mv ${src_data}/text.tmp ${src_data}/text
        cat ${src_data}/segments | awk '{printf("%s kising\n", $1);}' > ${src_data}/utt2spk
        utils/utt2spk_to_spk2utt.pl < ${src_data}/utt2spk > ${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files label ${src_data}
    done
fi
