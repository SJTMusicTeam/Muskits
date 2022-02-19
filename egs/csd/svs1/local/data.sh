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
lang=english # english or korean

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${CSD}" ]; then
    log "Fill the value of 'CSD' of db.sh"
    exit 1
fi

mkdir -p ${CSD}

train_set=tr_no_dev
train_dev=dev
recog_set=eval

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    wget -O ${CSD}/csd.zip https://zenodo.org/record/4785016/files/CSD.zip?download=1
    unzip ${CSD}/csd.zip -d ${CSD}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Dataset split "
    # We use a pre-defined split (see details in local/dataset_split.py)"
    python local/dataset_split.py ${CSD}/${lang} \
        data/${train_set} data/${train_dev} data/${recog_set}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 2: Prepare segments"
    for x in ${train_set} ${train_dev} ${recog_set}; do
        src_data=data/${x}
        local/prep_segments.py --silence pau --silence sil ${src_data} 10000 # in ms
        mv ${src_data}/segments.tmp ${src_data}/segments
        mv ${src_data}/label.tmp ${src_data}/label
        mv ${src_data}/text.tmp ${src_data}/text
        cat ${src_data}/segments | awk '{printf("%s csd\n", $1);}' > ${src_data}/utt2spk
        utils/utt2spk_to_spk2utt.pl < ${src_data}/utt2spk > ${src_data}/spk2utt
        utils/fix_data_dir.sh --utt_extra_files label ${src_data}
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
