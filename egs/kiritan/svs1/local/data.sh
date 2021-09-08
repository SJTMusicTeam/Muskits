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
stop_stage=1

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${KIRITAN}" ]; then
    log "Fill the value of 'KIRITAN' of db.sh"
    exit 1
fi

mkdir -p ${KIRITAN}

train_set=tr_no_dev
train_dev=dev
recog_set=eval1

#cd /data2/qt/Muskits/egs/kiritan

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage -1: Data Download"
    # The KIRITAN data should be downloaded from https://zunko.jp/kiridev/login.php
    # with Facebook authentication

fi


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Dataset split "
    python local/dataset_split.py ${KIRITAN}/kiritan_singing/wav $(`pwd`)/data 0.1 0.1
fi


for dataset in train dev eval1; do
  echo "process for subset: ${dataset}"
  # dataset=test
  if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
      log "stage 0: local/data_pre.sh"
      # scp files generation
      local/data_pre.sh data/${dataset}_raw data/${dataset} 48000
  fi

  if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      log "stage 1: local/prep_segments.py"
      local/prep_segments.py --silence pau data/${dataset} 13500
  fi

#  if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#      log "stage 2: local/format_scp.py"
#      local/format_scp.py data/${dataset} data/${dataset}_seg 22050 40
#  fi

#   if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#       log "stage 2: local/format_other_scp.py"
#       local/format_other_scp.py data/${dataset} data/${dataset}_seg
#   fi

#   if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#     log "stage 3: local/format_wav_midi_scp.py"
#     local/format_wav_midi_scp.py data/${dataset} data/${dataset}_seg 22050 40
#   fi

done

log "Successfully finished. [elapsed=${SECONDS}s]"
