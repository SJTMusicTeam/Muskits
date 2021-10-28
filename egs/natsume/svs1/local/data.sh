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

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${Natsume}" ]; then
    log "Fill the value of 'Natsume' of db.sh"
    exit 1
fi

mkdir -p ${Natsume}

train_set=tr_no_dev
train_dev=dev
recog_set=eval1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data Download"
    # The Natsume data should be downloaded from Webset
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Dataset split "
    python local/dataset_split.py ${Natsume}/Natsume_Singing_DB/wav `pwd`/data 0.1 0.1

fi


for dataset in train dev eval1; do
  echo "process for subset: ${dataset}"
  if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      log "stage 2: Generate data directory"
      # scp files generation
      local/data_pre.sh data/${dataset}_raw data/${dataset} 48000
  fi

  if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      log "stage 3: Prepare segments"
      src_data=data/${dataset}
      local/prep_segments.py --silence pau --silence sil ${src_data} 10000 # in ms
      mv ${src_data}/segments.tmp ${src_data}/segments
      mv ${src_data}/label.tmp ${src_data}/label
      mv ${src_data}/text.tmp ${src_data}/text
      cat ${src_data}/segments | awk '{printf("%s natsume\n", $1);}' > ${src_data}/utt2spk
      sort ${src_data}/utt2spk -o ${src_data}/utt2spk
      utils/utt2spk_to_spk2utt.pl < ${src_data}/utt2spk > ${src_data}/spk2utt
      utils/fix_data_dir.sh --utt_extra_files label ${src_data}
  fi

done

log "Successfully finished. [elapsed=${SECONDS}s]"
