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
stage=0
stop_stage=100

log "$0 $*"

. utils/parse_options.sh || exit 1;

if [ -z "${HTS}" ]; then
    log "Fill the value of 'HTS' of db.sh"
    exit 1
fi

mkdir -p ${HTS}

train_set=tr_no_dev
train_dev=dev
recog_set=eval1

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    log "stage -1: Data Download"
    # The HTS data should be downloaded from http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-SONG070-F001.tar.bz2 HTS-demo_NIT-SONG070-F001.tar.bz2
    # with Facebook authentication

fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Data prepare"
    python local/prepare_data.py ${HTS}/HTS-demo_NIT-SONG070-F001/data/raw ${HTS}/HTS-demo_NIT-SONG070-F001/data/labels/mono data/local/raw \
    --label_type r \
    --wav_extention raw \
    --sil pau sil
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Dataset split "
    python local/dataset_split.py data/local/raw/ data/local 0.1 0.1
fi


for dataset in train dev eval1; do
  echo "process for subset: ${dataset}"
  # dataset=test
  if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      log "stage 2: Generate data directory"
      # scp files generation
      local/data_pre.sh data/local/${dataset}_raw data/${dataset} 48000
  fi

  if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      log "stage 3: Prepare segments"
      src_data=data/${dataset}
      local/prep_segments.py --silence pau --silence sil ${src_data} 10000 # in ms
      mv ${src_data}/segments.tmp ${src_data}/segments
      mv ${src_data}/label.tmp ${src_data}/label
      mv ${src_data}/text.tmp ${src_data}/text
      cat ${src_data}/segments | awk '{printf("%s kiritan\n", $1);}' > ${src_data}/utt2spk
      utils/utt2spk_to_spk2utt.pl < ${src_data}/utt2spk > ${src_data}/spk2utt
      utils/fix_data_dir.sh --utt_extra_files label ${src_data}
  fi


done

log "Successfully finished. [elapsed=${SECONDS}s]"
