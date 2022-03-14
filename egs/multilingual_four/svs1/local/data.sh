#!/usr/bin/env bash

set -e
set -u
set -o pipefail

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

combine_path=""
lang_seq=""
fs=

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0

log "$0 $*"

. utils/parse_options.sh || exit 1;


mkdir -p data

train_set=tr_no_dev
train_dev=dev
test_set=eval

combine_path=$(echo "${combine_path}" | tr '|' "\n")
lang_seq=$(echo "${lang_seq}" | tr '|' "\n")

log "combine data: start "
log "[IMPORTANT] assume merging with dumpped files"

log "assume the dump features are with tr_no_dev, dev, eval (if not, please add a softlink)"
for dataset in ${train_set} ${train_dev} ${test_set}; do
    echo "process for subset: ${dataset}"
    opts="data/${dataset}"
    while IFS= read -r -u3 dir; IFS= read -r -u4 lang; do
    # for dir in ${combine_dir}; do
        echo "dir: ${dir}"
        if [ -d ${dir} ]; then
	    org_workspace=$(realpath ${dir}/../../..)
	    org_name=$(basename ${org_workspace})${lang}
	    scripts/utils/copy_data_dir.sh ${dir}/${dataset} data/"${org_name}_${dataset}"
	    python local/relative_path_convert.py ${org_workspace}/svs1 data/"${org_name}_${dataset}"/midi.scp \
		    data/"${org_name}_${dataset}"/midi.scp.tmp
	    python local/relative_path_convert.py ${org_workspace}/svs1 data/"${org_name}_${dataset}"/wav.scp \
		    data/"${org_name}_${dataset}"/wav.scp.tmp
	    sort -o data/"${org_name}_${dataset}"/midi.scp.tmp data/"${org_name}_${dataset}"/midi.scp.tmp
	    sort -o data/"${org_name}_${dataset}"/wav.scp.tmp data/"${org_name}_${dataset}"/wav.scp.tmp
	    mv data/"${org_name}_${dataset}"/midi.scp.tmp data/"${org_name}_${dataset}"/midi.scp
	    mv data/"${org_name}_${dataset}"/wav.scp.tmp data/"${org_name}_${dataset}"/wav.scp
	    awk '{print $1}' data/"${org_name}_${dataset}"/wav.scp  | sed "s/.*/& $lang/" > data/"${org_name}_${dataset}"/utt2lang
	    opts+=" data/${org_name}_${dataset}"
        fi
    done 3<<<"${combine_path}" 4<<<"${lang_seq}"
    scripts/utils/combine_data.sh ${opts} 
done

log "Successfully finished. [elapsed=${SECONDS}s]"
