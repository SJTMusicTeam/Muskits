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

log "$0 $*"

. utils/parse_options.sh || exit 1;


mkdir -p data

train_set=tr_no_dev
train_dev=dev
test_set=eval

log "combine data: start "
log "[IMPORTANT] assume merging with dumpped files"
for dataset in ${train_set} ${train_dev} ${test_set}; do
    echo "process for subset: ${dataset}"
    opts="data/${dataset}"
    for dir in $*; do
        # echo "dir: ${dir}"
        if [ -d ${dir} ]; then
	    org_workspace=$(realpath ${dir}/../../..)
	    org_name=$(basename ${org_workspace})
	    scripts/utils/copy_data_dir.sh ${dir}/${dataset} data/"${org_name}_${dataset}"
	    python local/relative_path_convert.py ${org_workspace}/svs1 data/"${org_name}_${dataset}"/midi.scp \
		    data/"${org_name}_${dataset}"/midi.scp.tmp
	    python local/relative_path_convert.py ${org_workspace}/svs1 data/"${org_name}_${dataset}"/wav.scp \
		    data/"${org_name}_${dataset}"/wav.scp.tmp
	    sort -o data/"${org_name}_${dataset}"/midi.scp.tmp data/"${org_name}_${dataset}"/midi.scp.tmp
	    sort -o data/"${org_name}_${dataset}"/wav.scp.tmp data/"${org_name}_${dataset}"/wav.scp.tmp
	    mv data/"${org_name}_${dataset}"/midi.scp.tmp data/"${org_name}_${dataset}"/midi.scp
	    mv data/"${org_name}_${dataset}"/wav.scp.tmp data/"${org_name}_${dataset}"/wav.scp
	    opts+=" data/${org_name}_${dataset}"
        fi
    done
    scripts/utils/combine_data.sh ${opts} 
done

log "Successfully finished. [elapsed=${SECONDS}s]"
