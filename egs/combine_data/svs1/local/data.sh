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

if [ -z "${KIRITAN}" ]; then
    log "Fill the value of 'KIRITAN' of db.sh"
    exit 1
fi

if [ -z "${ONIKU}" ]; then
    log "Fill the value of 'ONIKU' of db.sh"
    exit 1
fi

if [ -z "${OFUTON}" ]; then
    log "Fill the value of 'OFUTON' of db.sh"
    exit 1
fi

if [ -z "${NATSUME}" ]; then
    log "Fill the value of 'NATSUME' of db.sh"
    exit 1
fi

if [ -z "${COMBINE}" ]; then
    log "Fill the value of 'COMBINE' of db.sh"
    exit 1
fi

mkdir -p ${COMBINE}/data

train_set=tr_no_dev
train_dev=dev
recog_set=eval

log "combine data: start "
for dataset in ${train_set} ${train_dev} ${recog_set}; do
    echo "process for subset: ${dataset}"
    opts="${COMBINE}data/${dataset}"
    for dir in $*; do
        # echo "dir: ${dir}"
        if test -d ${dir}; then
            opts+=" ${dir}${dataset}"
            # echo "valid dir: ${dir}"
        fi
    done
    utils/combine_data.sh --extra_files "midi.scp label" ${opts} 
done

log "Successfully finished. [elapsed=${SECONDS}s]"
