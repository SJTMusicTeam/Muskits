#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1
. ./cmd.sh || exit 1

# spectrogram-related arguments
fs=24000
fmin=80
fmax=7600
n_fft=2048
n_shift=300
win_length=1200

score_feats_extract=frame_score_feats   # frame_score_feats | syllable_score_feats
# expdir=exp/mlp_org
expdir=exp/mlp_scheduler_200
# inference_model=68epoch.pth

opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format wav "
fi

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval"

# training and inference configuration
# train_config=conf/tuning/train_xiaoice.yaml
# train_config=conf/tuning/train_xiaoice_noDP.yaml
train_config=conf/tuning/train_mlp.yaml
# train_config=conf/train.yaml
inference_config=conf/decode.yaml

# text related processing arguments
g2p=none
cleaner=none

./svs.sh \
    --lang jp \
    --stage 0 \
    --stop_stage 6 \
    --local_data_opts "--stage 0" \
    --feats_type raw \
    --pitch_extract None \
    --fs "${fs}" \
    --fmax "${fmax}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --g2p ${g2p} \
    --cleaner ${cleaner} \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --score_feats_extract "${score_feats_extract}" \
    --srctexts "data/${train_set}/text" \
    --svs_exp ${expdir} \
    --ngpu 1 \
    ${opts} "$@"
