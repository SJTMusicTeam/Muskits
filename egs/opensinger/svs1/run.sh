#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# spectrogram-related arguments
fs=24000
fmin=80
fmax=12000
f0min=60
f0max=1100
n_fft=2048
n_shift=300
win_length=1200

score_feats_extract=syllable_score_feats   # frame_score_feats | syllable_score_feats

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
train_config=conf/tuning/train_xiaoice_noDP.yaml
# train_config=conf/train.yaml
inference_config=conf/decode.yaml
expdir=exp_xiaoice_pitch

# text related processing arguments
g2p=none
cleaner=none


./svs.sh \
    --lang zh \
    --local_data_opts "--stage 1 $(pwd)" \
    --feats_type raw \
    --pitch_extract dio \
    --fs "${fs}" \
    --fmin "${fmin}" \
    --fmax "${fmax}" \
    --f0min "${f0min}" \
    --f0max "${f0max}" \
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
    --expdir "${expdir}"\
    --ngpu 1 \
    ${opts} "$@"
