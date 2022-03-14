#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# spectrogram-related arguments
fs=24000
fmin=80
fmax=7600
n_fft=2048
n_shift=300
win_length=1200
use_sid=true

combine_path=""
combine_path+="$(realpath ../../oniku_kurumi_utagoe_db/svs1/dump/raw/)"
combine_path+=" $(realpath ../../ofuton_p_utagoe_db/svs1/dump/raw/)"
combine_path+=" $(realpath ../../kiritan/svs1/dump/raw/)"
combine_path+=" $(realpath ../../natsume/svs1/dump/raw/)"


score_feats_extract=syllable_score_feats # frame_score_feats | syllable_score_feats
expdir=exp/3-14-glu

opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=tr_no_dev
valid_set=dev
test_sets=eval

# training and inference configuration
# train_config=conf/train.yaml
train_config=conf/tuning/train_glu_transformer.yaml
# train_config=conf/tuning/train_naive_rnn.yaml
inference_config=conf/decode.yaml

# text related processing arguments
g2p=none
cleaner=none

./svs.sh \
    --lang jp \
    --stage 6 \
    --stop_stage 7 \
    --local_data_opts "${combine_path}" \
    --feats_type raw \
    --use_sid ${use_sid} \
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
    --vocoder_file "/data3/qt/train_nodev_kiritan_ofuton_oniku_natsume/checkpoint-310000steps.pkl" \
    --ngpu 1 \
    ${opts} "$@"
