#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1
. ./cmd.sh || exit 1

fs=24000
n_fft=2048
n_shift=300
win_length=1200

opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

lang=kr # en or kr
if [ "${lang}" == "en" ]; then
    # To suppress recreation, specify wav format
    local_data_opts="--lang english "
    full_lang="english"
else
    local_data_opts="--lang korean "
    full_lang="korean"
fi


train_set=tr_no_dev_${full_lang}
valid_set=dev_${full_lang}
test_sets="dev_${full_lang} eval_${full_lang}"

train_config=conf/tuning/train_naive_rnn.yaml
inference_config=conf/decode.yaml

# text related processing arguments
g2p=none
cleaner=none

./svs.sh \
    --lang ${lang} \
    --stage 7 \
    --stop_stage 7 \
    --dumpdir dump_${lang} \
    --local_data_opts "${local_data_opts}" \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length ${win_length} \
    --token_type phn \
    --g2p ${g2p} \
    --cleaner ${cleaner} \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --vocoder_file "/home/exx/jiatong/projects/svs/ParallelWaveGAN/egs/csd/voc1/exp/tr_no_dev_kr_csd_hifigan.v1_korean/checkpoint-250000steps.pkl" \
    ${opts} "$@"
