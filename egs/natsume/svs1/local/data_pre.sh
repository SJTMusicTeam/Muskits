#!/usr/bin/env bash

db=$1
data_dir=$2
fs=$3
dump=wav_dump

./utils/parse_options.sh || exit 1

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <data_dir> <fs>"
    echo "e.g.: $0 downloads/jsut_ver1.1 data/all 24000"
    exit 1
fi

set -euo pipefail

# check directory existence
[ ! -e "${data_dir}" ] && mkdir -p "${data_dir}"

wav_scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
midi_scp=${data_dir}/midi.scp
text_scp=${data_dir}/text
label_scp=${data_dir}/label

# check file existence
[ -e "${wav_scp}" ] && rm "${wav_scp}"
[ -e "${midi_scp}" ] && rm "${midi_scp}"
[ -e "${text_scp}" ] && rm "${text_scp}"
[ -e "${label_scp}" ] && rm "${label_scp}"

# for single spk id
utt_prefix=natsume
NOWPATH=`pwd`

mkdir -p ${dump}

# make wav.scp
find "${db}" -name "*.wav" | sort | while read -r filename; do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    if [ "${fs}" -eq 48000 ]; then
        # default sampling rate
        echo "${utt_prefix}${id} sox ${filename} -b 16 -t wav - |" >> "${wav_scp}"
    else
        rootp=$(dirname ${filename})
        fname=$(basename ${filename} .wav)
        fname_16bits="${dump}/${fname}_bits16.wav"
        sox ${NOWPATH}/${filename} -c 1 -t wavpcm -b 16 -r ${fs} ${NOWPATH}/${fname_16bits}
        echo "${utt_prefix}${id} ${NOWPATH}/${fname_16bits}" >> "${wav_scp}"
    fi

	    echo "${utt_prefix}${id} ${utt_prefix}" >> "${utt2spk}"
    
done
echo "finished making wav.scp."

# make midi.scp
(find "${db}" -name "*.mid" || find "${db}" -name "*.midi" )| sort | while read -r filename; do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    echo "${utt_prefix}${id} ${NOWPATH}/${filename}" >> "${midi_scp}"
done

echo "finished making midi.scp."

# make text and duration
find "${db}" -name "*.lab" | sort | while read -r filename; do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    echo -n "${utt_prefix}${id}" >> "${text_scp}"
    echo -n "${utt_prefix}${id}" >> "${label_scp}"
    cat ${filename} | while read -r starter end text_tmp; do
        text=$(echo ${text_tmp} | sed -e "s/\r/\n/g")
        echo -n " ${text}" >> "${text_scp}"
        echo -n " ${starter} ${end}  ${text}" >> "${label_scp}"
    done
    echo "" >> "${text_scp}"
    echo "" >> "${label_scp}"
    echo "process"
done
echo "finished making text and label.scp"

sort ${wav_scp} -o ${wav_scp}
sort ${utt2spk} -o ${utt2spk}
sort ${midi_scp} -o ${midi_scp}
sort ${text_scp} -o ${text_scp}
sort ${label_scp} -o ${label_scp}

echo "finished making .scp or _scp files."
