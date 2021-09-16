#!/usr/bin/env bash

db=$1
data_dir=$2
fs=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db> <data_dir> <fs>"
    echo "e.g.: $0 downloads/jsut_ver1.1 data/all 24000"
    exit 1
fi

set -euo pipefail

# check directory existence
[ ! -e "${data_dir}" ] && mkdir -p "${data_dir}"

## set filenames
#scp=${data_dir}/wav.scp
#utt2spk=${data_dir}/utt2spk
#spk2utt=${data_dir}/spk2utt
#text=${data_dir}/text
#
## check file existence
#[ -e "${scp}" ] && rm "${scp}"
#[ -e "${utt2spk}" ] && rm "${utt2spk}"
#[ -e "${spk2utt}" ] && rm "${spk2utt}"
#[ -e "${text}" ] && rm "${text}"

wav_scp=${data_dir}/wav.scp
midi_scp=${data_dir}/midi.scp
text_scp=${data_dir}/text_scp
duration_scp=${data_dir}/duration_scp
label_scp=${data_dir}/label_scp

# check file existence
[ -e "${wav_scp}" ] && rm "${wav_scp}"
[ -e "${midi_scp}" ] && rm "${midi_scp}"
[ -e "${text_scp}" ] && rm "${text_scp}"
[ -e "${duration_scp}" ] && rm "${duration_scp}"
[ -e "${label_scp}" ] && rm "${label_scp}"

# make wav.scp
find "${db}" -name "*.wav" | sort | while read -r filename; do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    if [ "${fs}" -eq 48000 ]; then
        # default sampling rate
        echo "${id} ${filename}" >> "${wav_scp}"
    else
        echo "${id} sox ${filename} -t wav -r $fs - |" >> "${wav_scp}" # ?
    fi
done
echo "finished making wav.scp."

# make midi.scp
(find "${db}" -name "*.mid" || find "${db}" -name "*.midi" )| sort | while read -r filename; do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")
    if [ "${fs}" -eq 48000 ]; then
        # default sampling rate
        echo "${id} ${filename}" >> "${midi_scp}"
    else
        echo "${id} sox ${filename} -t wav -r $fs - |" >> "${midi_scp}"
    fi
done


echo "finished making midi.scp."

# make text_scp and duration_scp
find "${db}" -name "*.lab" | sort | while read -r filename; do
    id=$(basename ${filename} | sed -e "s/\.[^\.]*$//g")

    echo -n "${id}" >> "${text_scp}"
    echo -n "${id}" >> "${duration_scp}"
    echo -n "${id}" >> "${label_scp}"
    cat ${filename} | while read -r start end text
    do
      echo -n " ${text}" >> "${text_scp}"
      echo -n " ${start} ${end}" >> "${duration_scp}"
      echo -n " ${start} ${end}  ${text}" >> "${label_scp}"
    done
    echo "" >> "${text_scp}"
    echo "" >> "${duration_scp}"
    echo "" >> "${label_scp}"
done
echo "finished making text_scp, duration_scp."

echo "finished making .scp or _scp files."