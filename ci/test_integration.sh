#!/usr/bin/env bash

python="coverage run --append"

touch .coverage

cwd=$(pwd)
test_data=test

# test svs recipe
cd ./egs/${test_data}/tts1 || exit 1
ln -sf ${cwd}/.coverage .
echo "==== SVS ==="
./run.sh --stage 1 --stop-stage 1 --python "${python}"
feats_types="raw fbank stft"
for t in ${feats_types}; do
    echo "==== feats_type=${t} ==="
    # TODO
done
# Remove generated files in order to reduce the disk usage
rm -rf exp dump data
cd "${cwd}" || exit 1

# Validate configuration files
echo "<blank>" > dummy_token_list
echo "==== Validation configuration files ==="
if python3 -c 'import torch as t; from distutils.version import LooseVersion as L; assert L(t.__version__) >= L("1.6.0")' &> /dev/null;  then
    for f in egs/*/asr1/conf/train_asr*.yaml; do
        python3 -m muskit.bin.svs_train --config "${f}" --iterator_type none --dry_run true --output_dir out --token_list dummy_token_list
    done
fi

# These files must be same each other.
for base in cmd.sh conf/slurm.conf conf/queue.conf conf/pbs.conf; do
    file1=
    for f in egs/*/*/"${base}"; do
        if [ -z "${file1}" ]; then
            file1="${f}"
        fi
        diff "${file1}" "${f}" || { echo "Error: ${file1} and ${f} differ: To solve: for f in egs/*/*/${base}; do cp egs/TEMPLATE/svs1/${base} \${f}; done" ; exit 1; }
    done
done


echo "==== test setup.sh ==="
for d in egs/TEMPLATE/*; do
    if [ -d "${d}" ]; then
        d="${d##*/}"
        egs/TEMPLATE/"$d"/setup.sh egs/test/"${d}"
    fi
done
echo "=== report ==="

coverage report
coverage xml
