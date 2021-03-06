#!/usr/bin/env bash

. tools/activate_python.sh

if [ ! -e tools/kaldi ]; then
    git clone https://github.com/kaldi-asr/kaldi --depth 1 tools/kaldi
fi

# build sphinx document under doc/
mkdir -p doc/_gen

# NOTE allow unbound variable (-u) inside kaldi scripts
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}
set -euo pipefail
# generate tools doc
(
    cd ./utils
    ../doc/argparse2rst.py ./*.py > ../doc/_gen/utils_py.rst
)

./doc/argparse2rst.py ./muskit/bin/*.py > ./doc/_gen/muskit_bin.rst


find ./utils/{*.sh,spm_*} -exec ./doc/usage2rst.sh {} \; | tee ./doc/_gen/utils_sh.rst
find ./muskit/bin/*.py -exec ./doc/usage2rst.sh {} \; | tee ./doc/_gen/muskit_bin.rst

# generate package doc
./doc/module2rst.py --root muskit --dst ./doc --exclude muskit.bin

# build html
travis-sphinx build --source=doc --nowarn

touch doc/build/.nojekyll

