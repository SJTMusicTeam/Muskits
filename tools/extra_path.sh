#!/usr/bin/env bash
if [ -n "${BASH_VERSION:-}" ]; then
    # shellcheck disable=SC2046
    TOOL_DIR="$( cd $( dirname ${BASH_SOURCE[0]} ) >/dev/null 2>&1 && pwd )"
elif [ -n "${ZSH_VERSION:-}" ]; then
    # shellcheck disable=SC2046
    TOOL_DIR="$( cd $( dirname ${(%):-%N} ) >/dev/null 2>&1 && pwd )"
else
    # assume something else
    echo "ERROR: Must be executed by bash or zsh."
fi

if [ -z "${TOOL_DIR}" ]; then
    echo "ERROR: Cannot derive the directory path of SVS_system/tools. This might be a bug."
    return 1
fi


export LD_LIBRARY_PATH="${TOOL_DIR}"/lib:"${TOOL_DIR}"/lib64:"${LD_LIBRARY_PATH:-}"
