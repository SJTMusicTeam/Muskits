# check extra module installation
if ! python3 -c "from praatio import textgrid" > /dev/null; then
    echo "Error: praatio is not installed." >&2
    echo "Error: please install praatio and its dependencies as follows:" >&2
    echo "Error: 'conda activate /Muskits/tools/anaconda/envs/muskit' (you can get the path by 'conda info --env') and 'pip install praatio' " >&2
    return 1
fi