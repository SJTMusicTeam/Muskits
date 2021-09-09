# Set the path of your corpus
# "downloads" means the corpus can be downloaded by the recipe automatically

KIRITAN=

# For only JHU environment
if [[ "$(hostname -d)" == clsp.jhu.edu ]]; then
    KIRITAN=/export/c06/jiatong/svs/SVS_system/egs/public_dataset/kiritan/downloads/
fi