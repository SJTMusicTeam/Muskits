# Set the path of your corpus
# "downloads" means the corpus can be downloaded by the recipe automatically

HTS=/data3/qt

# For only JHU environment
if [[ "$(hostname -d)" == clsp.jhu.edu ]]; then
    HTS=/export/c06/jiatong/svs/SVS_system/egs/public_dataset/hts/downloads/
fi