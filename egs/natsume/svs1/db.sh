# Set the path of your corpus
# "downloads" means the corpus can be downloaded by the recipe automatically

Natsume=../download/

# For only JHU environment
if [[ "$(hostname -d)" == clsp.jhu.edu ]]; then
    Natsume=/export/c06/jiatong/svs/SVS_system/egs/public_dataset/natsume/downloads/
fi