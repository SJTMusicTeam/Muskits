# Set the path of your corpus
# "downloads" means the corpus can be downloaded by the recipe automatically

KIRITAN=
ONIKU=
OFUTON=
NATSUME=

# For only JHU environment
if [[ "$(hostname -d)" == clsp.jhu.edu ]]; then
    KIRITAN=/export/c06/jiatong/svs/SVS_system/egs/public_dataset/kiritan/downloads/
    ONIKU=/export/c06/jiatong/svs/data/ONIKU_KURUMI_UTAGOE_DB
    OFUTON=/export/c06/jiatong/svs/data/OFUTON_P_UTAGOE_DB
    NATSUME=/export/c06/jiatong/svs/SVS_system/egs/public_dataset/natsume/downloads/
fi

# For only venus environment
if [[ `hostname` == venus_qt_2241 ]]; then
    KIRITAN=/data3/qt/
    ONIKU=/data3/qt/OFUTON_P_UTAGOE_DB/
    OFUTON=/data3/qt/ONIKU_KURUMI_UTAGOE_DB/
    NATSUME=/data3/qt/
fi
