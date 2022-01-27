# Set the path of your corpus
# "downloads" means the corpus can be downloaded by the recipe automatically

KIRITAN=/root/data/hku_kg_cuda/NanHUO_HKU/Muskits_3/Muskits/egs/kiritan/download/
ONIKU=
OFUTON=/root/data/hku_kg_cuda/NanHUO_HKU/Muskits_3/Muskits/egs/ofuton_p_utagoe_db/download/OFUTON_P_UTAGOE_DB
Natsume=/root/data/hku_kg_cuda/NanHUO_HKU/Muskits_3/Muskits/egs/natsume/download/

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

# For only uranus environment
if [[ `hostname` == uranus_gs_2223 ]]; then
    KIRITAN=/data1/gs/Muskits/egs/kiritan/svs1/data/
    ONIKU=/data1/gs/dataset/ONIKU_KURUMI_UTAGOE_DB
    OFUTON=/data1/gs/dataset/OFUTON_P_UTAGOE_DB
    NATSUME=/data1/gs/dataset/Natsume_Singing_DB
fi

# For only capri environment
if [[ `hostname` == capri_gs_2345 ]]; then
    KIRITAN=/data5/gs/dataset/
    ONIKU=/data5/gs/dataset/ONIKU_KURUMI_UTAGOE_DB
    OFUTON=/data5/gs/dataset/OFUTON_P_UTAGOE_DB
    NATSUME=/data5/gs/dataset/
fi
