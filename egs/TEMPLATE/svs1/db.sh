# Set the path of your corpus
# "downloads" means the corpus can be downloaded by the recipe automatically

KIRITAN=/home/fangzhex/music/data/
NO7SINGING=/home/fangzhex/music/data/
ONIKU=/home/exx/jiatong/projects/svs/data/ONIKU_KURUMI_UTAGOE_DB
OFUTON=/home/exx/jiatong/projects/svs/data/OFUTON_P_UTAGOE_DB
OPENCPOP=/home/exx/jiatong/projects/svs/data/Opencpop
NATSUME=/home/exx/jiatong/projects/svs/data/
NIT_SONG070=/home/exx/jiatong/projects/svs/data/
KISING=/home/exx/jiatong/projects/svs/data/KiSing
COMBINE=
CSD=downloads
ITAKO=

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
    ONIKU=/data3/qt/ONIKU_KURUMI_UTAGOE_DB/
    OFUTON=/data3/qt/OFUTON_P_UTAGOE_DB/
    NATSUME=/data3/qt
    COMBINE=/data3/qt/Muskits/egs/combine_data/svs1/
fi

if [[ `hostname` == venus_wyn_2232 ]]; then
    KIRITAN=/data3/qt/
    ONIKU=/data3/qt/ONIKU_KURUMI_UTAGOE_DB/
    OFUTON=/data3/qt/OFUTON_P_UTAGOE_DB/
    NATSUME=/data3/qt
    PJS=/data1/wyn/Mus_data/PJS_corpus_ver1.1/
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
    COMBINE=/data5/gs/Muskits/egs/combine_data/svs1/
fi
