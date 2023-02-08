import argparse
import os
import librosa
import numpy as np

UTT_PREFIX = "OPENSINGER"
DEV_LIST = ["WomanRaw0047_左边0050.wav", \
            "WomanRaw0047_左边0051.wav", \
            "WomanRaw0047_左边0052.wav", \
            "WomanRaw0046_喜欢你0000.wav", \
            "WomanRaw0046_喜欢你0001.wav", \
            "WomanRaw0046_喜欢你0002.wav", \
            "WomanRaw0046_喜欢你0003.wav", \
            "WomanRaw0046_喜欢你0004.wav", \
            "WomanRaw0046_喜欢你0005.wav", \
            "ManRaw0025_一路向北0024.wav", \
            "ManRaw0025_一路向北0025.wav", \
            "ManRaw0025_一路向北0026.wav", \
            "ManRaw0001_鼓楼0014.wav", \
            "ManRaw0001_鼓楼0015.wav", \
            "ManRaw0001_鼓楼0016.wav", \
                ]
TEST_LIST = ["WomanRaw0047_左边0053.wav", \
            "WomanRaw0047_左边0054.wav", \
            "WomanRaw0047_左边0055.wav", \
            "WomanRaw0046_喜欢你0006.wav", \
            "WomanRaw0046_喜欢你0007.wav", \
            "WomanRaw0046_喜欢你0008.wav", \
            "WomanRaw0046_喜欢你0009.wav", \
            "WomanRaw0046_喜欢你0010.wav", \
            "ManRaw0025_一路向北0027.wav", \
            "ManRaw0025_一路向北0028.wav", \
            "ManRaw0025_一路向北0029.wav", \
            "ManRaw0025_一路向北0030.wav", \
            "ManRaw0001_鼓楼0017.wav", \
            "ManRaw0001_鼓楼0018.wav", \
            "ManRaw0001_鼓楼0019.wav", \
            ]

def pack_zero(string, size=20):
    if len(string) < size:
        string = "0" * (size - len(string)) + string
    return string


def train_check(song):
    return (song not in DEV_LIST) and (song not in TEST_LIST)


def dev_check(song):
    return song in DEV_LIST


def test_check(song):
    return song in TEST_LIST


def process_text_info(text):
    info = open(text, "r", encoding="utf-8")
    label_info = []
    text_info = []
    for line in info.readlines():
        line = line.strip().split()
        label_info.append(
            "{} {} {}".format(
                float(line[0]) , float(line[1]) , line[2].strip()
            )
        )
        text_info.append(line[2].strip())
    return " ".join(label_info), " ".join(text_info)


def process_subset(args, set_name, check_func):
    if not os.path.exists(os.path.join(args.tgt_dir, set_name)):
        os.makedirs(os.path.join(args.tgt_dir, set_name))

    wavscp = open(
        os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8"
    )
    label = open(os.path.join(args.tgt_dir, set_name, "label"), "w", encoding="utf-8")
    text = open(os.path.join(args.tgt_dir, set_name, "text"), "w", encoding="utf-8")
    utt2spk = open(
        os.path.join(args.tgt_dir, set_name, "utt2spk"), "w", encoding="utf-8"
    )

    src_dir = os.path.join(args.src_data)
    

    for song in os.listdir(src_dir):
        if not check_func(song):
            continue
        if song[-1] != 'v':
            continue
        if not os.path.exists(os.path.join(src_dir,song[:-3]+"lab")):
            # print(song[:-3]+"lab")
            continue
        utt = song.split(".")[0]
        # ManRaw0001_鼓楼0016
        utt_id = utt #"{}_{}".format(utt[:-4], pack_zero(utt[-4:]))

        cmd = f"sox {os.path.join(src_dir, song)} -c 1 -t wavpcm -b 16 -r {args.sr} {os.path.join(args.wav_dumpdir, utt_id)}_bits16.wav"
        os.system(cmd)

        wavscp.write("{} {}\n".format(
            utt_id, os.path.join(args.wav_dumpdir, utt_id) + "_bits16.wav"
        ))

        utt2spk.write("{} {}\n".format(utt_id, utt_id.split('-')[0]))
        label_info, text_info = process_text_info(
            os.path.join(src_dir, "{}.lab".format(utt))
        )
        text.write("{} {}\n".format(utt_id, text_info))
        label.write("{} {}\n".format(utt_id, label_info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Opencpop Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directoyr (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    args = parser.parse_args()

    if not os.path.exists(args.wav_dumpdir):
        os.makedirs(args.wav_dumpdir)


    process_subset(args, "tr_no_dev", train_check)
    process_subset(args, "dev", dev_check)
    process_subset(args, "eval", test_check)
