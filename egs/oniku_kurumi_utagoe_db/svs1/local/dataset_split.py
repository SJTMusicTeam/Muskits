import argparse
import os

UTT_PREFIX = "oniku"
DEV_LIST = ["chatsumi", "goin_home", "aoimeno_ningyou", "momiji", "tetsudou_shouka"]
TEST_LIST = ["usagito_kame", "sousyunfu", "romance_anonimo", "momotarou", "furusato"]

def train_check(song):
    return (song not in DEV_LIST) and (song not in TEST_LIST)

def dev_check(song):
    return song in DEV_LIST

def test_check(song):
    return song in TEST_LIST

def pack_zero(string, size=20):
    if len(string) < size:
        string = "0" * (size - len(string)) + string
    return string


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)

    os.makedirs(data_url)


def process_text_info(text):
    info = open(text, "r", encoding="utf-8")
    label_info = []
    text_info = []
    for line in info.readlines():
        line = line.strip().split()
        label_info.append(float(line[0])/1e6, float(line[1])/1e6, text[2])
        text_info.append(line[2])
   return " ".join(label_info), " ".join(text_info)


def process_subset(src_data, subset, check_func):
    subfolder = os.listdir(src_data))
    makedir(subset)
    wavscp = open(os.path.join(subset, "wav.scp"), "w", encoding="utf-8")
    utt2spk = open(os.path.join(subset, "utt2spk"), "w", encoding="utf-8")
    midiscp = open(os.path.join(subset, "midi.scp"), "w", encoding="utf-8")
    text_scp = open(os.path.join(subset, "text"), "w", encoding="utf-8")
    label_scp = open(os.path.join(subset, "label"), "w", encoding="utf-8")

    for folder in subfolder:
        if not check_func(folder):
            continue
        utt_id = "{}_{}".format(UTT_PREFIX, pack_zero(folder))
        wavscp.write("{} {}".format(utt_id, os.path.join(src_data, folder, "{}.wav".format(folder))))
        utt2spk.write("{} {}".format(utt_id, UTT_PREFIX))
        text_info, label_info = process_text_info(os.path.join(src_data, folder, "{}.lab".format(folder)))
        text_scp.write("{} {}".format(utt_id, text_info))
        label_scp.write("{} {}".format(utt_id, label_info))
        midi_scp.write("{} {}".format(utt_id, os.path.join(src_data, folder, "{}.mid".format(folder))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Data for Oniku Database"
    )
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("train", type=str, help="train set")
    parser.add_argument("dev", type=str, help="development set")
    parser.add_argument("test", type=str, help="test set")
    args = parser.parse_args()

    process_subset(args.src_data, args.train, train_check)
    process_subset(args.src_data, args.dev, dev_check)
    process_subset(args.src_data, args.test, test_check)
