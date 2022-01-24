#!/usr/bin/env python3
import os
import argparse
import sys
import random
import shutil
from shutil import copyfile


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare segments from HTS-style alignment files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("db_root", type=str, help="source dataset midi folder path")
    parser.add_argument("save_dir", type=str, help="path of dataset split for save")
    parser.add_argument("dev", type=float, help="absolute size of dev dataset")
    parser.add_argument("eval1", type=float, help="absolute size of dev dataset")
    return parser


def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False

    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)

    os.makedirs(data_url)


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    if args.db_root[-1] != "/":
        args.db_root = args.db_root + "/"

    if args.save_dir[-1] != "/":
        args.save_dir = args.save_dir + "/"

    source_root_url = args.db_root

    data_root_url = args.save_dir
    data_train_url = data_root_url + "tr_no_dev_raw"
    data_vaild_url = data_root_url + "dev_raw"
    data_test_url = data_root_url + "eval_raw"
    makedir(data_root_url)
    makedir(data_train_url)
    makedir(data_vaild_url)
    makedir(data_test_url)

    dataset = os.listdir(source_root_url)
    dev = int(len(dataset) * args.dev)
    eval = int(len(dataset) * args.eval1)
    train = len(dataset) - dev - eval
    print(f"trainset={train}, vaild_set={dev}, eval1_set={eval}.")
    if train <= 0:
        print("Error, train set is empty.")
        exit()
    random.shuffle(dataset)
    # copyfile(source_file, destination_file)
    train_set = dataset[:train]
    validation_set = dataset[train : train + dev]
    test_set = dataset[train + dev :]


def transition(dataset, des_url):
    # src_len = len(source_root_url)
    path = [des_url + item for item in ["/wav/", "/mono_label/", "/midi/"]]
    for p in path:
        makedir(p)
    for item in dataset:
        wav_path = source_root_url + item
        lab_path = wav_path.replace("wav/", "mono_label/").replace(".wav", ".lab")
        midi_path = wav_path.replace("wav/", "midi/").replace(".wav", ".mid")

        des_wav = path[0] + item
        des_lab = path[1] + item.replace(".wav", ".lab")
        des_midi = path[2] + item.replace(".wav", ".mid")

        # print(wav_path)
        # print(des_wav)
        # print(lab_path)
        # print(des_lab)
        # print(midi_path)
        # print(des_midi)
        # break
        if wav_path[-3:] != "wav":
            print("Error Path: ", wav_path)
            continue
        copyfile(wav_path, des_wav)
        copyfile(lab_path, des_lab)
        copyfile(midi_path, des_midi)


transition(train_set, data_train_url)
transition(validation_set, data_vaild_url)
transition(test_set, data_test_url)
