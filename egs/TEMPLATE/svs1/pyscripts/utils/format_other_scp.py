#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import shutil
import numpy as np

def pack_zero(file_id, number, length=4):
    number = str(number)
    return file_id + "_" + "0" * (length - len(number)) + number

def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare segments from HTS-style alignment files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp", type=str, help="scp folder")
    parser.add_argument("outdir", type=str, help="segmented data folder path")
    return parser

def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)
    Path(data_url).mkdir(parents=True, exist_ok=True)

def get_subsequence(start_time, end_time, seq, rate):
    if rate is not None:
        start = int(start_time*rate)
        end = int(end_time*rate)
        if start < 0:
            start = 0
        if end > len(seq):
            end = len(seq)
    else:
        start = np.searchsorted([item[0] for item in seq], start_time, 'left')
        end = np.searchsorted([item[1] for item in seq], end_time, 'left')
        # print(seq[start:end])
    return seq[start: end]


if __name__ == "__main__":
    # print(sys.path[0]+'/..')
    os.chdir(sys.path[0]+'/..')
    print(os.getcwd())
    args = get_parser().parse_args(sys.argv[1:])

    label_reader = {}
    with open(args.scp+"/wav.scp") as f:
        for line in f:
            if len(line) == 0:
                continue
            recording_id, path = line.replace('\n', '').split(' ')
            lab_path = path.replace("wav/", "mono_label/").replace(".wav", ".lab")
            assert os.path.exists(lab_path)

            labels = []
            with open(lab_path) as lab_f:
                for lin in lab_f:
                    label = lin.replace('\n', '').split(' ')
                    if len(label) != 3:
                        continue
                    labels.append([float(label[0]), float(label[1]), label[2]])
            label_reader[recording_id] = labels

    segments = {}
    with open(args.scp+"/segments") as f:
        for line in f:
            if len(line) == 0:
                continue
            key, segment_begin, segment_end = line.replace('\n', '').split(' ')
            utt_id, seg_id = key.split('_')
            seq = get_subsequence(float(segment_begin), float(segment_end), label_reader[utt_id], None)
            if len(seq) == 0:
                continue
            segments[key] = seq
            # print(segments[key])
    out_textscp = Path(args.outdir) / "text_scp"
    out_durationscp = Path(args.outdir) / "duration_scp"
    out_labelscp = Path(args.outdir) / "label_scp"


    out_textdir = Path(args.outdir) / "text"
    out_durationdir = Path(args.outdir) / "duration"
    out_labeldir = Path(args.outdir) / "label"

    makedir(args.outdir)
    makedir(out_textdir)
    makedir(out_durationdir)
    makedir(out_labeldir)

    for key, val in segments.items():
        # text, duration, label
        f_text = open(out_textdir / key, "w", encoding='utf-8')
        f_duration = open(out_durationdir / key, "w", encoding='utf-8')
        f_label = open(out_labeldir / key, "w", encoding='utf-8')

        f_text.write(
            "\n".join([phone for st, ed, phone in val])
        )
        f_duration.write(
            "\n".join([str(st) + " " + str(ed) for st, ed, phone in val])
        )
        f_label.write(
            "\n".join([str(st) + " " + str(ed) + " " + phone for st, ed, phone in val])
        )

        f_text.close()
        f_duration.close()
        f_label.close()

