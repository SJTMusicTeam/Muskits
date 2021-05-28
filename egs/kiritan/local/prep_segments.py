#!/usr/bin/env python3
import argparse
import os
import sys

# from nnmnkwii.io import hts


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare segments from HTS-style alignment files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("wav_scp", type=str, help="wav scp file")
    parser.add_argument("win_size", type=int, help="window size in ms.")
    parser.add_argument("win_shift", type=int, help="window shift in ms")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    Window_size = args.win_size
    Window_shift = args.win_shift

    labels = []
    with open(args.wav_scp) as f:
        for line in f:
            recording_id, path = line.split()
            lab_path = path.replace("wav/", "mono_label/").replace(".wav", ".lab")
            assert os.path.exists(lab_path)
            with open(lab_path) as lab_f:
                for line in lab_f:
                    start, end, phone = line.split(' ')
                    # print(line)
                    labels.append([ float(start), float(end), phone])

        segments = []
        max_time = labels[-1][1]
        seg_id = 0
        while seg_id*Window_shift + Window_size <= max_time:
            i = seg_id * Window_shift
            segment_begin = "{:.7f}".format(i)
            segment_end = "{:.7f}".format(i+Window_size)
            utt_id = recording_id
            sys.stdout.write(
                "{} {} {} {}\n".format(utt_id, seg_id, segment_begin, segment_end)
            )
            seg_id += 1

