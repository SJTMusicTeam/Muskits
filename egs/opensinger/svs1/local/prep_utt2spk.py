#!/usr/bin/env python3
import argparse
import math
import os
import sys

def get_parser():
    parser = argparse.ArgumentParser(
        description="From segments to utt2spk of opensinger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--segments", type=str, help="path of segment file")
    parser.add_argument("--utt2spk", type=str, help="path of utt2spk")
    return parser


if __name__ == "__main__":
    # os.chdir(sys.path[0]+'/..')
    args = get_parser().parse_args()
    

    segments = open(args.segments, "r", encoding="utf-8")
    label = open(args.utt2spk, "w", encoding="utf-8")

    for seg in segments.readlines():
        filename = seg.split(' ')[0]
        label.write("{} {}\n".format(filename, filename.split('_')[0]))

    segments.close()
    label.close()
