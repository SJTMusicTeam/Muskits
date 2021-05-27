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
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    Window_size = 60 * 1e-3
    Window_shift = 30 * 1e-3

    labels = []
    with open(args.wav_scp) as f:
        for line in f:
            recording_id, path = line.split()
            lab_path = path.replace("wav/", "mono_label/").replace(".wav", ".lab")
            # lab_path = path.replace("wav/", "lab/").replace(".wav", ".lab")
            # print(lab_path)
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




        # # labels = hts.load(lab_path)
        #     # assert "sil" in labels[0][-1]
        #     # assert "sil" in labels[-1][-1]
        #     segment_begin = "{:.7f}".format(labels[0][1])
        #     segment_end = "{:.7f}".format(labels[-1][0])
        #
        #
        #     # As we assume that there's only a single utterance per recording,
        #     # utt_id is same as recording_id.
        #     # https://kaldi-asr.org/doc/data_prep.html
        #     utt_id = recording_id
        #     sys.stdout.write(
        #         "{} {} {} {}\n".format(utt_id, recording_id, segment_begin, segment_end)
        #     )
