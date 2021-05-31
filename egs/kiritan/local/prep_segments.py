#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import shutil
import resampy
import numpy as np

os.chdir(sys.path[0]+'/../../..')
print(os.getcwd())

from muskit.fileio.sound_scp import SoundScpWriter, SoundScpReader
from muskit.fileio.midi_scp import MIDIScpWriter, MIDIScpReader
from muskit.fileio.read_text import read_2column_text, load_num_sequence_text, load_label_sequence

def pack_zero(file_id, number, length=4):
    number = str(number)
    return file_id + "_" + "0" * (length - len(number)) + number

def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare segments from HTS-style alignment files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp", type=str, help="wav scp file")
    parser.add_argument("threshold", type=int, help="threshold for silence identification.")
    parser.add_argument("win_size", type=int, help="window size in ms")
    parser.add_argument("win_shift", type=int, help="window shift in ms")
    parser.add_argument("outdir", type=str, help="segmented data folder path")
    parser.add_argument("fs", type=np.int16, default=None, help="If the sampling rate specified, " "Change the sampling rate.")
    return parser

def same_split(alignment, threshold):
    size = 2
    while (alignment[-1][1] - alignment[0][0]) / size > threshold:
        size += 1
    segments = []
    start = 0
    for i in range(size - 1):
        index = start
        while index + 1 < len(alignment) and alignment[index + 1][1] - alignment[start][0] <= threshold:
            index += 1
        segments.append(alignment[start:index+1])
        start = index + 1
    segments.append(alignment[start:])
    return segments, size

def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)

    Path(data_url).mkdir(parents=True, exist_ok=True)


def make_segment(file_id, alignment, threshold=13500 * 1e-3, sil="pau"):
    segment_info = {}
    start_id = 1
    seg_start = []
    seg_end = []
    for i in range(len(alignment)):
        if len(seg_start) == len(seg_end) and sil not in alignment[i][2]:
            seg_start.append(i)
        elif len(seg_start) != len(seg_end) and sil in alignment[i][2]:
            seg_end.append(i)
        else:
            continue
    if len(seg_start) != len(seg_end):
        seg_end.append(len(alignment) - 1)
    if len(seg_start) <= 1:
        start = alignment[seg_start[0]][0]
        end = alignment[seg_end[0]][0]

        st, ed = seg_start[0], seg_end[0]
        if end - start > threshold:
            segments, size = same_split(alignment[st:ed], threshold)
            for i in range(size):
                segment_info[pack_zero(file_id, start_id)] = segments[i]
                start_id += 1
        else:
            segment_info[pack_zero(file_id, start_id)] = alignment[st:ed]

    else:
        for i in range(len(seg_start)):
            start = alignment[seg_start[i]][0]
            end = alignment[seg_end[i]][0]
            st, ed = seg_start[i], seg_end[i]
            if end - start > threshold:
                segments, size = same_split(alignment[st:ed], threshold)
                for i in range(size):
                    segment_info[pack_zero(file_id, start_id)] = segments[i]
                    start_id += 1
                continue

            segment_info[pack_zero(file_id, start_id)] = alignment[st:ed]
            start_id += 1
    return segment_info

def get_subsequence(start_time, end_time, seq, rate):
    start = int(start_time*rate)
    end = int(end_time*rate)
    if start < 0:
        start = 0
    if end > len(seq):
        end = len(seq)
    return seq[start: end]

if __name__ == "__main__":
    # print(sys.path[0]+'/..')
    os.chdir(sys.path[0]+'/..')
    # print(os.getcwd())
    args = get_parser().parse_args(sys.argv[1:])
    args.threshold *= 1e-3
    segments = []

    with open(args.scp+"/wav.scp") as f:
        for line in f:
            if len(line) == 0:
                continue
            recording_id, path = line.replace('\n', '').split(' ')
            lab_path = path.replace("wav/", "mono_label/").replace(".wav", ".lab")
            assert os.path.exists(lab_path)
            with open(lab_path) as lab_f:
                labels = []
                quantized_align = []
                for line in lab_f:
                    label = line.replace('\n', '').split(' ')
                    if len(label) != 3:
                        continue
                    labels.append([float(label[0]), float(label[1]), label[2]])
                segments.append(make_segment(recording_id, labels, args.threshold))

    for file in segments:
        for key, val in file.items():
            segment_begin = "{:.7f}".format(val[0][0])
            segment_end = "{:.7f}".format(val[-1][1])

            sys.stdout.write(
                "{} {} {}\n".format(key, segment_begin, segment_end)
            )

    out_wavscp = Path(args.outdir) / "wav.scp"
    out_midiscp = Path(args.outdir) / "midi.scp"
    out_textscp = Path(args.outdir) / "text_scp"
    out_durationscp = Path(args.outdir) / "duration_scp"
    out_labelscp = Path(args.outdir) / "label_scp"


    out_wavdir = Path(args.outdir) / "wav"
    out_mididir = Path(args.outdir) / "midi"
    out_textdir = Path(args.outdir) / "text"
    out_durationdir = Path(args.outdir) / "duration"
    out_labeldir = Path(args.outdir) / "label"

    makedir(args.outdir)
    makedir(out_wavdir)
    makedir(out_mididir)
    makedir(out_textdir)
    makedir(out_durationdir)
    makedir(out_labeldir)


    wav_reader = SoundScpReader(args.scp+"/wav.scp")
    if args.fs is not None:
        midi_reader = MIDIScpReader(args.scp+"/midi.scp", rate=np.int16(args.fs))
    else:
        midi_reader = MIDIScpReader(args.scp+"/midi.scp")
    # Note: generate segmented file
    wav_writer = SoundScpWriter(
        out_wavdir,
        out_wavscp,
        format="wav",
    )
    if args.fs is None:
        args.fs = midi_reader.rate

    midi_writer = MIDIScpWriter(
        out_mididir,
        out_midiscp,
        format="midi",
        rate=args.fs
    )

    for file in segments:
        # wav, midi
        uttid, seg_id = key.split('_')
        rate, wave = wav_reader[uttid]
        note_seq = midi_reader[uttid]

        if args.fs is not None and args.fs != rate:
            wave = resampy.resample(
                wave.astype(np.float64), rate, args.fs, axis=0
            )
            wave = wave.astype(np.int16)
            rate = args.fs

        for key, val in file.items():

            segment_begin = "{:.7f}".format(val[0][0])
            segment_end = "{:.7f}".format(val[-1][1])

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

            segment_begin = val[0][0]
            segment_end = val[-1][1]

            sub_wav = get_subsequence(segment_begin, segment_end, wave, rate)
            sub_note = get_subsequence(segment_begin, segment_end, note_seq, midi_reader.rate)

            wav_writer[key] = int(rate), sub_wav
            midi_writer[key] = sub_note
