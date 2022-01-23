#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import shutil
import resampy
import numpy as np

from muskit.fileio.sound_scp import SoundScpWriter, SoundScpReader
from muskit.fileio.midi_scp import MIDIScpWriter, MIDIScpReader
from muskit.fileio.read_text import (
    read_2column_text,
    load_num_sequence_text,
    load_label_sequence,
)


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
    parser.add_argument(
        "fs",
        type=np.int32,
        default=None,
        help="If the sampling rate specified, " "Change the sampling rate.",
    )
    parser.add_argument("nj", type=int, default=40, help="number of workers.")
    return parser


def makedir(data_url):
    if os.path.exists(data_url):
        shutil.rmtree(data_url)
    Path(data_url).mkdir(parents=True, exist_ok=True)


def get_subsequence(start_time, end_time, seq, rate):
    if rate is not None:
        start = int(start_time * rate)
        end = int(end_time * rate)
        if start < 0:
            start = 0
        if end > len(seq):
            end = len(seq)
    else:
        start = np.searchsorted([item[0] for item in seq], start_time, "left")
        end = np.searchsorted([item[1] for item in seq], end_time, "left")
    return seq[start:end]


def segmentation(
    key,
    segment_begin,
    segment_end,
    args_fs,
    args_outdir,
    rate,
    wave,
    note_seq,
    tempo_seq,
):

    out_wavscp = Path(args_outdir) / "wav.scp"
    out_midiscp = Path(args_outdir) / "midi.scp"

    out_wavdir = Path(args_outdir) / "wav"
    out_mididir = Path(args_outdir) / "midi"

    if args_fs is not None and args.fs != rate:
        wave = resampy.resample(wave, rate, args_fs, axis=0)
        # wave = wave.astype(np.int16)
        rate = args_fs

    sub_wav = get_subsequence(segment_begin, segment_end, wave, rate)
    sub_note = get_subsequence(segment_begin, segment_end, note_seq, rate)
    sub_tempo = get_subsequence(segment_begin, segment_end, tempo_seq, rate)

    wav_writer = SoundScpWriter(out_wavdir, out_wavscp, format="wav",)
    midi_writer = MIDIScpWriter(out_mididir, out_midiscp, format="midi", rate=args_fs)

    wav_writer[key] = int(rate), sub_wav
    midi_writer[key] = sub_note, sub_tempo


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    label_reader = {}
    with open(args.scp + "/wav.scp") as f:
        for line in f:
            if len(line) == 0:
                continue
            recording_id, path = line.replace("\n", "").split(" ")
            lab_path = path.replace("wav/", "mono_label/").replace(".wav", ".lab")
            assert os.path.exists(lab_path)

            labels = []
            with open(lab_path) as lab_f:
                for lin in lab_f:
                    label = lin.replace("\n", "").split(" ")
                    if len(label) != 3:
                        continue
                    labels.append([float(label[0]), float(label[1]), label[2]])
            label_reader[recording_id] = labels

    segments = {}
    with open(args.scp + "/segments") as f:
        for line in f:
            if len(line) == 0:
                continue
            key, segment_begin, segment_end = line.replace("\n", "").split(" ")
            utt_id, seg_id = key.split("_")
            seq = get_subsequence(
                float(segment_begin), float(segment_end), label_reader[utt_id], None
            )
            if len(seq) == 0:
                continue
            segments[key] = seq
            # print(segments[key])
    out_wavscp = Path(args.outdir) / "wav.scp"
    out_midiscp = Path(args.outdir) / "midi.scp"

    out_wavdir = Path(args.outdir) / "wav"
    out_mididir = Path(args.outdir) / "midi"

    makedir(out_wavdir)
    makedir(out_mididir)

    wav_reader = SoundScpReader(args.scp + "/wav.scp")
    if args.fs is not None:
        midi_reader = MIDIScpReader(args.scp + "/midi.scp", rate=np.int32(args.fs))
    else:
        midi_reader = MIDIScpReader(args.scp + "/midi.scp")

    if args.fs is None:
        args.fs = midi_reader.rate
    # Note: generate segmented file
    wav_writer = SoundScpWriter(
        out_wavdir,
        out_wavscp,
        format="wav",
    )
    midi_writer = MIDIScpWriter(out_mididir, out_midiscp, format="midi", rate=args.fs)

    for key, val in segments.items():
        segment_begin = val[0][0]
        segment_end = val[-1][1]
        uttid, seg_id = key.split("_")
        rate, wave = wav_reader[uttid]
        note_seq, tempo_seq = midi_reader[uttid]
        segmentation(
            key,
            segment_begin,
            segment_end,
            args.fs,
            args.outdir,
            rate,
            wave,
            note_seq,
            tempo_seq,
        )


