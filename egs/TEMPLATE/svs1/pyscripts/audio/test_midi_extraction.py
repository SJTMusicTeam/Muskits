#!/usr/bin/env python3
import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional

import humanfriendly
import numpy as np
import librosa
import os
import resampy
import soundfile
import pyworld as pw
from tqdm import tqdm
from typeguard import check_argument_types

from muskit.utils.cli_utils import get_commandline_args
from muskit.fileio.read_text import read_2column_text
from muskit.fileio.midi_scp import MIDIScpWriter, MIDIScpReader


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


def str2int_tuple(integers: str) -> Optional[Tuple[int, ...]]:
    """

    >>> str2int_tuple('3,4,5')
    (3, 4, 5)

    """
    assert check_argument_types()
    if integers.strip() in ("none", "None", "NONE", "null", "Null", "NULL"):
        return None
    return tuple(map(int, integers.strip().split(",")))


def smooth_note(note, smooth_context, filter_seg_len=5):
    # filter out short segments
    seg = []
    i = 0
    j = -1
    running_note = -1
    while i < len(note) and i != j:
        pitch = note[i]
        j = i
        while j + 1 < len(note) and note[j + 1] == pitch:
            j += 1
        if j - i < filter_seg_len:
            note[i:j] = 0
        i = j
    
    # smooth by taking median
    pad = smooth_context - (len(note) % smooth_context)
    note = np.concatenate((note, np.zeros((pad,)))).astype(int)
    note = np.reshape(note, (len(note) // smooth_context, smooth_context))
    note = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), arr=note, axis=1)
    return np.repeat(note, smooth_context)[:-pad].astype(int)


def main(wavpath):
    # logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    # logging.basicConfig(level=logging.INFO, format=logfmt)
    # logging.info(get_commandline_args())

    # parser = argparse.ArgumentParser(
    #     description='Extract MIDI (not precise) directly from Singing"',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    # )
    # parser.add_argument("scp")
    # parser.add_argument("outdir")
    # parser.add_argument("--midi_dump", default="midi_dump", type=str)
    # parser.add_argument(
    #     "--name",
    #     default="midi",
    #     help="Specify the prefix word of output file name " 'such as "wav.scp"',
    # )
    # parser.add_argument("--fs", default=24000, help="sample rate")
    # parser.add_argument("--frame_period", default=5, help="f0 extraction period (default 5 ms)")
    # parser.add_argument("--downsample_pitch", default=5, help="f0 downsample rate for pitch extraction")
    # parser.add_argument("--smooth_context", default=20, help="smoothting factor of semitone estimation")
    # parser.add_argument("--filter_seg_len", default=10, help="the threshold to filter out note segments")
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument("--ref-channels", default=None, type=str2int_tuple)
    # group.add_argument("--utt2ref-channels", default=None, type=str)
    # args = parser.parse_args()


    utt2ref_channels = None
    fs = 24000
    smooth_context = 20
    filter_seg_len = 10
    frame_period = 5
    outdir = '/data3/qt/Muskits/egs/opensinger/svs1/pyscripts/audio'
    midi_dump = '/data3/qt/Muskits/egs/opensinger/svs1/pyscripts/audio/midi_dump'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(midi_dump).mkdir(parents=True, exist_ok=True)
    out_midiscp = Path(outdir) / "one_midi.scp"
    writer = MIDIScpWriter(midi_dump, out_midiscp, format="midi", rate=np.int32(fs))

    if os.path.exists(wavpath):
        wave, rate = soundfile.read(wavpath, dtype=np.float64)
    else:
        raise ValueError('Wrong path!')
    # if wave.ndim == 2 and utt2ref_channels is not None:
    #     wave = wave[:, utt2ref_channels(uttid)]
    
    # pitch extraction with DIO
    _f0, t = pw.dio(wave, rate, frame_period=frame_period)
    # 64hz - 1100hz

    print(f'_f0 before refinement:{_f0}')

    # pitch refinement
    f0 = pw.stonemask(wave, _f0, t, rate)
    f0[f0 < 40] = 50 # set lower bound to prevent underflow

    print(f'f0 after refinement:{f0}')
    mif0 = 12 * np.log2(62 / 440) + 49 +20 # round to piano keys
    mxf0 = 12 * np.log2(1100 / 440) + 49 +20# round to piano keys
    print(f'min-max f0 note :{mif0} - {mxf0}')
    f0 = 12 * np.log2(f0 / 440) + 49 # round to piano keys
    note = np.round(f0, 0) + 20 # round to midi note
    print(f'+20 note:{note}')
    # print(f'+20 nonzero_idx :{np.where(note!=0)}')
    note[note < 35] = 0
    note[note > 85] = 0
    print(f' 40-84 nonzero_idx:{np.where(note!=0)}')
    print(f' 40-84 =:{note}')
    note = note.astype(int)
    period_time =  int(rate / 1000 * frame_period)
    note = smooth_note(note, smooth_context, filter_seg_len)
    print(f' smooth nonzero_idx:{np.where(note!=0)}')
    note = np.repeat(note, period_time) # expand to sample level

    # # tempo estimation
    onset_env = librosa.onset.onset_strength(y=wave, sr=rate)
    # # beats often need longer hop_length for estimation
    # tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=rate, aggregate=None)
    # tempo = np.round(tempo, 0)
    # tempo = np.repeat(tempo, 512) # magic number (as defined in librosa)
    # tempo = tempo[:len(note)]
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=rate)
    tempo = np.ones_like(note) * tempo

    writer['one_test'] = note, tempo.astype(np.int32)


if __name__ == "__main__":
    wavpath = '/data3/qt/Muskits/egs/opensinger/svs1/wav_dump/ManRaw0018_给自己的歌0021_bits16.wav'
    main(wavpath)
