import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
import soundfile
from typeguard import check_argument_types
import miditoolkit

from muskit.fileio.read_text import read_2column_text

# what kind of data the downsteam task is need?
# the midi file should convert to ?
class MIDIScpReader(collections.abc.Mapping):
    """Reader class for 'midi.scp'.
    Examples:
        key1 /some/path/a.midi
        key2 /some/path/b.midi
        key3 /some/path/c.midi
        key4 /some/path/d.midi
        ...
        >>> reader = MIDIScpReader('midi.scp')
        >>> midi_obj = reader['key1']
    """

    def __init__(
            self,
            fname,
            dtype=np.int16,
            always_2d: bool = False,
            normalize: bool = False,
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d
        self.normalize = normalize
        self.data = read_2column_text(fname) # get key-value dict

    def __getitem__(self, key):
        return miditoolkit.midi.parser.MidiFile(self.data[key])
        # wav = self.data[key]
        # if self.normalize:
        #     # soundfile.read normalizes data to [-1,1] if dtype is not given
        #     array, rate = soundfile.read(wav, always_2d=self.always_2d)
        # else:
        #     array, rate = soundfile.read(
        #         wav, dtype=self.dtype, always_2d=self.always_2d
        #     )
        #
        # return rate, array

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class MIDIScpWriter:
    """Writer class for 'midi.scp'
    Examples:
        key1 /some/path/a.midi
        key2 /some/path/b.midi
        key3 /some/path/c.midi
        key4 /some/path/d.midi
        ...
        >>> writer = MIDIScpWriter('./data/', './data/midi.scp')
        >>> writer['aa'] = midi_obj
        >>> writer['bb'] = midi_obj
    """

    def __init__(
            self,
            outdir: Union[Path, str],
            scpfile: Union[Path, str],
            format="midi",
            dtype=None,
    ):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.format = format
        self.dtype = dtype

        self.data = {}

    def __setitem__(self, key: str, value):
        midi_obj = value
        midi_path = self.dir / f"{key}.{self.format}"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi_obj.dump(midi_path)

        self.fscp.write(f"{key} {midi_path}\n")

        # Store the file path
        self.data[key] = str(midi_path)

        # rate, array = value
        # assert isinstance(rate, int), type(rate)
        # assert isinstance(signal, np.ndarray), type(signal)
        # if signal.ndim not in (1, 2):
        #     raise RuntimeError(f"Input signal must be 1 or 2 dimension: {signal.ndim}")
        # if signal.ndim == 1:
        #     signal = signal[:, None]
        #
        # wav = self.dir / f"{key}.{self.format}"
        # wav.parent.mkdir(parents=True, exist_ok=True)
        # soundfile.write(str(wav), signal, rate)
        #
        # self.fscp.write(f"{key} {wav}\n")
        #
        # # Store the file path
        # self.data[key] = str(wav)

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()