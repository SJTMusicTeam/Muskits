import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
import soundfile
from typeguard import check_argument_types
import miditoolkit

from muskit.fileio.read_text import read_2column_text
from muskit.fileio.utils import midi_to_seq, seq_to_midi


class MIDIScpReader(collections.abc.Mapping):
    """Reader class for 'midi.scp'.
    Examples:
        key1 /some/path/a.midi
        key2 /some/path/b.midi
        key3 /some/path/c.midi
        key4 /some/path/d.midi
        ...
        >>> reader = MIDIScpReader('midi.scp')
        >>> pitch_array, tempo_array = reader['key1']
    """

    def __init__(
        self,
        fname,
        dtype=np.int16,
        loader_type: str = "representation",
        rate: np.int16 = np.int16(16000),
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.rep = loader_type
        self.rate = rate
        self.data = read_2column_text(fname)  # get key-value dict

    def __getitem__(self, key):
        # return miditoolkit.midi.parser.MidiFile(self.data[key])
        midi_obj = miditoolkit.midi.parser.MidiFile(self.data[key])

        if self.rep == "representation":
            note_seq, tempo_seq = midi_to_seq(midi_obj, self.dtype, self.rate)
        return note_seq, tempo_seq

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
        rate: np.int16 = np.int16(16000),
    ):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.rate = rate
        self.format = format
        self.dtype = dtype

        self.data = {}

    def __setitem__(self, key: str, value):
        note_seq, tempo_seq = value
        midi_path = self.dir / f"{key}.{self.format}"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi_obj = seq_to_midi(note_seq, tempo_seq, self.rate)
        midi_obj.dump(midi_path)

        self.fscp.write(f"{key} {midi_path}\n")

        # Store the file path
        self.data[key] = str(midi_path)

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


# if __name__ == "__main__":
#     path = '/data3/qt/Muskits/egs/kiritan/svs1/dump/raw/org/train/data/format_midi.18/kiritan11_0000.midi'
#     midi_obj = miditoolkit.midi.parser.MidiFile(path)
#     note_seq, tempo_seq = midi_to_seq(midi_obj, np.int16, np.int16(16000) )
