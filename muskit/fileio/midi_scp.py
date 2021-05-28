import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
import soundfile
from typeguard import check_argument_types
import miditoolkit

from muskit.fileio.read_text import read_2column_text
from muskit.fileio.utils import midi_to_noteseq, noteseq_to_midi

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
            loader_type: str = "representation",
            rate: np.int16 = 16000
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.rep = loader_type
        self.rate = rate
        self.data = read_2column_text(fname) # get key-value dict

    def __getitem__(self, key):
        # return miditoolkit.midi.parser.MidiFile(self.data[key])
        midi_obj = miditoolkit.midi.parser.MidiFile(self.data[key])
        seq = []
        if self.rep == "representation":
            seq = midi_to_noteseq(midi_obj, self.dtype, self.rate)
        return seq

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

        midi_path = self.dir / f"{key}.{self.format}"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        note_seq = value
        midi_obj = noteseq_to_midi(note_seq)
        midi_obj.dump(midi_path, self.rate)

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