import miditoolkit
import numpy as np

def midi_to_noteseq(midi_obj, dtype=np.int16 , rate=22050):
    """method for midi_obj.
    Input:
        miditoolkit_object, sampling rate
    Output:
        np.array([pitch1,pitch2....]), which length is equal to note.time*rate
    """
    tick_to_time = midi_obj.get_tick_to_time_mapping()
    max_time = tick_to_time[-1]
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))

    note_seq = np.zeros(int(rate * max_time), dtype=dtype)
    idx = 0
    for i in range(len(note_seq)):
        real_time = i / rate
        while idx + 1 < len(notes) and tick_to_time[notes[idx].end] < real_time:
            idx += 1
        if tick_to_time[notes[idx].start] <= real_time:
            note_seq[i] = notes[idx].pitch
    return note_seq


def noteseq_to_midi(note_seq, rate=22050, DEFAULT_RESOLUTION = 480, DEFAULT_TEMPO = 60, DEFAULT_VELOCITY = 60):
    """method for note_seq.
    Input:
        note_seq, sampling rate
    Output:
        miditoolkit_object with default resolution, tempo and velocity.
    """
    # get downbeat and note (no time)
    temp_notes = note_seq

    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4

    seconds_per_beat = 60 / DEFAULT_TEMPO
    seconds_per_tick = seconds_per_beat / float(ticks_per_beat)
    ticks_per_second = float(ticks_per_beat) / seconds_per_beat

    # get specific time for notes
    notes = []
    i = 0
    st = 0
    while i < len(temp_notes):
        pitch = temp_notes[i]
        j = i
        while j+1 < len(temp_notes) and temp_notes[j+1] == pitch:
            j += 1
        duration = int( (j-i+1) * ticks_per_second / rate)
        # duration (end time)
        ed = st + duration
        st = ed
        notes.append(miditoolkit.midi.containers.Note(start=st, end=ed, pitch=pitch, velocity=DEFAULT_VELOCITY))
        i = j + 1

    # get specific time for tempos
    tempos = [[0, DEFAULT_TEMPO]]

    # write
    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = DEFAULT_RESOLUTION
    # write instrument
    inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
    inst.notes = notes
    midi.instruments.append(inst)
    # write tempo
    tempo_changes = []
    for st, bpm in tempos:
        tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
    midi.tempo_changes = tempo_changes
    return midi
