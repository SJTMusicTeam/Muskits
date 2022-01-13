import miditoolkit
import numpy as np


def get_tick_to_time_mapping(ticks_per_beat, tempo_changes, max_tick=np.int32(1e6)):
    """
    Get mapping from ticks to seconds with tempo information
    """
    tick_to_time = np.zeros(max_tick + 1)
    num_tempi = len(tempo_changes)

    fianl_tick = max_tick
    acc_time = 0

    for idx in range(num_tempi):
        start_tick = tempo_changes[idx].time
        cur_tempo = int(tempo_changes[idx].tempo)

        # compute tick scale
        seconds_per_beat = 60 / cur_tempo
        seconds_per_tick = seconds_per_beat / float(ticks_per_beat)

        # set end tick of interval
        end_tick = tempo_changes[idx + 1].time if (idx + 1) < num_tempi else fianl_tick

        # wrtie interval
        ticks = np.arange(end_tick - start_tick + 1)
        tick_to_time[start_tick : end_tick + 1] = acc_time + seconds_per_tick * ticks
        acc_time = tick_to_time[end_tick]
    return tick_to_time


def midi_to_seq(midi_obj, dtype=np.int16, rate=22050, pitch_aug_factor=0, time_aug_factor=1):
    """method for midi_obj.
    Input:
        miditoolkit_object, sampling rate
    Output:
        note_seq: np.array([pitch1,pitch2....]), which length is equal to note.time*rate
        tempo_seq:np.array([pitch1,pitch2....]), which length is equal to note.time*rate
    """
    tick_to_time = midi_obj.get_tick_to_time_mapping()
    max_time = tick_to_time[-1]
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))

    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: (x.time, x.tempo))
    assert len(tempos) == 1
    tempo_BPM = tempos[0].tempo         # global information, beats per minute
    tempo_BPS = tempo_BPM / 60.0        # global information, beats per second

    note_seq = np.zeros(int(rate * max_time * time_aug_factor), dtype=dtype)
    tempo_seq = np.zeros(int(rate * max_time * time_aug_factor), dtype=dtype)
    for i in range(len(notes)):
        st = int(tick_to_time[notes[i].start] * rate * time_aug_factor)
        ed = int(tick_to_time[notes[i].end] * rate * time_aug_factor)
        note_seq[st:ed] = notes[i].pitch if (pitch_aug_factor == 0 or notes[i].pitch == 0) else (notes[i].pitch + pitch_aug_factor)

        st_time = tick_to_time[notes[i].start] * time_aug_factor
        ed_time = tick_to_time[notes[i].end] * time_aug_factor
        note_duration = ed_time - st_time       # Beats in seconds
        beat_num = note_duration * tempo_BPS  # Beats nums in note
        beat_input = int(beat_num / 0.0125 + 0.5)
        tempo_seq[st:ed] = beat_input

    return note_seq, tempo_seq


def seq_to_midi(
    note_seq,
    tempo_seq,
    rate=24000,
    DEFAULT_RESOLUTION=960,
    DEFAULT_TEMPO=120,
    DEFAULT_VELOCITY=64,
):
    """method for note_seq.
    Input:
        note_seq, tempo_seq, sampling rate
    Output:
        miditoolkit_object with default resolution, tempo and velocity.
    """
    # get downbeat and note (no time)
    temp_notes = note_seq
    temp_tempos = tempo_seq
    ticks_per_beat = DEFAULT_RESOLUTION

    # get specific time for tempos
    tempos = []
    i = 0
    last_i = 0
    # acc_time = 0
    acc_tick = 0
    while i < len(temp_tempos):
        bpm = temp_tempos[i]
        ticks_per_second = DEFAULT_RESOLUTION * bpm / 60
        j = i
        while j + 1 < len(temp_tempos) and temp_tempos[j + 1] == bpm:
            j += 1
        if bpm == 0:
            bpm = DEFAULT_TEMPO
        tempos.append(miditoolkit.midi.containers.TempoChange(bpm, acc_tick))
        acc_tick += int((j - last_i + 1) * ticks_per_second / rate)

        last_i = j
        i = j + 1
    tick_to_time = get_tick_to_time_mapping(ticks_per_beat, tempos)

    # get specific time for notes
    notes = []
    i = 0
    while i < len(temp_notes):
        pitch = temp_notes[i]
        j = i
        while j + 1 < len(temp_notes) and temp_notes[j + 1] == pitch:
            j += 1
        st = i / rate
        ed = j / rate

        start = np.searchsorted(tick_to_time, st, "left")
        end = np.searchsorted(tick_to_time, ed, "left")
        if pitch > 0 and pitch <= 128:
            notes.append(
                miditoolkit.midi.containers.Note(
                    start=start, end=end, pitch=pitch, velocity=DEFAULT_VELOCITY
                )
            )

        i = j + 1

    # write
    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = DEFAULT_RESOLUTION
    # write instrument
    inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
    inst.notes = notes
    midi.instruments.append(inst)
    # write tempo
    midi.tempo_changes = tempos
    return midi


if __name__ == "__main__":
    import os

    # paths = os.listdir(
    #     "/data3/qt/Muskits/egs/kiritan/svs1/dump/raw/org/train/data/format_midi.18/"
    # )
    # # print(paths)
    # for p in paths:
    #     # path = '/data3/qt/Muskits/egs/kiritan/svs1/dump/raw/org/train/data/format_midi.18/kiritan11_0001.midi'
    #     path = (
    #         "/data3/qt/Muskits/egs/kiritan/svs1/dump/raw/org/train/data/format_midi.18/"
    #         + p
    #     )
    #     print(path)
    #     midi_obj = miditoolkit.midi.parser.MidiFile(path)
    #     note_seq, tempo_seq = midi_to_seq(midi_obj, np.int16, np.int16(16000))
    #     midi_obj = seq_to_midi(note_seq, tempo_seq, np.int16(16000))
    #     midi_path = "/data3/qt/songmass/output_res_prev/midiscp.mid"
    #     midi_obj.dump(midi_path) 

    import miditoolkit
    path = "/data5/gs/Muskits/egs/ofuton_p_utagoe_db/svs1/dump/raw/org/dev/data/format_midi.1/oniku_00000000000000momiji_0000.midi"
    midi_obj = miditoolkit.midi.parser.MidiFile(path)

    # note_seq, tempo_seq = midi_to_seq(midi_obj, np.int16, np.int16(24000))

    
