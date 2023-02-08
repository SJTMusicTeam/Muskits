import argparse
import os
import librosa
import numpy as np

from muskit.fileio.midi_scp import MIDIScpWriter


def load_midi_note_scp(midi_note_scp):
    # Note(jiatong): midi format is 0-based
    midi_mapping = {}
    with open(midi_note_scp, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n")
        for key in content:
            key = key.split("\t")
            midi_mapping[key[1]] = int(key[0])
    return midi_mapping


def create_midi(notes, tempo, duration, sr):
    # we assume the tempo is static tempo with only one value
    assert len(notes) == len(duration)
    note_info = []
    for i in range(len(notes)):
        note_dur = int(duration[i] * sr + 0.5)
        note_info.extend(note_dur * [notes[i]])
    tempo_info = [tempo] * len(note_info)
    return note_info, tempo_info


def process_utterance(
    midi_scp_writer,
    wavscp,
    text,
    utt2spk,
    label,
    audio_dir,
    wav_dumpdir,
    segment,
    midi_mapping,
    tgt_sr=24000,
):
    uid, _, phns, notes, syb_dur, phn_dur, keep = segment.strip().split("|")
    phns = phns.split(" ")
    notes = notes.split(" ")
    phn_dur = phn_dur.split(" ")
    syb_dur = syb_dur.split(" ")
    
    y, sr = librosa.load(os.path.join(audio_dir, uid) + ".wav")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # estimate a static tempo for midi format
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    tempo = int(tempo)

    # note to midi index
    notes = [midi_mapping[note] if note != "rest" else 0 for note in notes]

    # duration type convert
    phn_dur = [float(dur) for dur in phn_dur]
    syb_dur = [float(dur) for dur in syb_dur]

    midi_seq = create_midi(notes, tempo, phn_dur, tgt_sr)

    midi_scp_writer["opencpop_{}".format(uid)] = midi_seq
    text.write("opencpop_{} {}\n".format(uid, " ".join(phns)))
    utt2spk.write("opencpop_{} {}\n".format(uid, "opencpop"))

    # apply bit convert, there is a known issue in direct convert in format wavscp
    cmd = f"sox {os.path.join(audio_dir, uid)}.wav -c 1 -t wavpcm -b 16 -r {tgt_sr} {os.path.join(wav_dumpdir, uid)}_bits16.wav"
    os.system(cmd)

    wavscp.write("opencpop_{} {}_bits16.wav\n".format(uid, os.path.join(wav_dumpdir, uid)))

    running_dur = 0
    assert len(phn_dur) == len(phns)
    label_entry = []
    for i in range(len(phns)):
        start = running_dur
        end = running_dur + phn_dur[i]
        label_entry.append("{:.3f} {:.3f} {}".format(start, end, phns[i]))
        running_dur += phn_dur[i]

    label.write("opencpop_{} {}\n".format(uid, " ".join(label_entry)))


def process_subset(args, set_name):
    midi_writer = MIDIScpWriter(
        args.midi_dumpdir,
        os.path.join(args.tgt_dir, set_name, "midi.scp"),
        format="midi",
        rate=np.int32(args.sr),
    )
    wavscp = open(
        os.path.join(args.tgt_dir, set_name, "wav.scp"), "w", encoding="utf-8"
    )
    label = open(os.path.join(args.tgt_dir, set_name, "label"), "w", encoding="utf-8")
    text = open(os.path.join(args.tgt_dir, set_name, "text"), "w", encoding="utf-8")
    utt2spk = open(
        os.path.join(args.tgt_dir, set_name, "utt2spk"), "w", encoding="utf-8"
    )

    midi_mapping = load_midi_note_scp(args.midi_note_scp)

    with open(
        os.path.join(args.src_data, "segments", set_name + ".txt"),
        "r",
        encoding="utf-8",
    ) as f:
        segments = f.read().strip().split("\n")
        for segment in segments:
            process_utterance(
                midi_writer,
                wavscp,
                text,
                utt2spk,
                label,
                os.path.join(args.src_data, "segments", "wavs"),
                args.wav_dumpdir,
                segment,
                midi_mapping,
                tgt_sr=args.sr,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Data for Opencpop Database")
    parser.add_argument("src_data", type=str, help="source data directory")
    parser.add_argument("--tgt_dir", type=str, default="data")
    parser.add_argument(
        "--midi_note_scp",
        type=str,
        help="midi note scp for information of note id",
        default="local/midi-note.scp",
    )
    parser.add_argument(
        "--midi_dumpdir", type=str, help="midi obj dump directory", default="midi_dump"
    )
    parser.add_argument(
        "--wav_dumpdir", type=str, help="wav dump directoyr (rebit)", default="wav_dump"
    )
    parser.add_argument("--sr", type=int, help="sampling rate (Hz)")
    args = parser.parse_args()

    for name in ["train", "test"]:
        process_subset(args, name)
