import librosa
import logging
from math import log2
from math import pow
import numpy as np
import os
import random
import torch


def _load_sing_quality(quality_file, standard=3):
    """_load_sing_quality."""
    quality = []
    with open(quality_file, "r") as f:
        data = f.read().split("\n")[1:]
        data = list(map(lambda x: x.split(","), data))
        for sample in data:
            if sample[1] != "" and int(sample[1]) >= standard:
                quality.append("0" * (4 - len(sample[0])) + sample[0])
    return quality

def _phone2char(phones, char_max_len):
    """_phone2char."""
    ini = -1
    chars = []
    phones_index = 0
    for phone in phones:
        if phone != ini:
            chars.append(phone)
            ini = phone
        phones_index += 1
        if len(chars) == char_max_len:
            break
    return chars, phones_index

def _Hz2Semitone(freq):
    """_Hz2Semitone."""
    A4 = 440
    C0 = A4 * pow(2, -4.75)
    name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    if freq == 0:
        return "Sil"  # silence
    else:
        h = round(12 * log2(freq / C0))
        octave = h // 12
        n = h % 12
        return name[n] + "_" + str(octave)

def _full_semitone_list(semitone_min, semitone_max):
    """_full_semitone_list."""
    name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    name_min, octave_min = semitone_min.split("_")
    name_max, octave_max = semitone_max.split("_")

    assert octave_min <= octave_max

    res = ["Sil"]
    flag_insert = 0
    for octave in range(int(octave_min), int(octave_max) + 1):
        for res_name in name:
            if res_name == name_min and octave == int(octave_min):
                flag_insert = 1
            elif res_name == name_max and octave == int(octave_max):
                res.append(res_name + "_" + str(octave))
                flag_insert = 0
                break
            if flag_insert == 1:
                res.append(res_name + "_" + str(octave))

    return res

def _calculate_phone_element_freq(phone_array):
    """Return the phone list and freq of given phone_array."""
    phone_list = [
        phone_array[index]
        for index in range(len(phone_array))
        if index == 0 or phone_array[index] != phone_array[index - 1]
    ]
    phone_freq = []

    begin_index = 0
    for phone in phone_list:
        freq = 0
        for index in range(begin_index, len(phone_array)):
            if phone_array[index] == phone:
                freq += 1
            else:
                phone_freq.append(freq)
                begin_index = index
                break
    phone_freq.append(freq)

    assert len(phone_list) == len(phone_freq)

    return phone_list, phone_freq

def _phone_shift(phone_array, phone_shift_size):
    phone_list, phone_freq = _calculate_phone_element_freq(phone_array)

    shift_side = random.randint(0, 1)  # 0 - left, 1 - right
    # do phone shift augment
    for index in range(len(phone_freq)):

        shift_size = random.randint(1, phone_shift_size)
        flag_shift = 1 if random.random() > 0.5 else 0

        if flag_shift:
            if phone_freq[index] > 2 * shift_size:

                if shift_side == 0 and index != 0:
                    # left shift
                    phone_freq[index] -= shift_size
                    phone_freq[index - 1] += shift_size
                elif shift_side == 1 and index != len(phone_freq) - 1:
                    # right shift
                    phone_freq[index] -= shift_size
                    phone_freq[index + 1] += shift_size

    # reconstruct phone array based on its freq
    res = []
    for index in range(len(phone_freq)):
        for freq in range(phone_freq[index]):
            res.append(phone_list[index])
    res = np.array(res)

    assert len(res) == len(phone_array)

    return res


def _pitch_shift(f0_array, semitone_list):
    f0_list = [f0 for f0 in f0_array if f0 != 0]

    if len(f0_list) == 0:
        # no shift
        return [semitone_list.index(_Hz2Semitone(f0)) for f0 in f0_array]

    f0_min = np.min(f0_list)
    f0_max = np.max(f0_list)

    semitone_min = _Hz2Semitone(f0_min)
    semitone_max = _Hz2Semitone(f0_max)

    index_min = semitone_list.index(semitone_min)
    index_max = semitone_list.index(semitone_max)

    flag_left, flag_right = False, False
    if index_min - 12 >= 1:
        flag_left = True
    if index_max + 12 <= len(semitone_list) - 1:
        flag_right = True

    # decide shift direction
    if flag_left is True and flag_right is True:
        shift_side = random.randint(0, 1)  # 0 - left, 1 - right
    elif flag_left is True:
        shift_side = 0
    elif flag_right is True:
        shift_side = 1
    else:
        shift_side = -1

    # decide whether to shift
    flag_shift = 1 if random.random() > 0.5 else 0

    if shift_side == -1 or flag_shift == 0:
        # no shift
        return [semitone_list.index(_Hz2Semitone(f0)) for f0 in f0_array]
    else:
        if shift_side == 0:
            # left shift
            res = []
            for f0 in f0_array:
                if f0 == 0:
                    res.append(semitone_list.index(_Hz2Semitone(f0)))
                else:
                    res.append(semitone_list.index(_Hz2Semitone(f0)) - 12)
            return res
        elif shift_side == 1:
            # right shift
            res = []
            for f0 in f0_array:
                if f0 == 0:
                    res.append(semitone_list.index(_Hz2Semitone(f0)))
                else:
                    res.append(semitone_list.index(_Hz2Semitone(f0)) + 12)
            return res


class SVSCollator(object):
    """SVSCollator."""

    def __init__(
        self,
    ):
        """init."""

    def __call__(self,):
        """call."""



class SVSDataset(Dataset):
    """SVSDataset."""

    def __init__(
        self,
    ):
        """init."""
 
    def __len__(self):
        """len."""

    def __getitem__(self,):
        """getitem."""
