import librosa
import numpy as np
import torch
from torch import nn


class MaskedLoss(torch.nn.Module):
    """MaskedLoss."""

    def __init__(self, loss, mask_free=False):
        """init."""
        pass

    def forward(self, output, target, length):
        """forward."""
        pass


class PerceptualEntropy(nn.Module):
    """PerceptualEntropy."""

    def __init__(self, bark_num, spread_function, fs, win_len, psd_dict):
        """init."""
        pass

    def forward(self, log_magnitude, real, imag):
        """forward."""
        pass


def perceptual_Loss(filename):
    pass


if __name__ == "__main__":
    pass
