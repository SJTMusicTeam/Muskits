#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Jiatong Shi
# Modified from https://github.com/neosapience/mlp-singer
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""

import logging

from typing import Any
from typing import List
from typing import Tuple

import torch

from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, d_model, seq_len, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = FeedForward(seq_len, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, d_model, seq_len)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, seq_len, d_model)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, seq_len, d_model)
        out = x + residual
        return out


class MixerBlock(nn.Module):
    def __init__(self, d_model, seq_len, expansion_factor, dropout):
        super().__init__()
        self.token_mixer = TokenMixer(d_model, seq_len, expansion_factor, dropout)
        self.channel_mixer = ChannelMixer(d_model, expansion_factor, dropout)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, seq_len, d_model)
        return x


class MLPMixer(nn.Module):
    def __init__(
        self, d_model=256, seq_len=256, expansion_factor=2, dropout=0.5, num_layers=6
    ):
        super().__init__()
        self.model = nn.Sequential(
            *[
                MixerBlock(d_model, seq_len, expansion_factor, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        return self.model(x)