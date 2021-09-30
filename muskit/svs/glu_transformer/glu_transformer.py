# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer-SVS related modules."""

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from typeguard import check_argument_types

from muskit.svs.bytesing.bytesing import GuidedAttentionLoss
from muskit.svs.bytesing.bytesing import Tacotron2Loss as TransformerLoss
from muskit.torch_utils.nets_utils import make_non_pad_mask
from muskit.torch_utils.nets_utils import make_pad_mask
from muskit.svs.bytesing.decoder import Postnet
from muskit.svs.bytesing.decoder import Prenet as DecoderPrenet
from muskit.svs.bytesing.encoder import Encoder as EncoderPrenet
from muskit.layers.transformer.attention import MultiHeadedAttention
from muskit.layers.transformer.decoder import Decoder
from muskit.layers.transformer.embedding import PositionalEncoding
from muskit.layers.transformer.embedding import ScaledPositionalEncoding
from muskit.layers.transformer.encoder import Encoder
from muskit.layers.transformer.mask import subsequent_mask
from muskit.torch_utils.device_funcs import force_gatherable
from muskit.torch_utils.initialize import initialize
from muskit.svs.abs_svs import AbsSVS
from muskit.svs.gst.style_encoder import StyleEncoder


class PositionalEncoding(nn.Module):
    """Positional Encoding.
    Modified from
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, device="cuda"):
        """init."""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pe = pe.to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Input of forward function.
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Encoder_Postnet(nn.Module):
    """Encoder Postnet."""

    def __init__(self, embed_size, semitone_size=59, Hz2semitone=False):
        """init."""
        super(Encoder_Postnet, self).__init__()

        self.Hz2semitone = Hz2semitone
        if self.Hz2semitone:
            self.emb_pitch = nn.Embedding(semitone_size, embed_size)
        else:
            self.fc_pitch = nn.Linear(1, embed_size)
        # Remember! embed_size must be even!!
        self.fc_pos = nn.Linear(embed_size, embed_size)
        # only 0 and 1 two possibilities
        self.emb_beats = nn.Embedding(2, embed_size)
        self.pos = PositionalEncoding(embed_size)

    def aligner(self, encoder_out, align_phone, text_phone):
        """aligner."""
        # align_phone = [batch_size, align_phone_length]
        # text_phone = [batch_size, text_phone_length]
        # align_phone_length( = frame_num) > text_phone_length
        # batch
        align_phone = align_phone.long()
        for i in range(align_phone.shape[0]):
            before_text_phone = text_phone[i][0]
            encoder_ind = 0
            line = encoder_out[i][0].unsqueeze(0)
            # frame
            for j in range(1, align_phone.shape[1]):
                if align_phone[i][j] == before_text_phone:
                    temp = encoder_out[i][encoder_ind]
                    line = torch.cat((line, temp.unsqueeze(0)), dim=0)
                else:
                    encoder_ind += 1
                    if encoder_ind >= text_phone[i].size()[0]:
                        break
                    before_text_phone = text_phone[i][encoder_ind]
                    temp = encoder_out[i][encoder_ind]
                    line = torch.cat((line, temp.unsqueeze(0)), dim=0)
            if i == 0:
                out = line.unsqueeze(0)
            else:
                out = torch.cat((out, line.unsqueeze(0)), dim=0)

        return out



class GLU_TransformerSVS(nn.Module):
    """Transformer Network."""

    def __init__(
        self,
        phone_size,
        embed_size,
        hidden_size,
        glu_num_layers,
        dropout,
        dec_num_block,
        dec_nhead,
        output_dim,
        n_mels=-1,
        double_mel_loss=True,
        local_gaussian=False,
        semitone_size=59,
        Hz2semitone=False,
        device="cuda",
    ):
        """init."""
        super(GLU_TransformerSVS, self).__init__()
        self.encoder = Encoder(
            phone_size,
            embed_size,
            hidden_size,
            dropout,
            glu_num_layers,
            num_layers=1,
            glu_kernel=3,
        )
        self.enc_postnet = Encoder_Postnet(embed_size, semitone_size, Hz2semitone)

        self.use_mel = n_mels > 0
        if self.use_mel:
            self.double_mel_loss = double_mel_loss
        else:
            self.double_mel_loss = False

        if self.use_mel:
            self.decoder = Decoder(
                dec_num_block,
                embed_size,
                n_mels,
                dec_nhead,
                dropout,
                local_gaussian=local_gaussian,
                device=device,
            )
            if self.double_mel_loss:
                self.double_mel = PostNet(n_mels, n_mels, n_mels)
            self.postnet = PostNet(n_mels, output_dim, (output_dim // 2 * 2))
        else:
            self.decoder = Decoder(
                dec_num_block,
                embed_size,
                output_dim,
                dec_nhead,
                dropout,
                local_gaussian=local_gaussian,
                device=device,
            )
            self.postnet = PostNet(output_dim, output_dim, (output_dim // 2 * 2))

    def forward(
        self,
        characters,
        phone,
        pitch,
        beat,
        pos_text=True,
        pos_char=None,
        pos_spec=None,
    ):
        """forward."""
        encoder_out, text_phone = self.encoder(characters.squeeze(2), pos=pos_char)
        post_out = self.enc_postnet(encoder_out, phone, text_phone, pitch, beat)
        mel_output, att_weight = self.decoder(post_out, pos=pos_spec)

        if self.double_mel_loss:
            mel_output2 = self.double_mel(mel_output)
        else:
            mel_output2 = mel_output
        output = self.postnet(mel_output2)

        return output, att_weight, mel_output, mel_output2

