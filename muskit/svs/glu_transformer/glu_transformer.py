# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer-SVS related modules."""

import logging

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import torch
import math
import torch.nn.functional as F

from typeguard import check_argument_types

from muskit.torch_utils.nets_utils import make_pad_mask
from muskit.layers.rnn.attentions import AttForward
from muskit.layers.rnn.attentions import AttForwardTA
from muskit.layers.rnn.attentions import AttLoc
# from muskit.layers.transformer.attention import MultiHeadedAttention
# from muskit.svs.bytesing.encoder import Encoder as EncoderPrenet
# from muskit.svs.bytesing.decoder import Postnet
from muskit.svs.naive_rnn.naive_rnn import NaiveRNNLoss
from muskit.torch_utils.device_funcs import force_gatherable
from muskit.svs.abs_svs import AbsSVS
from muskit.svs.gst.style_encoder import StyleEncoder

SCALE_WEIGHT = 0.5 ** 0.5

def _shape_transform(x):
    """Tranform the size of the tensors to fit for conv input."""
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)

def _get_activation_fn(activation):
    """_get_activation_fn."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

class PostNet(torch.nn.Module):
    """CBHG Network (mel --> linear)."""

    def __init__(self, input_channel, output_channel, hidden_state):
        """init."""
        super(PostNet, self).__init__()
        self.pre_projection = torch.nn.Conv1d(
            input_channel,
            hidden_state,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )
        self.cbhg = CBHG(hidden_state, projection_size=hidden_state)
        self.post_projection = torch.nn.Conv1d(
            hidden_state,
            output_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )

    def forward(self, x):
        """forward."""
        x = x.transpose(1, 2)
        x = self.pre_projection(x)
        x = self.cbhg(x).transpose(1, 2)
        output = self.post_projection(x).transpose(1, 2)

        return output

class CBHG(torch.nn.Module):
    """CBHG Module."""

    def __init__(
        self,
        hidden_size,
        K=16,
        projection_size=256,
        num_gru_layers=2,
        max_pool_kernel_size=2,
        is_post=False,
    ):
        """init."""
        # :param hidden_size: dimension of hidden unit
        # :param K: # of convolution banks
        # :param projection_size: dimension of projection unit
        # :param num_gru_layers: # of layers of GRUcell
        # :param max_pool_kernel_size: max pooling kernel size
        # :param is_post: whether post processing or not
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.convbank_list = torch.nn.ModuleList()
        self.convbank_list.append(
            torch.nn.Conv1d(
                in_channels=projection_size,
                out_channels=hidden_size,
                kernel_size=1,
                padding=int(np.floor(1 / 2)),
            )
        )

        for i in range(2, K + 1):
            self.convbank_list.append(
                torch.nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=i,
                    padding=int(np.floor(i / 2)),
                )
            )

        self.batchnorm_list = torch.nn.ModuleList()
        for i in range(1, K + 1):
            self.batchnorm_list.append(torch.nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K

        self.conv_projection_1 = torch.nn.Conv1d(
            in_channels=convbank_outdim,
            out_channels=hidden_size,
            kernel_size=3,
            padding=int(np.floor(3 / 2)),
        )
        self.conv_projection_2 = torch.nn.Conv1d(
            in_channels=hidden_size,
            out_channels=projection_size,
            kernel_size=3,
            padding=int(np.floor(3 / 2)),
        )
        self.batchnorm_proj_1 = torch.nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = torch.nn.BatchNorm1d(projection_size)

        self.max_pool = torch.nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = torch.nn.GRU(
            self.projection_size,
            self.hidden_size // 2,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

    def _conv_fit_dim(self, x, kernel_size=3):
        """_conv_fit_dim."""
        if kernel_size % 2 == 0:
            return x[:, :, :-1]
        else:
            return x

    def forward(self, input_):
        """forward."""
        input_ = input_.contiguous()
        # batch_size = input_.size(0)
        # total_length = input_.size(-1)

        convbank_list = list()
        convbank_input = input_

        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(
            zip(self.convbank_list, self.batchnorm_list)
        ):
            convbank_input = torch.relu(
                batchnorm(self._conv_fit_dim(conv(convbank_input), k + 1).contiguous())
            )
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:, :, :-1]

        # Projection
        conv_projection = torch.relu(
            self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat)))
        )
        conv_projection = (
            self.batchnorm_proj_2(
                self._conv_fit_dim(self.conv_projection_2(conv_projection))
            )
            + input_
        )

        # Highway networks
        highway = self.highway.forward(conv_projection.transpose(1, 2))

        # Bidirectional GRU

        self.gru.flatten_parameters()
        out, _ = self.gru(highway)

        return out


class Highwaynet(torch.nn.Module):
    """Highway network."""

    def __init__(self, num_units, num_layers=4):
        """init."""
        # :param num_units: dimension of hidden unit
        # :param num_layers: # of highway layers

        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(torch.nn.Linear(num_units, num_units))
            self.gates.append(torch.nn.Linear(num_units, num_units))

    def forward(self, input_):
        """forward."""
        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):

            h = torch.relu(fc1.forward(out))
            t_ = torch.sigmoid(fc2.forward(out))

            c = 1.0 - t_
            out = h * t_ + out * c

        return out

class PositionalEncoding(torch.nn.Module):
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
        self.dropout = torch.nn.Dropout(p=dropout)

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

class Encoder_Postnet(torch.nn.Module):
    """Encoder Postnet."""

    def __init__(self, embed_size, semitone_size=59, Hz2semitone=False):
        """init."""
        super(Encoder_Postnet, self).__init__()

        self.Hz2semitone = Hz2semitone
        if self.Hz2semitone:
            self.emb_pitch = torch.nn.Embedding(semitone_size, embed_size)
        else:
            self.fc_pitch = torch.nn.Linear(1, embed_size)
        # Remember! embed_size must be even!!
        self.fc_pos = torch.nn.Linear(embed_size, embed_size)
        # only 0 and 1 two possibilities
        self.emb_beats = torch.nn.Embedding(2, embed_size)
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

    def forward(self, encoder_out, align_phone, text_phone, pitch, beats):
        """pitch/beats:[batch_size, frame_num]->[batch_size, frame_num，1]."""
        # batch_size = pitch.shape[0]
        # frame_num = pitch.shape[1]
        # embedded_dim = encoder_out.shape[2]

        aligner_out = self.aligner(encoder_out, align_phone, text_phone)

        if self.Hz2semitone:
            pitch = self.emb_pitch(pitch.squeeze(-1))
        else:
            pitch = self.fc_pitch(pitch)
        out = aligner_out + pitch

        beats = self.emb_beats(beats.squeeze(2))
        out = out + beats

        pos_encode = self.pos(torch.transpose(aligner_out, 0, 1))
        pos_out = self.fc_pos(torch.transpose(pos_encode, 0, 1))

        out = out + pos_out

        return out


class GatedConv(torch.nn.Module):
    """GatedConv."""

    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        """init."""
        super(GatedConv, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=input_size,
            out_channels=2 * input_size,
            kernel_size=(width, 1),
            stride=(1, 1),
            padding=(width // 2 * (1 - nopad), 0),
        )
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=(4 * (1 - dropout)) ** 0.5)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x_var):
        """forward."""
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * torch.sigmoid(gate)
        return out


class StackedCNN(torch.nn.Module):
    """Stacked CNN class."""

    def __init__(self, num_layers, input_size, cnn_kernel_width=3, dropout=0.2):
        """init."""
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GatedConv(input_size, cnn_kernel_width, dropout))

    def forward(self, x):
        """forward."""
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
        return x

class GLU(torch.nn.Module):
    """GLU."""

    def __init__(self, num_layers, hidden_size, cnn_kernel_width, dropout, input_size):
        """init."""
        super(GLU, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size, cnn_kernel_width, dropout)

    def forward(self, emb):
        """forward."""
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)

        emb_remap = _shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        return out.squeeze(3).contiguous()

class MultiheadAttention(torch.nn.Module):
    """Multihead attention mechanism (dot attention)."""

    def __init__(self, num_hidden_k):
        """:param num_hidden_k: dimension of hidden."""
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = torch.nn.Dropout(p=0.1)

    def forward(
        self, key, value, query, mask=None, query_mask=None, gaussian_factor=None
    ):
        """forward."""
        # Get attention score
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        if gaussian_factor is not None:
            attn = attn - gaussian_factor

        # Masking to ignore padding (key side)
        if mask is not None:
            attn = attn.masked_fill(mask, -(2 ** 32) + 1)
            attn = t.softmax(attn, dim=-1)
        else:
            attn = t.softmax(attn, dim=-1)

        # Masking to ignore padding (query side)
        if query_mask is not None:
            attn = attn * query_mask

        # Dropout
        # attn = self.attn_dropout(attn)

        # Get Context Vector
        result = t.bmm(attn, value)

        return result, attn

class Attention(torch.nn.Module):
    """Attention Network."""

    def __init__(self, num_hidden, h=4, local_gaussian=False):
        """init."""
        # :param num_hidden: dimension of hidden
        # :param h: num of heads

        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = torch.nn.Linear(num_hidden, num_hidden, bias=False)
        self.value = torch.nn.Linear(num_hidden, num_hidden, bias=False)
        self.query = torch.nn.Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.local_gaussian = local_gaussian
        if local_gaussian:
            self.local_gaussian_factor = Variable(
                t.tensor(30.0), requires_grad=True
            ).float()
        else:
            self.local_gaussian_factor = None

        self.residual_dropout = torch.nn.Dropout(p=0.1)

        self.final_linear = torch.nn.Linear(num_hidden * 2, num_hidden)

        self.layer_norm_1 = torch.nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):
        """forward."""
        batch_size = memory.size(0)
        seq_k = memory.size(1)
        seq_q = decoder_input.size(1)

        # Repeat masks h times
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
            query_mask = query_mask.repeat(self.h, 1, 1)
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)

        # Make multihead
        key = self.key(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(memory).view(
            batch_size, seq_k, self.h, self.num_hidden_per_attn
        )
        query = self.query(decoder_input).view(
            batch_size, seq_q, self.h, self.num_hidden_per_attn
        )

        key = (
            key.permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, seq_k, self.num_hidden_per_attn)
        )
        value = (
            value.permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, seq_k, self.num_hidden_per_attn)
        )
        query = (
            query.permute(2, 0, 1, 3)
            .contiguous()
            .view(-1, seq_q, self.num_hidden_per_attn)
        )

        # add gaussian or not
        if self.local_gaussian:
            row = (
                t.arange(1, seq_k + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(batch_size * self.h, 1, seq_k)
            )
            col = (
                t.arange(1, seq_k + 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size * self.h, seq_k, 1)
            )
            local_gaussian = t.pow(row - col, 2).float().to(key.device.type)
            self.local_gaussian_factor = self.local_gaussian_factor.to(key.device.type)
            local_gaussian = local_gaussian / self.local_gaussian_factor
        else:
            local_gaussian = None

        # Get context vector
        result, attns = self.multihead(
            key,
            value,
            query,
            mask=mask,
            query_mask=query_mask,
            gaussian_factor=local_gaussian,
        )

        attns = attns.view(self.h, batch_size, seq_q, seq_k)
        attns = attns.permute(1, 0, 2, 3)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = t.cat([decoder_input, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = result + decoder_input

        # result = self.residual_dropout(result)

        # Layer normalization
        result = self.layer_norm_1(result)

        return result, attns

class TransformerGLULayer(torch.nn.Module):
    """TransformerGLULayer."""

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        activation="relu",
        glu_kernel=3,
        local_gaussian=False,
        device="cuda",
    ):
        """init."""
        super(TransformerGLULayer, self).__init__()
        self.self_attn = Attention(
            h=nhead, num_hidden=d_model, local_gaussian=local_gaussian
        )
        self.GLU = GLU(1, d_model, glu_kernel, dropout, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        """__setstate__."""
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerGLULayer, self).__setstate__(state)

    def forward(self, src, mask=None, query_mask=None):
        """forward."""
        src1 = self.norm1(src)
        src2, att_weight = self.self_attn(src1, src1, mask=mask, query_mask=query_mask)
        src3 = src + self.dropout1(src2)
        src3 = src3 * SCALE_WEIGHT
        src4 = self.norm2(src3)
        src5 = self.GLU(src4)
        src5 = src5.transpose(1, 2)
        src6 = src3 + self.dropout2(src5)
        src6 = src6 * SCALE_WEIGHT
        return src6, att_weight

class GLUEncoder(torch.nn.Module):
    """Encoder Network."""

    def __init__(
        self,
        phone_size,
        embed_size,
        padding_idx,
        hidden_size,
        dropout,
        GLU_num,
        num_layers=1,
        glu_kernel=3,
    ):
        """init."""
        # :param para: dictionary that contains all parameters
        super(GLUEncoder, self).__init__()

        self.emb_phone = torch.nn.Embedding(
                num_embeddings=phone_size, embedding_dim=embed_size, padding_idx=padding_idx
            )
        # full connected
        self.fc_1 = torch.nn.Linear(embed_size, hidden_size)

        self.GLU_list = torch.nn.ModuleList()
        for i in range(int(GLU_num)):
            self.GLU_list.append(
                GLU(num_layers, hidden_size, glu_kernel, dropout, hidden_size)
            )
        # self.GLU =
        # GLU(num_layers, hidden_size, glu_kernel, dropout, hidden_size)

        self.fc_2 = torch.nn.Linear(hidden_size, embed_size)

    def forward(self, text_phone, pos=None):
        """forward."""
        # text_phone dim: [batch_size, text_phone_length]
        # output dim : [batch_size, text_phone_length, embedded_dim]

        # don't use pos in glu, but leave the field for uniform interface
        embedded_phone = self.emb_phone(text_phone)
        glu_in = self.fc_1(embedded_phone)

        batch_size = glu_in.shape[0]
        text_phone_length = glu_in.shape[1]
        embedded_dim = glu_in.shape[2]

        for glu in self.GLU_list:
            glu_out = glu(glu_in)
            glu_in = glu_out.reshape(batch_size, text_phone_length, embedded_dim)

        glu_out = self.fc_2(glu_in)

        out = embedded_phone + glu_out

        out = out * math.sqrt(0.5)
        return out, text_phone


class GLUDecoder(torch.nn.Module):
    """Decoder Network."""

    def __init__(
        self,
        num_block,
        hidden_size,
        output_dim,
        nhead=4,
        dropout=0.1,
        activation="relu",
        glu_kernel=3,
        local_gaussian=False,
        device="cuda",
    ):
        """init."""
        super(GLUDecoder, self).__init__()
        self.input_norm = torch.nn.LayerNorm(hidden_size)
        decoder_layer = TransformerGLULayer(
            hidden_size,
            nhead,
            dropout,
            activation,
            glu_kernel,
            local_gaussian=local_gaussian,
            device=device,
        )
        self.decoder = torch.nn.TransformerEncoder(decoder_layer, num_block)
        self.output_fc = torch.nn.Linear(hidden_size, output_dim)

        self.hidden_size = hidden_size

    def forward(self, src, pos):
        """forward."""
        if self.training:
            query_mask = pos.ne(0).type(torch.float)
        else:
            query_mask = None
        mask = pos.eq(0).unsqueeze(1).repeat(1, src.size(1), 1)

        src = self.input_norm(src)
        memory, att_weight = self.decoder(src, mask=mask, query_mask=query_mask)
        output = self.output_fc(memory)
        return output, att_weight

# /muskit/layers/transformer
class GLU_Transformer(AbsSVS):
    """Transformer Network."""

    def __init__(
        # network structure related
        self,
        idim,# phone_size,
        odim,#output_dim,
        midi_dim,
        tempo_dim,
        embed_dim,
        # prenet :
        eprenet_conv_layers: int = 3,
        eprenet_conv_chans: int = 256,
        eprenet_conv_filts: int = 5,
        # glu_tf encoder:
        elayers: int = 3,# enc_num_block,
        ehead: int = 4,# enc_nhead,
        eunits: int = 256,
        glu_num_layers: int = 3,
        glu_kernel: int = 3,
        
        dlayers: int = 3,# dec_num_block,
        dhead: int = 4,# dec_nhead,
        dunits: int = 1024,
        postnet_layers: int = 5,
        postnet_chans: int = 256,
        postnet_filts: int = 5,
        n_mels=-1,
        double_mel_loss=True,
        local_gaussian=False,
        semitone_size=59,
        Hz2semitone=False,
        use_batch_norm: bool = True,
        reduction_factor: int = 1,
        # extra embedding related
        embed_integration_type : str = 'add',
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        eprenet_dropout_rate: float = 0.5,
        edropout_rate: float = 0.1,# dropout
        ddropout_rate: float = 0.1,# dropout
        postnet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        loss_type: str = "L1",
    ):
        """init."""
        assert check_argument_types()
        super().__init__()
        # store hyperparameters
        self.idim = idim
        self.midi_dim = midi_dim
        # self.eunits = eunits
        self.odim = odim
        self.eos = idim - 1
        self.embed_integration_type = embed_integration_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        # # define transformer encoder
        # if eprenet_conv_layers != 0:
        #     # encoder prenet
        #     self.encoder_input_layer = torch.nn.Sequential(
        #         EncoderPrenet(
        #             idim=idim,
        #             embed_dim=embed_dim,
        #             elayers=0,
        #             econv_layers=eprenet_conv_layers,
        #             econv_chans=eprenet_conv_chans,
        #             econv_filts=eprenet_conv_filts,
        #             use_batch_norm=use_batch_norm,
        #             dropout_rate=eprenet_dropout_rate,
        #             padding_idx=self.padding_idx,
        #         ),
        #         torch.nn.Linear(eprenet_conv_chans, eunits),
        #     )
        #     self.midi_encoder_input_layer = torch.nn.Sequential(
        #         EncoderPrenet(
        #             idim=midi_dim,
        #             embed_dim=embed_dim,
        #             elayers=0,
        #             econv_layers=eprenet_conv_layers,
        #             econv_chans=eprenet_conv_chans,
        #             econv_filts=eprenet_conv_filts,
        #             use_batch_norm=use_batch_norm,
        #             dropout_rate=eprenet_dropout_rate,
        #             padding_idx=self.padding_idx,
        #         ),
        #         torch.nn.Linear(eprenet_conv_chans, eunits),
        #     )
        #     self.tempo_encoder_input_layer = torch.nn.Sequential(
        #         EncoderPrenet(
        #             idim=tempo_dim,
        #             embed_dim=embed_dim,
        #             elayers=0,
        #             econv_layers=eprenet_conv_layers,
        #             econv_chans=eprenet_conv_chans,
        #             econv_filts=eprenet_conv_filts,
        #             use_batch_norm=use_batch_norm,
        #             dropout_rate=eprenet_dropout_rate,
        #             padding_idx=self.padding_idx,
        #         ),
        #         torch.nn.Linear(eprenet_conv_chans, eunits),
        #     )
        # else:
        #     self.encoder_input_layer = torch.nn.Embedding(
        #         num_embeddings=idim, embedding_dim=eunits, padding_idx=self.padding_idx
        #     )
        #     self.midi_encoder_input_layer = torch.nn.Embedding(
        #         num_embeddings=midi_dim, embedding_dim=eunits, padding_idx=self.padding_idx
        #     )
        #     self.tempo_encoder_input_layer = torch.nn.Embedding(
        #         num_embeddings=tempo_dim, embedding_dim=eunits, padding_idx=self.padding_idx
        #     )

        self.label_encoder_input_layer = GLUEncoder(
            phone_size=idim,
            embed_size=embed_dim,
            padding_idx=self.padding_idx,
            hidden_size=eunits,
            dropout=edropout_rate,
            GLU_num=glu_num_layers,
            num_layers=elayers,
            glu_kernel=glu_kernel,
        )
        self.midi_encoder_input_layer = GLUEncoder(
            phone_size=midi_dim,
            embed_size=embed_dim,
            padding_idx=self.padding_idx,
            hidden_size=eunits,
            dropout=edropout_rate,
            GLU_num=glu_num_layers,
            num_layers=elayers,
            glu_kernel=glu_kernel,
        )
        self.tempo_encoder_input_layer = GLUEncoder(
            phone_size=tempo_dim,
            embed_size=embed_dim,
            padding_idx=self.padding_idx,
            hidden_size=eunits,
            dropout=edropout_rate,
            GLU_num=glu_num_layers,
            num_layers=elayers,
            glu_kernel=glu_kernel,
        )
        if self.embed_integration_type == "add":
            self.projection = torch.nn.Linear(eunits, eunits)
        else:
            self.projection = torch.nn.linear(3 * eunits, eunits)

        self.enc_postnet = Encoder_Postnet(embed_dim, semitone_size, Hz2semitone)

        self.use_mel = n_mels > 0
        if self.use_mel:
            self.double_mel_loss = double_mel_loss
        else:
            self.double_mel_loss = False
        
        if self.use_mel:
            self.decoder = GLUDecoder(
                num_block=dlayers,
                hidden_size=eunits,
                output_dim=n_mels,
                nhead=dhead,
                dropout=ddropout_rate,
                glu_kernel=glu_kernel,
                local_gaussian=local_gaussian
            )
            if self.double_mel_loss:
                self.double_mel = PostNet(n_mels, n_mels, n_mels)
                # # define postnet
                # self.double_mel_loss = (
                #     None
                #     if postnet_layers == 0
                #     else Postnet(
                #         idim=n_mels,
                #         odim=n_mels,
                #         n_layers=postnet_layers,
                #         n_chans=postnet_chans,
                #         n_filts=postnet_filts,
                #         use_batch_norm=use_batch_norm,
                #         dropout_rate=postnet_dropout_rate,
                #     )
                # )
            self.postnet = PostNet(n_mels, odim, (odim // 2 * 2))
            # # define postnet
            # self.postnet = (
            #     None
            #     if postnet_layers == 0
            #     else Postnet(
            #         idim=n_mels,
            #         odim=odim,
            #         n_layers=postnet_layers,
            #         n_chans=postnet_chans,
            #         n_filts=postnet_filts,
            #         use_batch_norm=use_batch_norm,
            #         dropout_rate=postnet_dropout_rate,
            #     )
            # )
        else:
            self.decoder = GLUDecoder(
                num_block=dlayers,
                hidden_size=eunits,
                output_dim=odim,
                nhead=dhead,
                dropout=ddropout_rate,
                glu_kernel=glu_kernel,
                local_gaussian=local_gaussian
            )
            self.postnet = PostNet(odim, odim, (odim // 2 * 2))

            # # define postnet
            # self.postnet = (
            #     None
            #     if postnet_layers == 0
            #     else Postnet(
            #         idim=odim,
            #         odim=odim,
            #         n_layers=postnet_layers,
            #         n_chans=postnet_chans,
            #         n_filts=postnet_filts,
            #         use_batch_norm=use_batch_norm,
            #         dropout_rate=postnet_dropout_rate,
            #     )
            # )
        # self.feat_out = torch.nn.Linear(eunits, odim * reduction_factor)
        # define loss function
        self.criterion = NaiveRNNLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
        )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
        )

    def _reset_parameters(self, init_type):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        label: torch.Tensor,
        label_lengths: torch.Tensor,
        midi: torch.Tensor,
        midi_lengths: torch.Tensor,
        tempo: torch.Tensor,# delete None!
        tempo_lengths: torch.Tensor,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.
        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, Lmax, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            label
            label_lengths
            midi
            midi_lengths
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
        GS Fix:
            arguements from forward func. V.S. **batch from muskit_model.py
            label == durations

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.
        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        midi = midi[:, : midi_lengths.max()]  # for data-parallel
        label = label[:, : label_lengths.max()]  # for data-parallel
        if tempo is not None:
            tempo = tempo[:, : tempo_lengths.max()]  # for data-parallel
        batch_size = text.size(0)

        # label_emb = self.label_encoder_input_layer(label)   # FIX ME: label Float to Int
        # midi_emb = self.midi_encoder_input_layer(midi)
        # tempo_emb = self.tempo_encoder_input_layer(tempo)

        hs_label = self.label_encoder_input_layer(label)   # FIX ME: label Float to Int
        hs_midi = self.midi_encoder_input_layer(midi)
        hs_tempo = self.tempo_encoder_input_layer(tempo)
        # encoder
        # hs_label = self.label_encoder(label_emb)
        # hs_midi = self.midi_encoder(midi_emb)
        # hs_tempo = self.tempo_encoder(tempo_emb)

        if self.embed_integration_type == "add":
            # hs = hs_label + hs_midi
            # if hs_tempo is not None:
            #     hs = hs + hs_tempo
            hs = hs_label + hs_midi + hs_tempo
        else:
            hs = torch.cat(hs_label, hs_midi, dim=-1)
            # if hs_tempo is not None:
            hs = torch.cat(hs, hs_tempo, dim=-1)
        
        hs = self.projection(hs)

        # integrate spk & lang embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)
        
        # decoder
        mel_output, att_weight = self.decoder(post_out, pos=pos_spec)
        if self.double_mel_loss:
            mel_output2 = self.double_mel(mel_output)
        else:
            mel_output2 = mel_output
        zs = self.postnet(mel_output2)

        # zs = zs[:, self.reduction_factor - 1 :: self.reduction_factor]

        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            assert feats_lengths.ge(
                self.reduction_factor
            ).all(), "Output length must be greater than or equal to reduction factor."
            olens = feats_lengths.new(
                [olen - olen % self.reduction_factor for olen in feats_lengths]
            )
            max_olen = max(olens)
            ys = feats[:, :max_olen]
        else:
            ys = feats
            olens = feats_lengths

        # calculate loss values
        l1_loss, l2_loss = self.criterion(after_outs[:, : olens.max()], before_outs[:, : olens.max()], ys, olens)

        if self.loss_type == "L1":
            loss = l1_loss
        elif self.loss_type == "L2":
            loss = l2_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)

        stats = dict(
            l1_loss=l1_loss.item(),
            l2_loss=l2_loss.item(),
        )

        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device
        )
        return loss, stats, weight
    
    # def forward(
    #     self,
    #     characters,
    #     phone,
    #     pitch,
    #     beat,
    #     pos_text=True,
    #     pos_char=None,
    #     pos_spec=None,
    # ):
    #     """forward."""
    #     encoder_out, text_phone = self.encoder(characters.squeeze(2), pos=pos_char)
    #     post_out = self.enc_postnet(encoder_out, phone, text_phone, pitch, beat)
    #     mel_output, att_weight = self.decoder(post_out, pos=pos_spec)

    #     if self.double_mel_loss:
    #         mel_output2 = self.double_mel(mel_output)
    #     else:
    #         mel_output2 = mel_output
    #     output = self.postnet(mel_output2)

    #     return output, att_weight, mel_output, mel_output2

    def inference(
        self,
        text: torch.Tensor,
        label: torch.Tensor,
        midi: torch.Tensor,
        feats: torch.Tensor = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.
        Args:
            text (LongTensor): Batch of padded character ids (Tmax).
            label (Tensor)
            midi (Tensor)
            feats (Tensor): Batch of padded target features (Lmax, odim).
            spembs (Optional[Tensor]): Batch of speaker embeddings (spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (1).
            lids (Optional[Tensor]): Batch of language IDs (1).

        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
        """
        text = text.unsqueeze(0)  # for data-parallel
        feats = feats.unsqueeze(0)  # for data-parallel
        midi = midi.unsqueeze(0)  # for data-parallel
        label = label.unsqueeze(0)  # for data-parallel

        label_emb = self.encoder_input_layer(label)
        midi_emb = self.midi_encoder_input_layer(midi)

        hs_label, (_, _) = self.encoder(label_emb)
        hs_midi, (_, _) = self.midi_encoder(midi_emb)

        if self.midi_embed_integration_type == "add":
            hs = hs_label + hs_midi
            hs = self.midi_projection(hs)
        else:
            hs = torch.cat(hs_label, hs_midi, dim=-1)
            hs = self.midi_projection(hs)

        # integrate spk & lang embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        zs = torch.nn.utils.rnn.pack_padded_sequence(
            hs, label_lengths, batch_first=True
        )

        zs = zs[:, self.reduction_factor - 1 :: self.reduction_factor]
        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return after_outs

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.
        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).
        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).
        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.
        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, eunits).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).
        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, eunits) if
                integration_type is "add" else (B, Tmax, eunits + spk_embed_dim).
        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        else:
            raise NotImplementedError("support only add or concat.")

        return hs
