# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer-SVS related modules."""

import logging
from os import supports_follow_symlinks

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from humanfriendly import parse_size

import numpy as np
import torch
import torch as t
from torch.autograd import Variable
import math
import copy
import torch.nn.functional as F


from typeguard import check_argument_types

from muskit.torch_utils.nets_utils import make_pad_mask, make_non_pad_mask
from muskit.torch_utils.initialize import initialize
from muskit.layers.glu import GLU
from muskit.layers.transformer.attention import MultiHeadedAttention
from muskit.layers.transformer.embedding import PositionalEncoding
from muskit.layers.cbhg import CBHG
from muskit.layers.fastspeech.length_regulator import LengthRegulator

# from muskit.svs.bytesing.encoder import Encoder as EncoderPrenet
from muskit.svs.bytesing.decoder import Postnet
from muskit.svs.naive_rnn.naive_rnn import NaiveRNNLoss
from muskit.torch_utils.device_funcs import force_gatherable
from muskit.svs.abs_svs import AbsSVS
from muskit.svs.gst.style_encoder import StyleEncoder

SCALE_WEIGHT = 0.5**0.5


def _get_activation_fn(activation):
    """_get_activation_fn."""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, N):
    """_get_clones."""
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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


class Encoder_Postnet(torch.nn.Module):
    """Encoder Postnet."""

    def __init__(self):
        """init."""
        super(Encoder_Postnet, self).__init__()

    def aligner(self, encoder_out, align_phone, text_phone):
        """aligner."""
        # align_phone = [batch_size, align_phone_length]
        # text_phone = [batch_size, text_phone_length]
        # align_phone_length( = frame_num) > text_phone_length
        batch = encoder_out.size()[0]
        align_phone_length = align_phone.size()[1]
        emb_dim = encoder_out.size()[2]
        align_phone = align_phone.long()
        text_phone = text_phone.long()
        out = torch.zeros(
            (batch, align_phone_length, emb_dim),
            dtype=torch.float,
            device=encoder_out.device,
        )
        for i in range(batch):
            before_text_phone = text_phone[i][0]
            encoder_ind = 0
            out[i, 0, :] = encoder_out[i][0]
            for j in range(1, align_phone_length):
                if align_phone[i][j] == before_text_phone:
                    out[i, j, :] = encoder_out[i][encoder_ind]
                else:
                    encoder_ind += 1
                    if encoder_ind >= text_phone[i].size()[0]:
                        break
                    before_text_phone = text_phone[i][encoder_ind]
                    out[i, j, :] = encoder_out[i][encoder_ind]

        return out

    def forward(self, encoder_out, align_phone, text_phone):
        """pitch/beats:[batch_size, frame_num]->[batch_size, frame_num, 1]."""
        aligner_out = self.aligner(encoder_out, align_phone, text_phone)

        return aligner_out

class LookUpDurationModel(torch.nn.Module):
    """Attention Network."""

    def __init__(self, phone_size, padding_idx):
        """init."""
        super(LookUpDurationModel, self).__init__()
        self.sum_duration = torch.zeros(phone_size)
        self.cnt_duration = torch.zeros(phone_size)
        self.duration = torch.nn.Parameter(torch.zeros(phone_size), requires_grad=False)
        self.rv = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=False)
        self.padding_idx = padding_idx
        self.dn = torch.nn.Parameter(torch.FloatTensor([0.0]), requires_grad=False)
    
    def forward(self, idx, ds=None):
        if self.sum_duration.device != idx.device:
            self.sum_duration = self.sum_duration.to(idx.device)
            self.cnt_duration = self.cnt_duration.to(idx.device)
            self.duration = self.duration.to(idx.device)
            self.rv = self.rv.to(idx.device)
            self.dn = self.dn.to(idx.device)

        bsz, seqlen = idx.size()[0], idx.size()[1]
        if self.training and ds is not None:
            for i in range(bsz):
                for j in range(seqlen):
                    txt, dur = idx[i,j], ds[i,j]
                    # logging.info(f' txt.dev = { txt.device}')
                    # logging.info(f' dur.dev = { dur.device}')
                    # logging.info(f' sum_duration.dev = { self.sum_duration.device}')

                    self.sum_duration[txt] += dur
                    self.cnt_duration[txt] += 1
                    self.duration[txt] = self.sum_duration[txt] / self.cnt_duration[txt]
            if ((1<idx)&(idx<7)).any():
                self.dn = torch.nn.Parameter(torch.FloatTensor([(self.sum_duration[2:7].sum()/self.cnt_duration[2:7].sum().item())]), requires_grad=False)

            return ds
        else:
            dur = torch.zeros_like(idx, device=idx.device)
            for i in range(bsz):
                for j in range(seqlen):
                    dur[i,j] = self.duration[idx[i,j]]
            for i in range(bsz):
                rc = 1
                n = 1
                for j in range(1,seqlen):
                    if idx[i,j] == self.padding_idx:
                        n = j
                        break
                if n != 1:
                    rc = min(1, ((int(self.dn.item()) - int(self.rv.item()*self.dn.item())) / (dur[i, 1:n].sum().item())))
                delta = 0
                if dur.size()[1] > 1:
                    delta = dur[i, 1:].max()
                if delta < 1:
                    delta = 1
                dur[i, 0] =  max(1, int(self.dn) - delta)
                for j in range(1, n):
                    dur[i, j] = max(1, round(int(rc*dur[i,j].item())))
            return dur

class Attention(torch.nn.Module):
    """Attention Network."""

    def __init__(self, num_hidden, h=4, local_gaussian=False, dropout_rate=0.1):
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

        self.multihead = MultiHeadedAttention(
            self.h, self.num_hidden_per_attn, dropout_rate
        )

        self.local_gaussian = local_gaussian
        if local_gaussian:
            self.local_gaussian_factor = Variable(
                t.tensor(30.0), requires_grad=True
            ).float()
        else:
            self.local_gaussian_factor = None

        self.residual_dropout = torch.nn.Dropout(p=dropout_rate)

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
        # result, attns = self.multihead(
        result = self.multihead(
            key,
            value,
            query,
            mask=mask,
            gaussian_factor=local_gaussian,
        )

        # attns = attns.view(self.h, batch_size, seq_q, seq_k)
        # attns = attns.permute(1, 0, 2, 3)

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

        return result  # , attns


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
    ):
        """init."""
        super(TransformerGLULayer, self).__init__()
        self.self_attn = Attention(
            h=nhead,
            num_hidden=d_model,
            local_gaussian=local_gaussian,
            dropout_rate=dropout,
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
        # src2, att_weight = self.self_attn(src1, src1, mask=mask, query_mask=query_mask)
        src2 = self.self_attn(src1, src1, mask=mask, query_mask=query_mask)
        src3 = src + self.dropout1(src2)
        src3 = src3 * SCALE_WEIGHT
        src4 = self.norm2(src3)
        src5 = self.GLU(src4)
        src5 = src5.transpose(1, 2)
        src6 = src3 + self.dropout2(src5)
        src6 = src6 * SCALE_WEIGHT
        return src6


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


class TransformerEncoder(torch.nn.Module):
    """TransformerEncoder is a stack of N encoder layers.
    Args:
        encoder_layer: an instance of the
            TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        """init."""
        super(TransformerEncoder, self).__init__()
        assert num_layers > 0
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, query_mask=None):
        """Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask:
                the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        # #type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor

        output = src

        for mod in self.layers:
            # output, att_weight = mod(output, mask=mask, query_mask=query_mask)
            output = mod(output, mask=mask, query_mask=query_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output  # , att_weight


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
        )
        self.decoder = TransformerEncoder(decoder_layer, num_block)
        self.output_fc = torch.nn.Linear(hidden_size, output_dim)

        self.hidden_size = hidden_size

    def forward(self, src, pos):  # pos: pad = False mask
        """forward."""
        if self.training:
            query_mask = pos.ne(0).type(torch.float)
        else:
            query_mask = None
        mask = pos.unsqueeze(1).repeat(1, src.size(1), 1)  # pad=Fasle mask

        src = self.input_norm(src)
        # memory, att_weight = self.decoder(src, mask=mask, query_mask=query_mask)
        memory = self.decoder(src, mask=mask, query_mask=query_mask)
        output = self.output_fc(memory)
        return output  # , att_weight


# /muskit/layers/transformer
class GLU_Transformer(AbsSVS):
    """Transformer Network."""

    def __init__(
        # network structure related
        self,
        idim,  # phone_size,
        odim,  # output_dim,
        midi_dim,
        tempo_dim,
        embed_dim,
        # prenet :
        # eprenet_conv_layers: int = 3,
        # eprenet_conv_chans: int = 256,
        # eprenet_conv_filts: int = 5,
        # glu_tf encoder:
        elayers: int = 3,  # enc_num_block,
        # ehead: int = 4,# enc_nhead,
        eunits: int = 256,
        glu_num_layers: int = 3,
        glu_kernel: int = 3,
        dlayers: int = 3,  # dec_num_block,
        dhead: int = 4,  # dec_nhead,
        # dunits: int = 1024,
        postnet_layers: int = 5,
        postnet_chans: int = 256,
        postnet_filts: int = 5,
        local_gaussian=False,
        use_batch_norm: bool = True,
        reduction_factor: int = 1,
        # extra embedding related
        embed_integration_type: str = "add",
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        edropout_rate: float = 0.1,  # dropout
        ddropout_rate: float = 0.1,  # dropout
        postnet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        use_duration: bool = False,
        loss_type: str = "L1",
    ):
        """init."""
        assert check_argument_types()
        super().__init__()
        # store hyperparameters
        self.idim = idim
        self.midi_dim = midi_dim
        self.odim = odim
        self.eos = idim - 1
        self.embed_dim = embed_dim
        self.embed_integration_type = embed_integration_type
        self.reduction_factor = reduction_factor
        self.loss_type = loss_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        self.pos = PositionalEncoding(embed_dim, edropout_rate)

        self.phone_encoder = GLUEncoder(
            phone_size=idim,
            embed_size=embed_dim,
            padding_idx=self.padding_idx,
            hidden_size=eunits,
            dropout=edropout_rate,
            GLU_num=glu_num_layers,
            num_layers=elayers,
            glu_kernel=glu_kernel,
        )
        
        self.midi_encoder_input_layer = torch.nn.Embedding(
            num_embeddings=midi_dim,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
        )

        if self.embed_integration_type == "add":
            self.projection = torch.nn.Linear(embed_dim, embed_dim)
        else:
            self.projection = torch.nn.Linear(2 * embed_dim, embed_dim)
        self.use_duration = use_duration
        # if use_duration:
        if True:
            self.durationmodel = LookUpDurationModel(idim, self.padding_idx)
        # else:
        #     self.durationmodel = None
        # self.enc_postnet = Encoder_Postnet() 
        # define length regulator
        self.length_regulator = LengthRegulator()

        self.fc_midi = torch.nn.Linear(embed_dim, embed_dim)
        self.fc_pos = torch.nn.Linear(embed_dim, embed_dim)

        self.decoder = GLUDecoder(
            num_block=dlayers,
            hidden_size=eunits,
            output_dim=odim,
            nhead=dhead,
            dropout=ddropout_rate,
            glu_kernel=glu_kernel,
            local_gaussian=local_gaussian,
        )

        # define final projection
        self.feat_out = torch.nn.Linear(
            odim // reduction_factor, odim * reduction_factor
        )

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=odim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate,
            )
        )

        # define loss function
        self.criterion = NaiveRNNLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
        )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
        )
        # logging.info(f'spks:{spks}')
        # assert spks is None
        # define spk and lang embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, eunits)
        self.langs = None
        if langs is not None and langs > 1:
            # TODO (not encode yet)
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, eunits)

        # define projection layer
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, eunits)
            else:
                self.projection = torch.nn.Linear(eunits + self.spk_embed_dim, eunits)

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
        tempo: Optional[torch.Tensor] = None,
        tempo_lengths: Optional[torch.Tensor] = None,
        ds: torch.Tensor = None,
        flag_IsValid=False,
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
        # tempo = label[:, : tempo_lengths.max()]  # for data-parallel
        batch_size = text.size(0)

        phone_emb, _ = self.phone_encoder(text)
        midi_emb = self.midi_encoder_input_layer(midi)

        label_emb = self.length_regulator(phone_emb, ds)
        # label_emb = self.enc_postnet(
        #     phone_emb, label, text
        # )
        if self.use_duration:
            ds = self.durationmodel(text, ds)

        midi_emb = F.leaky_relu(self.fc_midi(midi_emb))

        if self.embed_integration_type == "add":
            hs = label_emb + midi_emb
        else:
            hs = torch.cat((label_emb, midi_emb), dim=-1)

        # hs = F.leaky_relu(self.projection(hs))

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

        pos_emb = self.pos(hs)
        pos_out = F.leaky_relu(self.fc_pos(pos_emb))

        hs = hs + pos_out

        # decoder
        zs = self.decoder(
            hs, pos=(~make_pad_mask(midi_lengths)).to(device=hs.device)
        )  # True mask

        zs = zs[:, self.reduction_factor - 1 :: self.reduction_factor]

        before_outs = F.leaky_relu(self.feat_out(zs).view(zs.size(0), -1, self.odim))

        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)

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
        l1_loss, l2_loss = self.criterion(
            after_outs[:, : olens.max()], before_outs[:, : olens.max()], ys, olens
        )

        if self.loss_type == "L1":
            loss = l1_loss
        elif self.loss_type == "L2":
            loss = l2_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)

        stats = dict(
            loss=loss.item(),
            l1_loss=l1_loss.item(),
            l2_loss=l2_loss.item(),
        )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        if flag_IsValid == False:
            return loss, stats, weight
        else:
            return loss, stats, weight, after_outs[:, : olens.max()], ys, olens

    def inference(
        self,
        text: torch.Tensor,
        label: torch.Tensor,
        midi: torch.Tensor,
        ds: torch.Tensor,
        feats: torch.Tensor = None,
        tempo: Optional[torch.Tensor] = None,
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
        phone_emb, _ = self.phone_encoder(text)
        midi_emb = self.midi_encoder_input_layer(midi)

        # if self.use_duration:
        #     ds = self.durationmodel(text, None)
        
        label_emb = self.length_regulator(phone_emb, ds)

        midi_emb = F.leaky_relu(self.fc_midi(midi_emb))

        if self.embed_integration_type == "add":
            # logging.info(f'label_emb.size():{label_emb.size()[1]}')
            # logging.info(f'midi_emb.size():{midi_emb.size()[1]}')
            if label_emb.size()[1] > midi_emb.size()[1]:
                hs = torch.zeros_like(label_emb, device=label_emb.device)
                hs[:, :midi_emb.size()[1], :] = midi_emb
                hs = hs + label_emb
            else:
                hs = torch.zeros_like(midi_emb, device=midi_emb.device)
                hs[:, :label_emb.size()[1], :] = label_emb
                hs = hs + midi_emb
            # hs = label_emb + midi_emb
        else:
            hs = torch.cat((label_emb, midi_emb), dim=-1)

        # hs = F.leaky_relu(self.projection(hs))

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

        pos_emb = self.pos(hs)
        pos_out = F.leaky_relu(self.fc_pos(pos_emb))

        hs = hs + pos_out

        # decoder
        # logging.info(f'midi.size():{torch.LongTensor([midi.size()[1]], device=hs.device)}')
        # torch.LongTensor([midi.size()[1]], device=hs.device)
        zs = self.decoder(
            hs, pos=(~make_pad_mask(torch.LongTensor([hs.size()[1]], device=hs.device))).to(device=hs.device)
        )  # True mask

        zs = zs[:, self.reduction_factor - 1 :: self.reduction_factor]

        before_outs = F.leaky_relu(self.feat_out(zs).view(zs.size(0), -1, self.odim))

        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return after_outs, None, None

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
