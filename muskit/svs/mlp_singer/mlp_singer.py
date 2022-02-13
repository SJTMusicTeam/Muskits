# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer-SVS related modules."""

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F

import logging
from typeguard import check_argument_types

from math import ceil
from muskit.torch_utils.nets_utils import make_non_pad_mask
from muskit.layers.mlp.mlp import MLPMixer
from muskit.layers.transformer.mask import subsequent_mask
from muskit.torch_utils.device_funcs import force_gatherable
from muskit.torch_utils.initialize import initialize
from muskit.svs.abs_svs import AbsSVS
from muskit.svs.bytesing.decoder import Postnet


class NaiveRNNLoss(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(self, use_masking=True, use_weighted_masking=False):
        """Initialize Tactoron2 loss module.
        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to apply weighted masking in loss calculation.
        """
        super(NaiveRNNLoss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)

        # NOTE(kan-bayashi): register pre hook function for the compatibility
        # self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(self, after_outs, before_outs, ys, olens):
        """Calculate forward propagation.
        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            ys (Tensor): Batch of padded target features (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
        Returns:
            Tensor: L1 loss value.
            Tensor: Mean square error loss value.
        """
        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            ys = ys.masked_select(masks)
            after_outs = after_outs.masked_select(masks)
            before_outs = before_outs.masked_select(masks)

        # calculate loss
        l1_loss = self.l1_criterion(after_outs, ys) + self.l1_criterion(before_outs, ys)
        mse_loss = self.mse_criterion(after_outs, ys) + self.mse_criterion(
            before_outs, ys
        )

        # make weighted mask and apply it
        if self.use_weighted_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            weights = masks.float() / masks.sum(dim=1, keepdim=True).float()
            out_weights = weights.div(ys.size(0) * ys.size(2))

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(masks).sum()
            mse_loss = mse_loss.mul(out_weights).masked_select(masks).sum()

        return l1_loss, mse_loss


class MLPSinger(AbsSVS):
    """MLPSinger-SVS module
    This is an implementation of MLPSinger for singing voice synthesis
    The features are processed directly over time-domain from music score and
    predict the singing voice features
    """

    def __init__(
        self,
        # network structure related
        idim: int,
        midi_dim: int,
        odim: int,
        embed_dim: int = 256,
        eunits: int = 1024,
        midi_embed_integration_type: str = "add",
        dlayers: int = 16,
        dunits: int = 256,
        chunk_size: int = 200,
        overlap_size: int = 30,
        postnet_layers: int = 5,
        postnet_chans: int = 256,
        postnet_filts: int = 5,
        postnet_dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        reduction_factor: int = 1,
        use_masking=True,
        use_weighted_masking=False,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        edropout_rate: float = 0.1,
        ddropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        loss_type: str = "L1",
    ):
        """Initialize NaiveRNN module.
        Args: TODO
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.midi_dim = midi_dim
        self.eunits = eunits
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        assert (
            chunk_size > 3 * overlap_size
        ), "overlap_size is too small to perform overlapped semgnetation"
        self.loss_type = loss_type

        self.midi_embed_integration_type = midi_embed_integration_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        self.encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=eunits, padding_idx=self.padding_idx
        )
        self.midi_encoder_input_layer = torch.nn.Embedding(
            num_embeddings=midi_dim,
            embedding_dim=eunits,
            padding_idx=self.padding_idx,
        )

        if self.midi_embed_integration_type == "add":
            self.encoder = torch.nn.Linear(eunits, dunits)
        else:
            self.encoder = torch.nn.Linear(2 * eunits, dunits)

        self.decoder = MLPMixer(
            d_model=dunits,
            seq_len=chunk_size,
            expansion_factor=2,
            dropout=ddropout_rate,
            num_layers=dlayers,
        )

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

        # define final projection
        self.feat_out = torch.nn.Linear(dunits, odim * reduction_factor)

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=idim,
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
        ds: torch.Tensor,
        tempo: Optional[torch.Tensor] = None,
        tempo_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        flag_IsValid=False,
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
            label == durations ｜ phone sequence
            midi -> pitch sequence
        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.
        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        midi = midi[:, : midi_lengths.max()]  # for data-parallel
        label = label[:, : label_lengths.max()]  # for data-parallel
        batch_size = text.size(0)

        label_emb = self.encoder_input_layer(label)  # FIX ME: label Float to Int
        midi_emb = self.midi_encoder_input_layer(midi)

        if self.midi_embed_integration_type == "add":
            hs = label_emb + midi_emb
        else:
            hs = torch.cat(label_emb, midi_emb, dim=-1)

        hs = self.encoder(hs)

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

        hs, pad, shape = self._chunking(hs)
        zs = self.decoder(hs)
        zs = self._cat_chunks(hs, pad, shape)

        zs = zs[:, self.reduction_factor - 1 :: self.reduction_factor]

        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)
        before_outs = F.leaky_relu(self.feat_out(zs).view(zs.size(0), -1, self.odim))
        # before_outs = F.leaky_relu(self.feat_out(hs).view(hs.size(0), -1, self.odim))

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
        label = midi.unsqueeze(0)  # for data-parallel

        label_emb = self.encoder_input_layer(label)
        midi_emb = self.midi_encoder_input_layer(midi)

        if self.midi_embed_integration_type == "add":
            hs = label_emb + midi_emb
        else:
            hs = torch.cat(label_emb, midi_emb, dim=-1)

        hs = self.encoder(hs)

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

        zs = self.decoder(hs)

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

    def _chunking(
        self, hs: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int], Tuple[int]]:
        """Chunking sequence into segments
        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
        Returns:
            Tensor: Batch of hidden state segments (B * segment_len, Chunk_size, adim)
            Tuple[Int]: Pad infomration (0, 0, pad_left, pad_right)
            Tuple[Int]: Segmentd shape information (B, segment_len, Chunk_size, adim)
        """
        batch_size, max_length, feat_dim = hs.size()

        # padding interpretation
        # [overlap_size] valid_value [overlap_size]
        # ---------------- segments ---------------
        valid_value = self.chunk_size - 2 * self.overlap_size
        if max_length % valid_value == 0:
            pad_right = self.overlap_size
        else:
            pad_right = valid_value - max_length % valid_value + self.overlap_size
        pad = (0, 0, self.overlap_size, pad_right)
        segment_len = ceil(max_length / valid_value)
        hs = torch.nn.functional.pad(hs, pad, "constant", 0)

        segmented_hs = hs.as_strided(
            (batch_size, segment_len, self.chunk_size, feat_dim),
            (max_length * feat_dim, valid_value * feat_dim, feat_dim, 1),
        )
        segmented_hs = segmented_hs.reshape(
            batch_size * segment_len, self.chunk_size, feat_dim
        )
        return segmented_hs, pad, (batch_size, segment_len, self.chunk_size, feat_dim)

    def _cat_chunks(
        self, segmented_hs: torch.Tensor, pad: Tuple[int], shape: Tuple[int]
    ) -> torch.Tensor:
        """Concatenate the segments into sequences
        Args:
            segmented_hs (Tensor): Batch of segmented hidden states (B * segment_len, Chunk_size, adim).
            pad (Tuple[int]): Pad information (0, 0, pad_left, pad_right)
            shape (Tuple[int]): Segmented shape information (B, segment_len, Chunk_size, adim).
        Return
            Tensor: Batch of hidden state sequences (B, Tmax, adim).
        """
        segmented_hs = segmented_hs.reshape(shape)
        batch_size, segment_len, chunk_size, feature_dim = shape
        _, _, pad_left, pad_right = pad
        valid_dim = chunk_size - 2 * self.overlap_size

        # remove overlap size
        segmented_hs = segmented_hs[:, :, self.overlap_size : -self.overlap_size]

        hs = segmented_hs.reshape(batch_size, segment_len * valid_dim, feature_dim)
        hs = hs[:, : -(pad_right - self.overlap_size), :]
        return hs
