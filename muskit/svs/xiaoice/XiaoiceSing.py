# Copyright 2020 Nagoya University (Tomoki Hayashi)
# Copyright 2021 Renmin University of China (Shuai Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""XiaoiceSing related modules."""

import logging

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import librosa

from typeguard import check_argument_types

from muskit.layers.conformer.encoder import (
    Encoder as ConformerEncoder,  # noqa: H301
)
from muskit.layers.fastspeech.fastspeechLoss import (
    FeedForwardTransformerLoss as XiaoiceSingLoss,  # NOQA
)
from muskit.layers.fastspeech.duration_predictor import DurationPredictor
from muskit.layers.fastspeech.length_regulator import LengthRegulator

from muskit.torch_utils.nets_utils import make_non_pad_mask
from muskit.torch_utils.nets_utils import make_pad_mask
from muskit.svs.bytesing.decoder import Postnet
from muskit.svs.xiaoice.pitchloss import PitchLoss, UVULoss
from muskit.layers.transformer.embedding import PositionalEncoding
from muskit.layers.transformer.embedding import ScaledPositionalEncoding

from muskit.layers.transformer.encoder import (
    Encoder as TransformerEncoder,  # noqa: H301
)

from muskit.torch_utils.device_funcs import force_gatherable
from muskit.torch_utils.initialize import initialize
from muskit.svs.abs_svs import AbsSVS
from muskit.svs.gst.style_encoder import StyleEncoder

from muskit.torch_utils.nets_utils import pad_list

import random
from torch.distributions import Beta

Beta_distribution = Beta(
    torch.tensor([0.5]), torch.tensor([0.5])
)  # NOTE(Shuai) Fix Me! Add to args


class XiaoiceSing(AbsSVS):
    """XiaoiceSing module for Singing Voice Synthesis.
    This is a module of XiaoiceSing. A high-quality singing voice synthesis system which
    employs an integrated network for spectrum, F0 and duration modeling. It follows the
    main architecture of FastSpeech while proposing some singing-specific design:
        1) Add features from musical score (e.g.note pitch and length)
        2) Add a residual connection in F0 prediction to attenuate off-key issues
        3) The duration of all the phonemes in a musical note is accumulated to calculate
        the syllable duration loss for rhythm enhancement (syllable loss)
    .. _`XiaoiceSing: A High-Quality and Integrated Singing Voice Synthesis System`:
        https://arxiv.org/pdf/2006.06261.pdf
    """

    def __init__(
        self,
        # network structure related
        idim: int,
        midi_dim: int,
        tempo_dim: int,
        odim: int,
        embed_dim: int,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        postnet_dropout_rate: float = 0.5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        decoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout_rate: float = 0.1,
        reduction_factor: int = 1,
        encoder_type: str = "transformer",
        decoder_type: str = "transformer",
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        # only for conformer
        conformer_rel_pos_type: str = "legacy",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        zero_triu: bool = False,
        conformer_enc_kernel_size: int = 7,
        conformer_dec_kernel_size: int = 31,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        # training related
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        loss_type: str = "L1",
    ):
        """Initialize XiaoiceSing module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            elayers (int): Number of encoder layers.
            eunits (int): Number of encoder hidden units.
            dlayers (int): Number of decoder layers.
            dunits (int): Number of decoder hidden units.
            postnet_layers (int): Number of postnet layers.
            postnet_chans (int): Number of postnet channels.
            postnet_filts (int): Kernel size of postnet.
            postnet_dropout_rate (float): Dropout rate in postnet.
            use_scaled_pos_enc (bool): Whether to use trainable scaled pos encoding.
            use_batch_norm (bool): Whether to use batch normalization in encoder prenet.
            encoder_normalize_before (bool): Whether to apply layernorm layer before
                encoder block.
            decoder_normalize_before (bool): Whether to apply layernorm layer before
                decoder block.
            encoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in encoder.
            decoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in decoder.
            duration_predictor_layers (int): Number of duration predictor layers.
            duration_predictor_chans (int): Number of duration predictor channels.
            duration_predictor_kernel_size (int): Kernel size of duration predictor.
            duration_predictor_dropout_rate (float): Dropout rate in duration predictor.
            reduction_factor (int): Reduction factor.
            encoder_type (str): Encoder type ("transformer" or "conformer").
            decoder_type (str): Decoder type ("transformer" or "conformer").
            transformer_enc_dropout_rate (float): Dropout rate in encoder except
                attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): Dropout rate after encoder
                positional encoding.
            transformer_enc_attn_dropout_rate (float): Dropout rate in encoder
                self-attention module.
            transformer_dec_dropout_rate (float): Dropout rate in decoder except
                attention & positional encoding.
            transformer_dec_positional_dropout_rate (float): Dropout rate after decoder
                positional encoding.
            transformer_dec_attn_dropout_rate (float): Dropout rate in decoder
                self-attention module.
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            spk_embed_integration_type: How to integrate speaker embedding.
            init_type (str): How to initialize transformer parameters.
            init_enc_alpha (float): Initial value of alpha in scaled pos encoding of the
                encoder.
            init_dec_alpha (float): Initial value of alpha in scaled pos encoding of the
                decoder.
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to apply weighted masking in loss
                calculation.
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.midi_dim = midi_dim
        self.tempo_dim = tempo_dim
        self.odim = odim
        self.embed_dim = embed_dim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.loss_type = loss_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )
        
        # check relative positional encoding compatibility
        if "conformer" in [encoder_type, decoder_type]:
            if conformer_rel_pos_type == "legacy":
                if conformer_pos_enc_layer_type == "rel_pos":
                    conformer_pos_enc_layer_type = "legacy_rel_pos"
                    logging.warning(
                        "Fallback to conformer_pos_enc_layer_type = 'legacy_rel_pos' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
                if conformer_self_attn_layer_type == "rel_selfattn":
                    conformer_self_attn_layer_type = "legacy_rel_selfattn"
                    logging.warning(
                        "Fallback to "
                        "conformer_self_attn_layer_type = 'legacy_rel_selfattn' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
            elif conformer_rel_pos_type == "latest":
                assert conformer_pos_enc_layer_type != "legacy_rel_pos"
                assert conformer_self_attn_layer_type != "legacy_rel_selfattn"
            else:
                raise ValueError(f"Unknown rel_pos_type: {conformer_rel_pos_type}")

        # define encoder
        self.phone_encode_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=embed_dim, padding_idx=self.padding_idx
        )
        self.midi_encode_layer = torch.nn.Embedding(
            num_embeddings=midi_dim,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
        )
        self.tempo_encode_layer = torch.nn.Embedding(
            num_embeddings=tempo_dim,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
        )
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=None,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=None,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_enc_kernel_size,
                zero_triu=zero_triu,
            )
        else:
            raise ValueError(f"{encoder_type} is not supported.")

        # define spk and lang embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, adim)
        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, adim)

        # define additional projection for speaker embedding
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        if decoder_type == "transformer":
            self.decoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        elif decoder_type == "conformer":
            self.decoder = ConformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_dec_kernel_size,
            )
        else:
            raise ValueError(f"{decoder_type} is not supported.")

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)
        self.f0_out = torch.nn.Linear(adim, 1)
        self.uvu_out = torch.nn.Linear(adim, 1)

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

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

        # define criterions
        self.criterion = XiaoiceSingLoss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )
        self.pitch_criterion = PitchLoss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )
        self.uvu_criterion = UVULoss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )

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
        tempo: torch.Tensor,
        tempo_lengths: torch.Tensor,
        ds: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        flag_IsValid=False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.
        Args:
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            durations (LongTensor): Batch of padded durations (B, T_text + 1).
            durations_lengths (LongTensor): Batch of duration lengths (B, T_text + 1).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.
        """

        # logging.info(f"tempo: {tempo.max()}")

        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        midi = midi[:, : midi_lengths.max()]  # for data-parallel
        label = label[:, : label_lengths.max()]  # for data-parallel
        tempo = tempo[:, : tempo_lengths.max()]  # for data-parallel
        if pitch is not None:
            pitch = pitch[:, : pitch_lengths.max()]  # for data-parallel
            logpitch = pitch[:,:,0].clone()
            uvu = (logpitch!=0)
            nonzero_idx, nonzero_idy = torch.where(logpitch!=0)
            logpitch[nonzero_idx, nonzero_idy] = torch.log(logpitch[nonzero_idx, nonzero_idy])
            logpitch = logpitch.unsqueeze(-1)

        midi2pitch = self.length_regulator(midi, ds)
        
        for i in range(len(midi2pitch)):
            note = midi2pitch[i,: ].cpu().numpy()
            nonzero_idx = np.where(note!=0)[0]
            note[nonzero_idx] = np.log(librosa.midi_to_hz(note[nonzero_idx]))
            midi2pitch[i, :] = torch.from_numpy(note).to(device=midi.device)

        midi2pitch = midi2pitch.unsqueeze(-1)
        batch_size = text.size(0)

        label_emb = self.phone_encode_layer(label)
        midi_emb = self.midi_encode_layer(midi)
        tempo_emb = self.tempo_encode_layer(tempo)
        input_emb = label_emb + midi_emb + tempo_emb

        x_masks = self._source_mask(label_lengths)
        hs, _ = self.encoder(input_emb, x_masks)  # (B, T_text, adim)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(label_lengths).to(input_emb.device)
        d_outs = self.duration_predictor(hs, d_masks)  # (B, T_text)

        hs = self.length_regulator(hs, ds)  # (B, T_feats, adim)

        # forward decoder
        h_masks = self._source_mask(feats_lengths)
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

        # logging.info(f'zs:{zs.shape}')

        f0_outs = self.f0_out(zs).view(
            zs.size(0), -1, 1
        ) + midi2pitch # (B, T_feats)
        # logging.info(f'f0_outs:{f0_outs.shape}')
        # logging.info(f'midi2pitch:{midi2pitch.shape}')

        uvu_outs = self.uvu_out(zs).view(
            zs.size(0), -1
        )  # (B, T_feats)

        # postnet -> (B, Lmax//r * r, odim)
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
            logpitch = logpitch[:, :max_olen]
            uvu = uvu[:, :max_olen]
        else:
            ys = feats
            olens = feats_lengths

        ilens = label_lengths
        l1_loss, duration_loss = self.criterion(
            after_outs, before_outs, d_outs, ys, ds, ilens, olens
        )
        pitch_loss = self.pitch_criterion(
            uvu, f0_outs.squeeze(-1), logpitch.squeeze(-1)
        )
        uvu_loss = self.uvu_criterion(
            olens, uvu_outs, uvu
        )
        loss = l1_loss + duration_loss + pitch_loss + uvu_loss

        stats = dict(
            loss=loss.item(),
            l1_loss=l1_loss.item(),
            duration_loss=duration_loss.item(),
            pitch_loss=pitch_loss.item(),
            uvu_loss=uvu_loss.item(),
        )

        # report extra information
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(
                encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
            )
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(
                decoder_alpha=self.decoder.embed[-1].alpha.data.item(),
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
        # alpha: float = 1.0,
        # use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.
        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor]): Feature sequence to extract style (N, idim).
            durations (Optional[LongTensor]): Groundtruth of duration (T_text + 1,).
            spembs (Optional[Tensor]): Speaker embedding (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            lids (Optional[Tensor]): Language ID (1,).
            alpha (float): Alpha to control the speed.
            use_teacher_forcing (bool): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.
        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * duration (Tensor): Duration sequence (T_text + 1,).
        """

        label_emb = self.phone_encode_layer(label)
        midi_emb = self.midi_encode_layer(midi)
        tempo_emb = self.tempo_encode_layer(tempo)
        input_emb = label_emb + midi_emb + tempo_emb

        x_masks = None  # self._source_mask(label_lengths)
        hs, _ = self.encoder(input_emb, x_masks)  # (B, T_text, adim)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # forward duration predictor and length regulator
        d_masks = None  # make_pad_mask(label_lengths).to(input_emb.device)
        d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, T_text)
        d_outs_int = torch.floor(d_outs + 0.5).to(dtype=torch.long)  # (B, T_text)

        logging.info(f"ds: {ds}")
        logging.info(f"ds.shape: {ds.shape}")
        logging.info(f"d_outs: {d_outs}")
        logging.info(f"d_outs.shape: {d_outs.shape}")

        # use G.T. duration
        # hs = self.length_regulator(hs, ds)  # (B, T_feats, adim)

        # use duration model output
        hs = self.length_regulator(hs, d_outs_int)  # (B, T_feats, adim)

        # forward decoder
        h_masks = None  # self._source_mask(feats_lengths)
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        # ilens = torch.tensor([len(label)])
        # olens = torch.tensor([len(feats)])
        # ys = feats
        # l1_loss, duration_loss = self.criterion(
        #     after_outs, before_outs, d_outs, ys, ds, ilens, olens
        # )
        # loss = l1_loss + duration_loss
        # logging.info(f"loss: {loss}, l1_loss: {l1_loss}, duration_loss: {duration_loss}")

        return after_outs, None, None

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.
        Args:
            hs (Tensor): Batch of hidden state sequences (B, T_text, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).
        Returns:
            Tensor: Batch of integrated hidden state sequences (B, T_text, adim).
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

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.
        Args:
            ilens (LongTensor): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(
        self, init_type: str, init_enc_alpha: float, init_dec_alpha: float
    ):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)


class XiaoiceSing_noDP(AbsSVS):
    """XiaoiceSing module for Singing Voice Synthesis.
    This is a module of XiaoiceSing. A high-quality singing voice synthesis system which
    employs an integrated network for spectrum, F0 and duration modeling. It follows the
    main architecture of FastSpeech while proposing some singing-specific design:
        1) Add features from musical score (e.g.note pitch and length)
        2) Add a residual connection in F0 prediction to attenuate off-key issues
        3) The duration of all the phonemes in a musical note is accumulated to calculate
        the syllable duration loss for rhythm enhancement (syllable loss)
    .. _`XiaoiceSing: A High-Quality and Integrated Singing Voice Synthesis System`:
        https://arxiv.org/pdf/2006.06261.pdf
    """

    def __init__(
        self,
        # network structure related
        idim: int,
        midi_dim: int,
        tempo_dim: int,
        odim: int,
        embed_dim: int,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        postnet_dropout_rate: float = 0.5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        decoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout_rate: float = 0.1,
        reduction_factor: int = 1,
        encoder_type: str = "transformer",
        decoder_type: str = "transformer",
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        # only for conformer
        conformer_rel_pos_type: str = "legacy",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        zero_triu: bool = False,
        conformer_enc_kernel_size: int = 7,
        conformer_dec_kernel_size: int = 31,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        # training related
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        loss_type: str = "L1",
        use_mixup_training: bool = False,
        loss_mixup_wight: float = 0.1,
        # cycle training process
        use_cycle_process: bool = False,
        cycle_type: str = "single",
        w_svs: float = 0.45,
        w_predictor: float = 0.45,
        w_cycle: float = 0.1,
        predict_type: list = ["midi", "label"],
        predict_criterion_type: str = "CrossEntropy",
    ):
        """Initialize XiaoiceSing module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            elayers (int): Number of encoder layers.
            eunits (int): Number of encoder hidden units.
            dlayers (int): Number of decoder layers.
            dunits (int): Number of decoder hidden units.
            postnet_layers (int): Number of postnet layers.
            postnet_chans (int): Number of postnet channels.
            postnet_filts (int): Kernel size of postnet.
            postnet_dropout_rate (float): Dropout rate in postnet.
            use_scaled_pos_enc (bool): Whether to use trainable scaled pos encoding.
            use_batch_norm (bool): Whether to use batch normalization in encoder prenet.
            encoder_normalize_before (bool): Whether to apply layernorm layer before
                encoder block.
            decoder_normalize_before (bool): Whether to apply layernorm layer before
                decoder block.
            encoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in encoder.
            decoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in decoder.
            duration_predictor_layers (int): Number of duration predictor layers.
            duration_predictor_chans (int): Number of duration predictor channels.
            duration_predictor_kernel_size (int): Kernel size of duration predictor.
            duration_predictor_dropout_rate (float): Dropout rate in duration predictor.
            reduction_factor (int): Reduction factor.
            encoder_type (str): Encoder type ("transformer" or "conformer").
            decoder_type (str): Decoder type ("transformer" or "conformer").
            transformer_enc_dropout_rate (float): Dropout rate in encoder except
                attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): Dropout rate after encoder
                positional encoding.
            transformer_enc_attn_dropout_rate (float): Dropout rate in encoder
                self-attention module.
            transformer_dec_dropout_rate (float): Dropout rate in decoder except
                attention & positional encoding.
            transformer_dec_positional_dropout_rate (float): Dropout rate after decoder
                positional encoding.
            transformer_dec_attn_dropout_rate (float): Dropout rate in decoder
                self-attention module.
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            spk_embed_integration_type: How to integrate speaker embedding.
            init_type (str): How to initialize transformer parameters.
            init_enc_alpha (float): Initial value of alpha in scaled pos encoding of the
                encoder.
            init_dec_alpha (float): Initial value of alpha in scaled pos encoding of the
                decoder.
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to apply weighted masking in loss
                calculation.
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.midi_dim = midi_dim
        self.tempo_dim = tempo_dim
        self.odim = odim
        self.embed_dim = embed_dim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.loss_type = loss_type

        # mixup - augmentation
        self.use_mixup_training = use_mixup_training
        self.loss_mixup_wight = loss_mixup_wight

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # check relative positional encoding compatibility
        if "conformer" in [encoder_type, decoder_type]:
            if conformer_rel_pos_type == "legacy":
                if conformer_pos_enc_layer_type == "rel_pos":
                    conformer_pos_enc_layer_type = "legacy_rel_pos"
                    logging.warning(
                        "Fallback to conformer_pos_enc_layer_type = 'legacy_rel_pos' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
                if conformer_self_attn_layer_type == "rel_selfattn":
                    conformer_self_attn_layer_type = "legacy_rel_selfattn"
                    logging.warning(
                        "Fallback to "
                        "conformer_self_attn_layer_type = 'legacy_rel_selfattn' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
            elif conformer_rel_pos_type == "latest":
                assert conformer_pos_enc_layer_type != "legacy_rel_pos"
                assert conformer_self_attn_layer_type != "legacy_rel_selfattn"
            else:
                raise ValueError(f"Unknown rel_pos_type: {conformer_rel_pos_type}")

        # define encoder
        self.phone_encode_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=embed_dim, padding_idx=self.padding_idx
        )
        self.midi_encode_layer = torch.nn.Embedding(
            num_embeddings=midi_dim,
            embedding_dim=embed_dim,
            padding_idx=self.padding_idx,
        )
        self.tempo_encode_layer = torch.nn.Embedding(
            num_embeddings=tempo_dim, embedding_dim=eunits, padding_idx=self.padding_idx
        )
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=None,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=None,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_enc_kernel_size,
                zero_triu=zero_triu,
            )
        else:
            raise ValueError(f"{encoder_type} is not supported.")

        # define spk and lang embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, adim)
        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, adim)

        # define additional projection for speaker embedding
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        if decoder_type == "transformer":
            self.decoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        elif decoder_type == "conformer":
            self.decoder = ConformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_dec_kernel_size,
            )
        else:
            raise ValueError(f"{decoder_type} is not supported.")

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

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

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

        from muskit.svs.naive_rnn.naive_rnn import NaiveRNNLoss

        # define loss function
        self.criterion = NaiveRNNLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
        )

        # whether to use cycle-process in training stage
        self.use_cycle_process = use_cycle_process
        if self.use_cycle_process:
            self.cycle_type = cycle_type
            from muskit.svs.predictor.TransformerPredictor import TransformerPredictor

            self.predictor = TransformerPredictor(
                predict_type=predict_type,
                predict_criterion_type=predict_criterion_type,
            )
            self.predict_type = predict_type
            self.predict_criterion_type = predict_criterion_type
            self.w_svs = w_svs
            self.w_predictor = w_predictor
            self.w_cycle = w_cycle

            assert cycle_type in ["single", "bi-direct"]
            assert predict_criterion_type in ["CrossEntropy", "CTC"]
            # when cycle_type is Bi-direct, the criterion type of predictor must be CrossEntropy, because of alignment related problems
            assert (
                cycle_type == "bi-direct" and predict_criterion_type == "CrossEntropy"
            ) or cycle_type == "single"

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
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            durations (LongTensor): Batch of padded durations (B, T_text + 1).
            durations_lengths (LongTensor): Batch of duration lengths (B, T_text + 1).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.
        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        midi = midi[:, : midi_lengths.max()]  # for data-parallel
        label = label[:, : label_lengths.max()]  # for data-parallel
        tempo = tempo[:, : tempo_lengths.max()]  # for data-parallel

        batch_size = text.size(0)
        batch_size_origin = batch_size

        args_mixup = None
        if self.use_mixup_training and flag_IsValid == False:
            batch_size_mixup = 2
            args_mixup = {}
            args_mixup["lst"] = random.sample(
                [i for i in range(batch_size)], batch_size_mixup * 2
            )  # mix-up per 2 samples
            args_mixup["w1_lst"] = []
            args_mixup["w2_wst"] = []
            args_mixup["batch_size_mixup"] = 2

            # mix-up augmentation
            feats_mixup = torch.zeros(
                (batch_size_mixup, feats.shape[1], feats.shape[2]),
                dtype=feats.dtype,
                layout=feats.layout,
                device=feats.device,
            )
            feats_lengths_mixup = torch.zeros(
                batch_size_mixup,
                dtype=feats_lengths.dtype,
                layout=feats_lengths.layout,
                device=feats_lengths.device,
            )

            for i in range(batch_size_mixup):
                index1 = args_mixup["lst"][2 * i]
                index2 = args_mixup["lst"][2 * i + 1]

                w1 = Beta_distribution.sample().to(
                    feats.device
                )  # !!! NOTE:  random.random()
                w2 = 1 - w1

                args_mixup["w1_lst"].append(w1)
                args_mixup["w2_wst"].append(w2)

                feats_mixup[i] = w1 * feats[index1] + w2 * feats[index2]
                feats_lengths_mixup[i] = max(
                    feats_lengths[index1], feats_lengths[index2]
                )

            feats_origin = feats
            feats_lengths_origin = feats_lengths
            feats = torch.cat((feats, feats_mixup), 0)
            feats_lengths = torch.cat((feats_lengths, feats_lengths_mixup), 0)

            batch_size = batch_size_origin + batch_size_mixup
            args_mixup["batch_size_origin"] = batch_size_origin
            args_mixup["batch_size"] = batch_size

        after_outs, before_outs, olens, l1_loss, l2_loss = self._singingGenerate(
            text,
            text_lengths,
            feats,
            feats_lengths,
            label,
            label_lengths,
            midi,
            midi_lengths,
            ds,
            tempo,
            tempo_lengths,
            spembs,
            sids,
            lids,
            flag_IsValid,
            args_mixup,
        )

        # calculate loss values - cycle process, when use cycle process & training stage
        if self.use_cycle_process and flag_IsValid == False:
            """Cycle Training Process.
            The cycle training process consists of four parts, which are used only in the training stage.
                Part-1: SVS (Singing Voice Synthesis)
                    Generate singing voice from given music-score(G.T.)
                Part-2: Predict (Automatic Music-score Transcription)
                    Predict music-score information(pitch, phone, spk-id) from given singing voice(G.T.)
                Part-3: Music-score Reconstruction: SVS -> Predict
                    step1. Generate singing voice from given music-score(G.T.)
                    step2. Predict music-score information(pitch, phone, spk-id) from the singing voice generated in step1 of Part-3
                Part-4: Singing Voice Reconstruction: Predict -> SVS
                    step1. Predict music-score information(pitch, phone, spk-id) from given singing voice(G.T.)
                    step2. Generate singing voice from the music-score generated in step1 of Part-4
            Note:
                When self.cycle_type == "single",
                    means the whole cycle training process is Part-1,2,3
                When self.cycle_type == "bi-direct"
                    means the whole cycle training process is Part-1,2,3,4
            """
            # Part-2: Predict (Automatic Music-score Transcription) - without mix-up
            feats_input = feats_origin if self.use_mixup_training else feats
            feats_lengths_input = (
                feats_lengths_origin if self.use_mixup_training else feats_lengths
            )

            if self.predict_criterion_type == "CrossEntropy":
                predictor_midi_loss, predictor_label_loss, predictor_speaker_loss, midi_predict, label_predict, speaker_predict, info_text_out = self.predictor(feats, midi, label, sids, feats_lengths, midi_lengths, label_lengths, args_mixup)
            elif self.predict_criterion_type == "CTC":
                predictor_midi_loss, predictor_label_loss, predictor_speaker_loss, _, _, speaker_predict, _ = self.predictor(feats_input, midi, text, sids, feats_lengths_input, midi_lengths, text_lengths)
            
            predictor_loss = predictor_midi_loss + predictor_label_loss + predictor_speaker_loss

            # Part-3: Music-score Reconstruction: SVS -> Predict - without mix-up
            if self.predict_criterion_type == "CrossEntropy":
                recon_midi_loss, recon_label_loss, recon_speaker_loss, _, _, _, _ = self.predictor(after_outs, midi, label, sids, olens, midi_lengths, label_lengths, args_mixup)
            elif self.predict_criterion_type == "CTC":
                recon_midi_loss, recon_label_loss, recon_speaker_loss, _, _, _, _ = self.predictor(after_outs[:batch_size_origin, : olens.max()], midi, text, sids, olens[:batch_size_origin], midi_lengths, text_lengths)

            # Part-4: Singing Voice Reconstruction: Predict -> SVS
            recon_mel_loss = 0
            if self.cycle_type == "bi-direct":
                if info_text_out != None:
                    # maybe None when only using midi-cycle
                    ds_text_predict = info_text_out['ds_text_outs']
                    text_predict = info_text_out['masked_text_outs']
                    text_lengths_predict = info_text_out['text_outs_lengths']

                text_input = text_predict if "label" in self.predict_type and self.predict_criterion_type == "CrossEntropy" and info_text_out != None else test
                text_lengths_input = text_lengths_predict if "label" in self.predict_type and self.predict_criterion_type == "CrossEntropy" and info_text_out != None else text_lengths
                ds_input = ds_text_predict if "label" in self.predict_type and self.predict_criterion_type == "CrossEntropy" and info_text_out != None else ds
                midi_input = midi_predict if "midi" in self.predict_type and self.predict_criterion_type == "CrossEntropy" else midi
                speaker_input = speaker_predict if "spk" in self.predict_type else sids

                mel_recon_after, mel_recon_before, olens, recon_mel_loss, _ = self._singingGenerate(
                                                                                text_input,             # using predict res
                                                                                text_lengths_input,     # using predict res
                                                                                feats,
                                                                                feats_lengths,
                                                                                label,
                                                                                label_lengths,
                                                                                midi_input,             # using predict res
                                                                                midi_lengths,
                                                                                ds_input,               # using predict res
                                                                                tempo,
                                                                                tempo_lengths,
                                                                                spembs,
                                                                                speaker_input,          # using predict res
                                                                                lids,
                                                                                flag_IsValid,
                                                                                args_mixup,
                                                                            )
            cycle_loss = recon_midi_loss + recon_label_loss + recon_speaker_loss + recon_mel_loss

        # calculate loss values
        if self.loss_type == "L1":
            loss = l1_loss
            if self.use_cycle_process and flag_IsValid == False:
                loss = (
                    self.w_svs * l1_loss
                    + self.w_cycle * cycle_loss
                    + self.w_predictor * predictor_loss
                )
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
        # if self.use_mixup_training and flag_IsValid == False:
        #     stats.update(l1_loss_mixup=l1_loss_mixup)
        if self.use_cycle_process and flag_IsValid == False:
            stats.update(
                cycle_loss=cycle_loss,
                predictor_loss=predictor_loss,
                predictor_midi_loss=predictor_midi_loss,
                predictor_label_loss=predictor_label_loss,
                predictor_speaker_loss=predictor_speaker_loss,
                recon_midi_loss=recon_midi_loss,
                recon_label_loss=recon_label_loss,
                recon_speaker_loss=recon_speaker_loss,
                recon_mel_loss=recon_mel_loss,
            )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        if flag_IsValid == False:
            # training stage
            return loss, stats, weight
        else:
            # validation stage
            return loss, stats, weight, after_outs[:, : olens.max()], feats, olens

    def _singingGenerate(
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
        args_mixup=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.
        Args:
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            durations (LongTensor): Batch of padded durations (B, T_text + 1).
            durations_lengths (LongTensor): Batch of duration lengths (B, T_text + 1).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.
        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        midi = midi[:, : midi_lengths.max()]  # for data-parallel
        label = label[:, : label_lengths.max()]  # for data-parallel
        tempo = tempo[:, : tempo_lengths.max()]  # for data-parallel

        batch_size = text.size(0)

        text_emb = self.phone_encode_layer(text)
        midi_emb = self.midi_encode_layer(midi)

        x_masks = self._source_mask(text_lengths)
        hs, _ = self.encoder(text_emb, x_masks)  # (B, T_text, adim)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        hs = self.length_regulator(hs, ds)

        if self.use_mixup_training and flag_IsValid == False:
            assert args_mixup != None
            batch_size_mixup = args_mixup["batch_size_mixup"]
            batch_size_origin = args_mixup["batch_size_origin"]
            batch_size = args_mixup["batch_size"]

            midi_embed_mixup = torch.zeros(
                (batch_size_mixup, midi_emb.shape[1], midi_emb.shape[2]),
                dtype=midi_emb.dtype,
                layout=midi_emb.layout,
                device=midi_emb.device,
            )
            midi_lengths_mixup = torch.zeros(
                batch_size_mixup,
                dtype=midi_lengths.dtype,
                layout=midi_lengths.layout,
                device=midi_lengths.device,
            )

            hs_mixup = torch.zeros(
                (batch_size_mixup, hs.shape[1], hs.shape[2]),
                dtype=hs.dtype,
                layout=hs.layout,
                device=hs.device,
            )

            for i in range(batch_size_mixup):
                index1 = args_mixup["lst"][2 * i]
                index2 = args_mixup["lst"][2 * i + 1]

                w1 = args_mixup["w1_lst"][i]
                w2 = args_mixup["w2_wst"][i]

                midi_embed_mixup[i] = w1 * midi_emb[index1] + w2 * midi_emb[index2]
                midi_lengths_mixup[i] = max(midi_lengths[index1], midi_lengths[index2])

                hs_mixup[i] = w1 * hs[index1] + w2 * hs[index2]

            midi_emb = torch.cat((midi_emb, midi_embed_mixup), 0)
            midi_lengths = torch.cat((midi_lengths, midi_lengths_mixup), 0)
            hs = torch.cat((hs, hs_mixup), 0)

        hs += midi_emb
        h_masks = self._source_mask(feats_lengths)
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

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
        if self.use_mixup_training and flag_IsValid == False:
            _olens = feats_lengths[:batch_size_origin]
            l1_loss_origin, l2_loss_origin = self.criterion(
                after_outs[:batch_size_origin, : _olens.max()],
                before_outs[:batch_size_origin, : _olens.max()],
                feats[:batch_size_origin],
                feats_lengths[:batch_size_origin],
            )
            _olens = feats_lengths[batch_size_origin:batch_size]
            l1_loss_mixup, l2_loss_mixup = self.criterion(
                after_outs[batch_size_origin:batch_size, : _olens.max()],
                before_outs[batch_size_origin:batch_size, : _olens.max()],
                feats[batch_size_origin:batch_size, : _olens.max()],
                feats_lengths[batch_size_origin:batch_size],
            )
            l1_loss = (
                1 - self.loss_mixup_wight
            ) * l1_loss_origin + self.loss_mixup_wight * l1_loss_mixup
            l2_loss = (
                1 - self.loss_mixup_wight
            ) * l2_loss_origin + self.loss_mixup_wight * l2_loss_mixup
        else:
            l1_loss, l2_loss = self.criterion(
                after_outs[:, : olens.max()], before_outs[:, : olens.max()], ys, olens
            )

        return after_outs, before_outs, olens, l1_loss, l2_loss

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
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.
        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor]): Feature sequence to extract style (N, idim).
            durations (Optional[LongTensor]): Groundtruth of duration (T_text + 1,).
            spembs (Optional[Tensor]): Speaker embedding (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            lids (Optional[Tensor]): Language ID (1,).
            alpha (float): Alpha to control the speed.
            use_teacher_forcing (bool): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.
        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * duration (Tensor): Duration sequence (T_text + 1,).
        """
        # logging.info(f"ds: {ds}")

        text_emb = self.phone_encode_layer(text)
        midi_emb = self.midi_encode_layer(midi)

        x_masks = None  # self._source_mask(text_lengths)
        hs, _ = self.encoder(text_emb, x_masks)  # (B, T_text, adim)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        hs = self.length_regulator(hs, ds)

        # Decoder
        hs += midi_emb
        h_masks = None  # self._source_mask(feats_lengths)
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

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
            hs (Tensor): Batch of hidden state sequences (B, T_text, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).
        Returns:
            Tensor: Batch of integrated hidden state sequences (B, T_text, adim).
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

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.
        Args:
            ilens (LongTensor): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(
        self, init_type: str, init_enc_alpha: float, init_dec_alpha: float
    ):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)