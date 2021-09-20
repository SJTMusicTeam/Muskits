# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer-SVS related modules."""

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from muskit.svs.bytesing.bytesing import Tacotron2Loss as NaiveRNNLoss
from muskit.torch_utils.nets_utils import make_non_pad_mask
from muskit.torch_utils.nets_utils import make_pad_mask
from muskit.svs.bytesing.encoder import Encoder as EncoderPrenet
from muskit.svs.bytesing.decoder import Postnet
from muskit.layers.transformer.embedding import PositionalEncoding
from muskit.layers.transformer.embedding import ScaledPositionalEncoding
from muskit.layers.transformer.mask import subsequent_mask
from muskit.torch_utils.device_funcs import force_gatherable
from muskit.torch_utils.initialize import initialize
from muskit.svs.abs_svs import AbsSVS


class NaiveRNN(AbsSVS):
    """NaiveRNN-SVS module
    This is an implementation of naive RNN for singing voice synthesis
    The features are processed directly over time-domain from music score and
    predict the singing voice features
    """

    def __init__(
        self,
        # network structure related
        idim: int,
        midi_dim: int,
        odim: int,
        embed_dim: int = 512,
        eprenet_conv_layers: int = 3,
        eprenet_conv_chans: int = 256,
        eprenet_conv_filts: int = 5,
        elayers: int = 3,
        eunits: int = 1024,
        ebidirectional: bool = True,
        midi_embed_integration_type: str = "add",
        dlayers: int = 3,
        dunits: int = 1024,
        dbidirectional: bool = True,
        postnet_layers: int = 5,
        postnet_chans: int = 256,
        postnet_filts: int = 5,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        reduction_factor: int = 1,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        eprenet_dropout_rate: float = 0.5,
        edropout_rate: float = 0.1,
        ddropout_rate: float = 0.1,
        postnet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        bce_pos_weight: float = 5.0,
        loss_type: str = "L1",
    ):
        """Initialize NaiveRNN module.
        Args: TODO
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        # store hyperparameters
        self.idim = idim
        self.midi_dim = midi_dim
        self.eunits = eunits
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.loss_type = loss_type

        self.midi_embed_integration_type = midi_embed_integration_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        # define transformer encoder
        if eprenet_conv_layers != 0:
            # encoder prenet
            self.encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=idim,
                    embed_dim=embed_dim,
                    elayers=0,
                    econv_layers=eprenet_conv_layers,
                    econv_chans=eprenet_conv_chans,
                    econv_filts=eprenet_conv_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=eprenet_dropout_rate,
                    padding_idx=self.padding_idx,
                ),
                torch.nn.Linear(eprenet_conv_chans, eunits),
            )
            self.midi_encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=midi_dim,
                    embed_dim=embed_dim,
                    elayers=0,
                    econv_layers=eprenet_conv_layers,
                    econv_chans=eprenet_conv_chans,
                    econv_filts=eprenet_conv_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=eprenet_dropout_rate,
                    padding_idx=self.padding_idx,
                ),
                torch.nn.Linear(eprenet_conv_chans, eunits),
            )
        else:
            self.encoder_input_layer = torch.nn.Embedding(
                num_embeddings=idim, embedding_dim=eunits, padding_idx=self.padding_idx
            )
            self.midi_input_layer = torch.nn.Embedding(
                num_embeddings=idim, embedding_dim=eunits, padding_idx=self.padding_idx
            )

        self.encoder = torch.nn.LSTM(
            input_size=eunits,
            hidden_size=eunits,
            num_layers=elayers,
            batch_first=True,
            dropout=edropout_rate,
            bidirectional=ebidirectional,
            proj_size=eunits,
        )

        if self.midi_embed_integration_type == "add":
            self.midi_projection = torch.nn.Linear(eunits, eunits)
        else:
            self.midi_projection = torch.nn.linear(2 * eunits, eunits)

        self.decoder = torch.nn.LSTM(
            input_size=dunits,
            hidden_size=dunits,
            num_layers=dlayers,
            batch_first=True,
            dropout=ddropout_rate,
            bidirectional=dbidirectional,
            proj_size=dunits,
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
        self.feat_out = torch.nn.Linear(eunits, odim * reduction_factor)
        self.prob_out = torch.nn.Linear(eunits, reduction_factor)

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
            bce_pos_weight=bce_pos_weight,
        )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
        )

    def _reset_parameters(self, init_type, init_enc_alpha=1.0):
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
            label_lenths
            midi
            midi_lengths
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
        midi = midi[:, : feats_lengths.max()]  # for data-parallel
        label = midi[:, : label_lengths.max()]  # for data-parallel
        batch_size = text.size(0)

        feats_masks = self._source_mask(label_lengths)
