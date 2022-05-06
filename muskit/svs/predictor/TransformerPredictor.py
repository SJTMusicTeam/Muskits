# Copyright 2021 Renmin University of China (Shuai Guo)

"""Muskit singing voice synthesis model."""

import logging
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from muskit.layers.conformer.encoder import Encoder as ConformerEncoder
from muskit.layers.transformer.encoder import Encoder as TransformerEncoder

from muskit.torch_utils.nets_utils import make_non_pad_mask
from muskit.torch_utils.nets_utils import make_pad_mask
from muskit.svs.bytesing.decoder import Postnet
from muskit.layers.transformer.embedding import PositionalEncoding
from muskit.layers.transformer.embedding import ScaledPositionalEncoding
from muskit.torch_utils.nets_utils import pad_list


class MaskCrossEntropyLoss(torch.nn.Module):
    """Loss function module for Tacotron2."""

    def __init__(self, use_masking=True):
        """Initialize Tactoron2 loss module.
        Args:
            use_masking (bool): Whether to apply masking
                for padded part in loss calculation.
        """
        super(MaskCrossEntropyLoss, self).__init__()
        self.use_masking = use_masking

        # define criterions
        reduction = "sum"       # "mean"
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, inputs, targets, olens=None):
        """Calculate forward propagation.
        Args:
            inputs (Tensor): Batch of padded inputs probs (B, Lmax, Class-nums).
            targets (Tensor): Batch of padded target (B, Lmax).
            olens (LongTensor): Batch of the lengths of each target (B,).
                                When using CrossEntropyLoss, inputs & targets shares the same length(frame-aligned)
        Returns:
            Tensor: Cross Entropy loss value.
        """
        # make mask and apply it
        if self.use_masking and olens != None:
            class_nums = inputs.shape[2]
            masks = make_non_pad_mask(olens).to(targets.device)
            inputs = inputs.masked_select(
                masks.unsqueeze(-1)
            )  # dim - [B * Lmax * Class-nums]
            inputs = inputs.reshape(-1, class_nums)  # dim - [B * Lmax, Class-nums]
            targets = targets.masked_select(masks)  # dim - [B * Lmax]

        # calculate loss
        loss = self.criterion(inputs, targets)

        return loss


class TransformerPredictor(nn.Module):
    """Singing information predictor. Extract midi, pitch, singer information from mel-spec.
    This is a module of Singing information predictor, used for cycle prrocess in the training stage.
    """

    def __init__(
        self,
        label_nums: int = 63,   # ofuton - 42 | opencpop - 63
        midi_nums: int = 129,
        speaker_nums: int = 4,
        adim: int = 80,
        aheads: int = 4,
        elayers: int = 4,  # original - 6
        eunits: int = 512,
        eunits_spk: int = 128,
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        use_scaled_pos_enc: bool = True,
        encoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        predict_type: list = ["midi", "label"],
        predict_criterion_type: str = "CrossEntropy",
    ):
        """init."""
        super(TransformerPredictor, self).__init__()

        # store hyperparameters
        self.label_nums = label_nums
        self.midi_nums = midi_nums
        self.speaker_nums = speaker_nums
        self.eunits_spk = eunits_spk
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.predict_type = predict_type

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        if "midi" in self.predict_type:
            # define midi predictor
            self.predictor_midi = TransformerEncoder(
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

            # define final midi projection
            self.predictor_linear_out_midi = torch.nn.Linear(adim, midi_nums)

        if "label" in self.predict_type:
            # define label predictor
            self.predictor_label = TransformerEncoder(
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

            # define final label projection
            self.predictor_linear_out_label = torch.nn.Linear(adim, label_nums)

        if "spk" in self.predict_type:
            # define speaker predictor
            self.predictor_spk = nn.GRU(
                input_size=adim,
                hidden_size=eunits_spk,
                num_layers=1,
                batch_first=True,
                dropout=0.1,
                bidirectional=True,
            )
            # define final speaker projection
            self.predictor_linear_out_spk = torch.nn.Linear(eunits_spk * 2, speaker_nums)
            # define speaker criterion
            self.predictor_spk_criterion = MaskCrossEntropyLoss(use_masking=False)

        # define criterion
        self.predict_criterion_type = predict_criterion_type
        if predict_criterion_type == "CrossEntropy":
            self.predictor_criterion = MaskCrossEntropyLoss(use_masking=True)
        elif predict_criterion_type == "CTC":
            self.predictor_criterion = torch.nn.CTCLoss(zero_infinity=True)


    def forward(self, feats, midi, label, spk_ids, feats_lengths, midi_lengths, label_lengths, args_mixup=None):
        """forward.
        Args:
            feats: Batch of lengths (B, T_feats, adim) or (B + B_mixup, T_feats, adim).
            midi: Batch of lengths (B, T_feats).
            label: Batch of lengths (B, T_feats).
            feats_lengths: Batch of input lengths (B,) or (B + B_mixup,). Note that the feats, midi, label are time-aligned on frame level.

            example - args_mixup: 
            {
                'lst': [2, 9, 1, 4], 
                'w1_lst': [tensor([0.3216], device='cuda:5'), tensor([0.0965], device='cuda:5')], 
                'w2_wst': [tensor([0.6784], device='cuda:5'), tensor([0.9035], device='cuda:5')], 
                'batch_size_mixup': 2, 
                'batch_size_origin': 14, 
                'batch_size': 16
            }
        """
        h_masks = self._source_mask(feats_lengths)  # (B, 1, T_feats)
        midi_loss, label_loss, speaker_loss = 0, 0, 0
        masked_midi_outs, masked_label_outs, speaker_predict = None, None, None
        info_text_out = None
        batch_size = feats.shape[0]

        if "midi" in self.predict_type:
            # midi predict
            zs_midi, _ = self.predictor_midi(feats, h_masks)        # (B, T_feats, adim=80)
            zs_midi = self.predictor_linear_out_midi(zs_midi)                 # (B, T_feats, midi classes=129)

            # loss calculation
            if self.predict_criterion_type == "CrossEntropy":
                probs_midi = zs_midi                                    # (B, T_feats, midi classes=129) 
                if args_mixup != None:
                    probs_midi = zs_midi[:args_mixup['batch_size_origin']]
                    probs_midi_mixup = zs_midi[args_mixup['batch_size_origin']:]

                feats_lengths_input = feats_lengths[:args_mixup['batch_size_origin']] if args_mixup != None else feats_lengths
                midi_loss = self.predictor_criterion(probs_midi, midi, feats_lengths_input)
                frame_total = torch.sum(feats_lengths)
                if args_mixup != None:
                    for i in range(args_mixup['batch_size_mixup']):
                        index1 = args_mixup['lst'][2*i]
                        index2 = args_mixup['lst'][2*i + 1]

                        w1 = args_mixup['w1_lst'][i]
                        w2 = args_mixup['w2_wst'][i]

                        midi_lengths_mixup = feats_lengths[i]

                        loss1 = self.predictor_criterion(probs_midi_mixup[i][:midi_lengths_mixup], midi[index1][:midi_lengths_mixup]) * w1
                        loss2 = self.predictor_criterion(probs_midi_mixup[i][:midi_lengths_mixup], midi[index2][:midi_lengths_mixup]) * w2

                        midi_loss = midi_loss + loss1 + loss2

                midi_loss /= frame_total
                
                # make midi predict output for cycle
                h_masks_input = h_masks[:args_mixup['batch_size_origin']] if args_mixup != None else h_masks
                masked_probs_midi = probs_midi * h_masks_input.permute(0,2,1)                             # (B, T_feats, midi classes=129)
                masked_midi_outs = torch.argmax(F.softmax(masked_probs_midi,dim=-1), dim=-1)        # (B, T_feats)

            elif self.predict_criterion_type == "CTC":
                # aggregate G.T.-midi
                # NOTE(Shuai) midi need "+1" in the begin of loss calculation & "-1" in the end of prediction
                # because index-0 is for <blank> in CTC-loss calculation. Note that CTC-loss can`t make time-aligned midi prediction for cycle-singing
                _midi_cal = []
                _midi_length_cal = []
                ds = []
                for i, _ in enumerate(midi_lengths):
                    _midi = midi[i, : midi_lengths[i]] + 1

                    _output, counts = torch.unique_consecutive(
                        _midi, return_counts=True
                    )

                    _midi_cal.append(_output)
                    _midi_length_cal.append(len(_output))
                    ds.append(counts)
                # ds = pad_list(ds, pad_value=0).to(midi.device)
                midi = pad_list(_midi_cal, pad_value=0).to(
                    midi.device, dtype=torch.long
                )
                midi_lengths = torch.tensor(_midi_length_cal).to(midi.device)

                # logging.info(f"midi: {midi.shape}")
                # logging.info(f"midi: {midi}")
                # logging.info(f"midi_lengths: {midi_lengths}")
                # quit()

                probs_midi = F.log_softmax(zs_midi, dim=-1).permute(
                    1, 0, 2
                )  # CTC need shape as (T, N-batch, Class num)
                midi_loss = self.predictor_criterion(
                    probs_midi, midi, feats_lengths, midi_lengths
                )

        if "label" in self.predict_type:
            # label predict
            zs_label, _ = self.predictor_label(feats, h_masks)                                      # (B, T_feats, adim=80)
            zs_label = self.predictor_linear_out_label(zs_label)                                              # (B, T_feats, label classes=50)

            if self.predict_criterion_type == "CrossEntropy":
                probs_label = zs_label                                                              # (B, T_feats, label classes=50) 
                if args_mixup != None:
                    probs_label = zs_label[:args_mixup['batch_size_origin']]
                    probs_label_mixup = zs_label[args_mixup['batch_size_origin']:]
                    
                feats_lengths_input = feats_lengths[:args_mixup['batch_size_origin']] if args_mixup != None else feats_lengths
                label_loss = self.predictor_criterion(probs_label, label, feats_lengths_input)
                frame_total = torch.sum(feats_lengths)
                if args_mixup != None:
                    for i in range(args_mixup['batch_size_mixup']):
                        index1 = args_mixup['lst'][2*i]
                        index2 = args_mixup['lst'][2*i + 1]

                        w1 = args_mixup['w1_lst'][i]
                        w2 = args_mixup['w2_wst'][i]

                        label_lengths_mixup = feats_lengths[i]

                        loss1 = self.predictor_criterion(probs_label_mixup[i][:label_lengths_mixup], label[index1][:label_lengths_mixup]) * w1
                        loss2 = self.predictor_criterion(probs_label_mixup[i][:label_lengths_mixup], label[index2][:label_lengths_mixup]) * w2

                        label_loss = label_loss + loss1 + loss2

                label_loss /= frame_total

                # make label predict output for cycle
                h_masks_input = h_masks[:args_mixup['batch_size_origin']] if args_mixup != None else h_masks
                masked_probs_label = probs_label * h_masks_input.permute(0,2,1)                           # (B, T_feats, label classes=50)
                masked_label_outs = torch.argmax(F.softmax(masked_probs_label,dim=-1), dim=-1)      # (B, T_feats)

                # make label output (time-aligned) to text output, as the transformer-based SVS use text in encoder, not label
                _text_cal = []
                _text_length_cal = []
                ds_text_outs = []
                for i, _ in enumerate(label_lengths):
                    _text = masked_label_outs[i, :label_lengths[i]]

                    _output, counts = torch.unique_consecutive(_text, return_counts=True)
                    
                    _text_cal.append(_output)
                    _text_length_cal.append(len(_output))
                    ds_text_outs.append(counts)
                ds_text_outs = pad_list(ds_text_outs, pad_value=0).to(midi.device)
                masked_text_outs = pad_list(_text_cal, pad_value=0).to(midi.device, dtype=torch.long)
                text_outs_lengths = torch.tensor(_text_length_cal).to(midi.device)

                info_text_out = {}
                info_text_out['ds_text_outs'] = ds_text_outs
                info_text_out['masked_text_outs'] = masked_text_outs
                info_text_out['text_outs_lengths'] = text_outs_lengths

            elif self.predict_criterion_type == "CTC":
                # aggregate G.T.-label
                probs_label = F.log_softmax(zs_label, dim=-1).permute(
                    1, 0, 2
                )  # CTC need shape as (T, N-batch, Class num)
                label_loss = self.predictor_criterion(
                    probs_label, label, feats_lengths, label_lengths
                )

        if "spk" in self.predict_type:
            # speaker predict
            packed_spk_embed = nn.utils.rnn.pack_padded_sequence(
                feats,
                feats_lengths.type(torch.int64).to("cpu"),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_spk_out, hn = self.predictor_spk(packed_spk_embed)
            spk_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_spk_out, batch_first=True
            )
            # hn - (dim direction * layer nums, N_batch, eunits)

            hn = hn.reshape(batch_size, -1)                                                         # (N_batch, dim direction * layer nums * eunits_spk)
            probs_spk = self.predictor_linear_out_spk(hn)                                                     # (N_batch, speaker classes=4)
            speaker_loss = self.predictor_spk_criterion(probs_spk, spk_ids)
            speaker_predict = torch.argmax(F.softmax(probs_spk,dim=-1), dim=-1)                     # (B, 1)


        return midi_loss, label_loss, speaker_loss, masked_midi_outs, masked_label_outs, speaker_predict, info_text_out
    


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
