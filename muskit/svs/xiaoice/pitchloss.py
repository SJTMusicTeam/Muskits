# Copyright 2019 Tomoki Hayashi
# Copyright 2021 Renmin University of China (Shuai Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FastSpeech related modules."""

import logging

import torch
import torch.nn.functional as F

from muskit.layers.fastspeech.duration_predictor import (
    DurationPredictorLoss,  # noqa: H301
)
from muskit.torch_utils.nets_utils import make_non_pad_mask
from muskit.torch_utils.nets_utils import make_pad_mask




class PitchLoss(torch.nn.Module):
    """Loss function module for feed-forward Transformer."""

    def __init__(self, use_masking=True, use_weighted_masking=False, use_f0=False):
        """Initialize feed-forward Transformer loss module.
        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.
        """
        super(PitchLoss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.use_f0 = use_f0

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l2_criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(self, out_masks, logf0=None, y_logf0=None):
        """Calculate forward propagation.
        Args:
            logf0 (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            y_logf0 (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
        Returns:
            Tensor: L1 loss value.
        """
        # apply mask to remove padded part
        if self.use_masking:
            logf0 = logf0.masked_select(out_masks)
            y_logf0 = y_logf0.masked_select(out_masks)

        # calculate loss
        
        l1_loss =  self.l1_criterion(logf0, y_logf0)
        l2_loss =  self.l2_criterion(logf0, y_logf0)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= y_logf0.size(0) * y_logf0.size(2)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            l2_loss = l2_loss.mul(out_weights).masked_select(out_masks).sum()

        return 0.5*(l1_loss + l2_loss)


class UVULoss(torch.nn.Module):
    """Loss function module for feed-forward Transformer."""

    def __init__(self, use_masking=True, use_weighted_masking=False, use_f0=False):
        """Initialize feed-forward Transformer loss module.
        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.
        """
        super(UVULoss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking
        self.use_f0 = use_f0

        # define criterions
        reduction = False if self.use_weighted_masking else True
        self.uvu_criterion = torch.nn.BCELoss(size_average=False, reduce=reduction)

    def forward(self, olens, uvu_pred=None, uvu=None):
        """Calculate forward propagation.
        Args:
            logf0 (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            y_logf0 (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            olens (LongTensor): Batch of the lengths of each target (B,).
        Returns:
            Tensor: L1 loss value.
        """
        # apply mask to remove padded part
        # logging.info(f'olens:{olens.shape}')
        # logging.info(f'uvu_pred:{uvu_pred.shape}')
        # logging.info(f'uvu:{uvu.shape}')
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).to(uvu.device)
            # logging.info(f'out_masks:{out_masks.shape}')
            uvu_pred = uvu_pred.masked_select(out_masks)
            uvu = uvu.masked_select(out_masks)

        # calculate loss
        
        bce_loss =  self.uvu_criterion(torch.sigmoid(uvu_pred), uvu.float())

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(uvu.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= uvu.size(0)

            # apply weight
            bce_loss = bce_loss.mul(out_weights).masked_select(out_masks).sum()

        return bce_loss


