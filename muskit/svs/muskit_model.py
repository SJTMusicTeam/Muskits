# Copyright 2020 Nagoya University (Tomoki Hayashi)
# Copyright 2021 Carnegie Mellon University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Muskit singing voice synthesis model."""

from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

from typeguard import check_argument_types

from muskit.layers.abs_normalize import AbsNormalize
from muskit.layers.inversible_interface import InversibleInterface
from muskit.train.abs_muskit_model import AbsMuskitModel
from muskit.svs.abs_svs import AbsSVS
from muskit.svs.feats_extract.abs_feats_extract import AbsFeatsExtract

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class MuskitSVSModel(AbsMuskitModel):
    """Muskit model for singing voice synthesis task."""

    def __init__(
        self,
        text_extract: Optional[AbsFeatsExtract],
        feats_extract: Optional[AbsFeatsExtract],
        score_feats_extract: Optional[AbsFeatsExtract],
        durations_extract: Optional[AbsFeatsExtract],
        pitch_extract: Optional[AbsFeatsExtract],
        tempo_extract: Optional[AbsFeatsExtract],
        energy_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        # text_normalize: Optional[AbsNormalize and InversibleInterface],
        # durations_normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        # tempo_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],
        svs: AbsSVS,
    ):
        """Initialize MuskitSVSModel module."""
        assert check_argument_types()
        super().__init__()
        self.text_extract = text_extract
        self.feats_extract = feats_extract
        self.score_feats_extract = score_feats_extract
        self.durations_extract = durations_extract
        self.pitch_extract = pitch_extract
        self.tempo_extract = tempo_extract
        self.energy_extract = energy_extract
        self.normalize = normalize
        # self.text_normalize = text_normalize
        # self.durations_normalize = durations_normalize
        self.pitch_normalize = pitch_normalize
        # self.tempo_normalize = tempo_normalize
        self.energy_normalize = energy_normalize
        self.svs = svs

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        score_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        tempo_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        flag_IsValid = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Caclualte outputs and return the loss tensor.
        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            duration (Optional[Tensor]): Duration tensor.
            duration_lengths (Optional[Tensor]): Duration length tensor (B,).
            score (Optional[Tensor]): Duration tensor.
            score_lengths (Optional[Tensor]): Duration length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor]): Energy tensor.
            energy_lengths (Optional[Tensor]): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
        Returns:
            Tensor: Loss scalar tensor.
            Dict[str, float]: Statistics to be monitored.
            Tensor: Weight tensor to summarize losses.
        """
        with autocast(False):
            # if self.text_extract is not None and text is None:
            #     text, text_lengths = self.text_extract(
            #         input=text,
            #         input_lengths=text_lengths,
            #     )
            # Extract features
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(singing, singing_lengths) # singing to spec feature (frame level)
            else:
                # Use precalculated feats (feats_type != raw case)
                feats, feats_lengths = singing, singing_lengths

            # logging.info(f"pitch_lengths - DIO: {pitch_lengths}")
            # logging.info(f"score_lengths - midi: {score_lengths}")
            # logging.info(f"singing_lengths: {singing_lengths}")

            # logging.info(f"phone_lengths: {durations_lengths}")
            # logging.info(f"feats_lengths: {feats_lengths}")
            
            # quit()

            # Extract auxiliary features
            if self.score_feats_extract is not None:
                durations, durations_lengths, score, score_lengths, \
                    tempo, tempo_lengths = self.score_feats_extract(durations=durations.unsqueeze(-1),\
                                                                    durations_lengths=durations_lengths,\
                                                                    score=score.unsqueeze(-1),\
                                                                    score_lengths=score_lengths,\
                                                                    tempo=tempo.unsqueeze(-1),\
                                                                    tempo_lengths=tempo_lengths)
                # score : 128 midi pitch
                # tempo : bpm
                # duration : 
                #   input-> phone-id seqence | output -> frame level(取众数 from window) or syllable level

            # # print(f"singing: {singing}")
            # print(f"singing.shape: {singing.shape}")
            # print(f"singing_lengths: {singing_lengths}")
            # # print(f"score.shape: {score.shape}")
            # print(f"score_lengths: {score_lengths}")

            if self.pitch_extract is not None and pitch is None:
                pitch, pitch_lengths = self.pitch_extract(
                    input=singing,
                    input_lengths=singing_lengths,
                    feats_lengths=feats_lengths
                )

            if self.energy_extract is not None and energy is None:
                energy, energy_lengths = self.energy_extract(
                    singing,
                    singing_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )

            # Normalize
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
            if self.pitch_normalize is not None:
                pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            if self.energy_normalize is not None:
                energy, energy_lengths = self.energy_normalize(energy, energy_lengths)

        # Make batch for svs inputs
        batch = dict(
            text=text,
            text_lengths=text_lengths,
            feats=feats,
            feats_lengths=feats_lengths,
            flag_IsValid = flag_IsValid,
        )

        # logging.info(f"pitch_lengths - DIO: {pitch_lengths}")
        # logging.info(f"score_lengths - midi: {score_lengths}")

        # logging.info(f"phone_lengths: {durations_lengths}")
        # logging.info(f"feats_lengths: {feats_lengths}")

        # quit()
        
        # batch_size = text.size(0)
        # for index in range(batch_size):
        #     if durations_lengths[index] != feats_lengths[index]:
        #         if score is not None and pitch is None:
        #             length = min(score_lengths[index], durations_lengths[index], feats_lengths[index])
        #         if self.pitch_extract is not None and pitch is not None:
        #             length = min(pitch_lengths[index], score_lengths[index], durations_lengths[index], feats_lengths[index])
        #             pitch[index][length : pitch_lengths[index]] = 0
        #             pitch_lengths[index] = length
        #         score[index][length : score_lengths[index]] = 0
        #         durations[index][length : durations_lengths[index]] = 0
        #         feats[index][length : feats_lengths[index]] = 0
        #         score_lengths[index], durations_lengths[index], feats_lengths[index] = length, length, length

        

        # Update batch for additional auxiliary inputs
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        if durations is not None:
            durations = durations.to(dtype=torch.long)
            batch.update(label=durations, label_lengths=durations_lengths)
        if score is not None and pitch is None:
            score = score.to(dtype=torch.long)
            batch.update(midi=score, midi_lengths=score_lengths)
        if self.pitch_extract is not None and pitch is not None:
            # batch.update(pitch=pitch, pitch_lengths=pitch_lengths)
            batch.update(midi=pitch, midi_lengths=pitch_lengths)
        if self.energy_extract is not None and energy is not None:
            batch.update(energy=energy, energy_lengths=energy_lengths)
        if self.svs.require_raw_singing:
            batch.update(singing=singing, singing_lengths=singing_lengths)

        return self.svs(**batch)

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        singing: torch.Tensor,
        singing_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        score_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        tempo_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.
        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            singing (Tensor): Singing waveform tensor (B, T_wav).
            singing_lengths (Tensor): Singing length tensor (B,).
            durations (Optional[Tensor): Duration tensor.
            durations_lengths (Optional[Tensor): Duration length tensor (B,).
            score (Optional[Tensor): Duration tensor.
            score_lengths (Optional[Tensor): Duration length tensor (B,).
            pitch (Optional[Tensor): Pitch tensor.
            pitch_lengths (Optional[Tensor): Pitch length tensor (B,).
            tempo (Optional[Tensor): Tempo tensor.
            tempo_lengths (Optional[Tensor): Tempo length tensor (B,).
            energy (Optional[Tensor): Energy tensor.
            energy_lengths (Optional[Tensor): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
        Returns:
            Dict[str, Tensor]: Dict of features.
        """
        # feature extraction
        # if self.text_extract is not None:
        #     text, text_lengths = self.text_extract(
        #         input=text,
        #         input_lengths=text_lengths,
        #     )
        if self.feats_extract is not None:
            feats, feats_lengths = self.feats_extract(singing, singing_lengths)
        else:
            # Use precalculated feats (feats_type != raw case)
            feats, feats_lengths = singing, singing_lengths
        if self.score_feats_extract is not None:
            durations, durations_lengths, score, score_lengths, \
                tempo, tempo_lengths = self.score_feats_extract(durations=durations.unsqueeze(-1),\
                                                                durations_lengths=durations_lengths,\
                                                                score=score.unsqueeze(-1),\
                                                                score_lengths=score_lengths,\
                                                                tempo=tempo.unsqueeze(-1),\
                                                                tempo_lengths=tempo_lengths)
        if self.pitch_extract is not None:
            pitch, pitch_lengths = self.pitch_extract(
                input=pitch.unsqueeze(-1),
                input_lengths=pitch_lengths,
            )
        if self.energy_extract is not None:
            energy, energy_lengths = self.energy_extract(
                singing,
                singing_lengths,
                feats_lengths=feats_lengths,
                durations=durations,
                durations_lengths=durations_lengths,
            )
        
        # store in dict
        feats_dict = dict(feats=feats, feats_lengths=feats_lengths)
        if pitch is not None:
            feats_dict.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if energy is not None:
            feats_dict.update(energy=energy, energy_lengths=energy_lengths)

        return feats_dict

    def inference(
        self,
        text: torch.Tensor,
        singing: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        **decode_config,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.
        Args:
            text (Tensor): Text index tensor (T_text).
            singing (Tensor): Singing waveform tensor (T_wav).
            spembs (Optional[Tensor]): Speaker embedding tensor (D,).
            sids (Optional[Tensor]): Speaker ID tensor (1,).
            lids (Optional[Tensor]): Language ID tensor (1,).
            durations (Optional[Tensor): Duration tensor.
            pitch (Optional[Tensor): Pitch tensor.
            energy (Optional[Tensor): Energy tensor.
        Returns:
            Dict[str, Tensor]: Dict of outputs.
        """
        input_dict = dict(text=text)
        if decode_config["use_teacher_forcing"] or getattr(self.svs, "use_gst", False):
            if singing is None:
                raise RuntimeError("missing required argument: 'singing'")
            if self.feats_extract is not None:
                feats = self.feats_extract(singing[None])[0][0]
            else:
                # Use precalculated feats (feats_type != raw case)
                feats = singing
            if self.normalize is not None:
                feats = self.normalize(feats[None])[0][0]
            input_dict.update(feats=feats)
            if self.svs.require_raw_singing:
                input_dict.update(singing=singing)

        if decode_config["use_teacher_forcing"]:
            if durations is not None:
                input_dict.update(durations=durations)

            if self.pitch_extract is not None:
                pitch = self.pitch_extract(
                    singing[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.pitch_normalize is not None:
                pitch = self.pitch_normalize(pitch[None])[0][0]
            if pitch is not None:
                input_dict.update(pitch=pitch)

            if self.energy_extract is not None:
                energy = self.energy_extract(
                    singing[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.energy_normalize is not None:
                energy = self.energy_normalize(energy[None])[0][0]
            if energy is not None:
                input_dict.update(energy=energy)

        if spembs is not None:
            input_dict.update(spembs=spembs)
        if sids is not None:
            input_dict.update(sids=sids)
        if lids is not None:
            input_dict.update(lids=lids)

        output_dict = self.svs.inference(**input_dict, **decode_config)

        if self.normalize is not None and output_dict.get("feat_gen") is not None:
            # NOTE: normalize.inverse is in-place operation
            feat_gen_denorm = self.normalize.inverse(
                output_dict["feat_gen"].clone()[None]
            )[0][0]
            output_dict.update(feat_gen_denorm=feat_gen_denorm)

        return output_dict
