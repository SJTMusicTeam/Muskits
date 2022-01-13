#!/usr/bin/env python3

"""SVS mode decoding."""

import argparse
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib
import numpy as np
import soundfile as sf
import torch
from typeguard import check_argument_types

from muskit.utils.cli_utils import get_commandline_args
from muskit.fileio.npy_scp import NpyScpWriter
from muskit.tasks.svs import SVSTask
from muskit.torch_utils.device_funcs import to_device
from muskit.torch_utils.set_all_random_seed import set_all_random_seed

from muskit.svs.naive_rnn.naive_rnn import NaiveRNN
from muskit.svs.glu_transformer.glu_transformer import GLU_Transformer
from muskit.svs.xiaoice.XiaoiceSing import XiaoiceSing
from muskit.svs.xiaoice.XiaoiceSing import XiaoiceSing_noDP

from muskit.utils import config_argparse
from muskit.utils.get_default_kwargs import get_default_kwargs
from muskit.utils.griffin_lim import Spectrogram2Waveform
from muskit.utils.nested_dict_action import NestedDictAction
from muskit.utils.types import str2bool
from muskit.utils.types import str2triple_str
from muskit.utils.types import str_or_none

from muskit.train.class_choices import ClassChoices
from muskit.svs.feats_extract.abs_feats_extract import AbsFeatsExtract
from muskit.layers.abs_normalize import AbsNormalize
from muskit.layers.inversible_interface import InversibleInterface
from muskit.svs.feats_extract.dio import Dio
from muskit.svs.feats_extract.score_feats_extract import FrameScoreFeats
from muskit.svs.feats_extract.score_feats_extract import SyllableScoreFeats
from muskit.svs.feats_extract.energy import Energy
from muskit.svs.feats_extract.log_mel_fbank import LogMelFbank
from muskit.svs.feats_extract.log_spectrogram import LogSpectrogram
from muskit.layers.global_mvn import GlobalMVN

import os
import yaml
from parallel_wavegan.utils import load_model


feats_extractor_choices = ClassChoices(
    "feats_extract",
    classes=dict(fbank=LogMelFbank, spectrogram=LogSpectrogram),
    type_check=AbsFeatsExtract,
    default="fbank",
)

score_feats_extractor_choices = ClassChoices(
    "score_feats_extract",
    classes=dict(
        frame_score_feats=FrameScoreFeats, syllable_score_feats=SyllableScoreFeats
    ),
    type_check=AbsFeatsExtract,
    default="frame_score_feats",
)

pitch_extractor_choices = ClassChoices(
    "pitch_extract",
    classes=dict(dio=Dio),
    type_check=AbsFeatsExtract,
    default=None,
    optional=True,
)
energy_extractor_choices = ClassChoices(
    "energy_extract",
    classes=dict(energy=Energy),
    type_check=AbsFeatsExtract,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(global_mvn=GlobalMVN),
    type_check=AbsNormalize,
    default="global_mvn",
    optional=True,
)
pitch_normalize_choices = ClassChoices(
    "pitch_normalize",
    classes=dict(global_mvn=GlobalMVN),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
energy_normalize_choices = ClassChoices(
    "energy_normalize",
    classes=dict(global_mvn=GlobalMVN),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)

class SingingGenerate:
    """SingingGenerate class

    Examples:
        >>> import soundfile
        >>> svs = SingingGenerate("config.yml", "model.pth")
        >>> wav = svs("Hello World")[0]
        >>> soundfile.write("out.wav", wav.numpy(), svs.fs, "PCM_16")

    """

    def __init__(
        self,
        train_config: Optional[Union[Path, str]],
        # Extraction Methods
        text_extract: Optional[AbsFeatsExtract],
        feats_extract: Optional[AbsFeatsExtract],
        score_feats_extract: Optional[AbsFeatsExtract],
        durations_extract: Optional[AbsFeatsExtract],
        pitch_extract: Optional[AbsFeatsExtract],
        tempo_extract: Optional[AbsFeatsExtract],
        energy_extract: Optional[AbsFeatsExtract],
        normalize: Optional[AbsNormalize and InversibleInterface],
        pitch_normalize: Optional[AbsNormalize and InversibleInterface],
        energy_normalize: Optional[AbsNormalize and InversibleInterface],

        model_file: Optional[Union[Path, str]] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        speed_control_alpha: float = 1.0,
        vocoder_conf: dict = None,
        vocoder_type: str = "HIFI-GAN",
        vocoder_config: str = "",
        vocoder_checkpoint: str = "None",
        dtype: str = "float32",
        device: str = "cpu",
        
    ):
        assert check_argument_types()

        model, train_args = SVSTask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()
        self.device = device
        self.dtype = dtype
        self.train_args = train_args
        self.model = model
        self.svs = model.svs
        self.normalize = model.normalize
        self.feats_extract = model.feats_extract
        # self.duration_calculator = DurationCalculator()
        self.preprocess_fn = SVSTask.build_preprocess_fn(train_args, False)
        self.use_teacher_forcing = use_teacher_forcing

        # Extraction Methods
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
        
        self.vocoder_type = vocoder_type

        logging.info(f"Normalization:\n{self.normalize}")
        logging.info(f"SVS:\n{self.svs}")

        decode_config = {}
        # if isinstance(self.svs, (Tacotron2, Transformer)):
        #     decode_config.update(
        #         {
        #             "threshold": threshold,
        #             "maxlenratio": maxlenratio,
        #             "minlenratio": minlenratio,
        #         }
        #     )
        # if isinstance(self.svs, Tacotron2):
        #     decode_config.update(
        #         {
        #             "use_att_constraint": use_att_constraint,
        #             "forward_window": forward_window,
        #             "backward_window": backward_window,
        #         }
        #     )
        # if isinstance(self.svs, (FastSpeech, FastSpeech2)):
        #     decode_config.update({"alpha": speed_control_alpha})
        decode_config.update({"use_teacher_forcing": use_teacher_forcing})

        self.decode_config = decode_config

        if vocoder_conf is None:
            vocoder_conf = {}
        if self.feats_extract is not None:
            vocoder_conf.update(self.feats_extract.get_parameters())
        if (
            "n_fft" in vocoder_conf
            and "n_shift" in vocoder_conf
            and "fs" in vocoder_conf
        ):
            logging.info(f"vocoder_conf: {vocoder_conf}")
            self.fs = vocoder_conf["fs"]
            if vocoder_type == "Grriffin-Lim":
                self.spc2wav = Spectrogram2Waveform(**vocoder_conf)
            elif vocoder_type == "HIFI-GAN":
                # load config - HiFi-GAN vocoder
                if vocoder_config == "":
                    dirname = os.path.dirname(vocoder_checkpoint)
                    print(f"dirname: {dirname}")
                    vocoder_config = os.path.join(dirname, "config.yml")
                logging.info(f"vocoder_config: {vocoder_config}")
                with open(vocoder_config) as f:
                    config = yaml.load(f, Loader=yaml.Loader)
                logging.info(f"config: {config}")
                logging.info(f"vocoder_config: {vocoder_config}")
                logging.info(f"device: {device}")
                
                # config.update(vars(args))

                model_vocoder = load_model(vocoder_checkpoint, config)
                logging.info(f"Loaded model parameters from {vocoder_checkpoint}.")
                if True:
                    assert hasattr(model_vocoder, "mean"), "Feature stats are not registered."
                    assert hasattr(model_vocoder, "scale"), "Feature stats are not registered."
                model_vocoder.remove_weight_norm()
                model_vocoder = model_vocoder.eval().to(device)

                self.spc2wav = model_vocoder
                logging.info(f"Vocoder: {self.spc2wav}")
        else:
            self.spc2wav = None
            logging.info("Vocoder is not used because vocoder_conf is not sufficient")

    @torch.no_grad()
    def __call__(
        self,
        text: torch.Tensor,
        durations: Union[torch.Tensor, np.ndarray],
        score: Optional[torch.Tensor],
        singing: torch.Tensor = None,
        pitch: Optional[torch.Tensor] = None,
        tempo: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        spembs: Union[torch.Tensor, np.ndarray] = None,
        speed_control_alpha: Optional[float] = None,
    ):
        assert check_argument_types()

        batch = dict(
            text=text,
            durations=durations,
            score=score,
            singing=singing,
            pitch=pitch,
            tempo=tempo,
            energy=energy,
            spembs=spembs,
        )

        cfg = self.decode_config
        # if speed_control_alpha is not None and isinstance(
        #     self.svs, (FastSpeech, FastSpeech2)
        # ):
        #     cfg = self.decode_config.copy()
        #     cfg.update({"alpha": speed_control_alpha})

        batch = to_device(batch, self.device)
        outs, outs_denorm, probs, att_ws = self.model.inference(**batch, **cfg)

        if att_ws is not None:
            duration, focus_rate = self.duration_calculator(att_ws)
        else:
            duration, focus_rate = None, None

        logging.info(f"outs.shape: {outs.shape}")

        assert outs.shape[0] == 1
        outs = outs.squeeze(0)
        outs_denorm = outs_denorm.squeeze(0)

        if self.spc2wav is not None:
            if self.vocoder_type == "Grriffin-Lim":
                wav = torch.tensor(self.spc2wav(outs_denorm.cpu().numpy()))
            elif self.vocoder_type == "HIFI-GAN":
                ### HiFi-GAN Vocoder
                wav = (
                    self.spc2wav.inference(outs_denorm.cpu().numpy(), normalize_before=True)
                    .view(-1)
                    .cpu()
                )
        else:
            wav = None

        return wav, outs, outs_denorm, probs, att_ws, duration, focus_rate


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    threshold: float,
    minlenratio: float,
    maxlenratio: float,
    use_teacher_forcing: bool,
    use_att_constraint: bool,
    backward_window: int,
    forward_window: int,
    speed_control_alpha: float,
    allow_variable_data_keys: bool,
    vocoder_conf: dict,
    vocoder_config: str,
    vocoder_checkpoint: str,
    # Extraction Methods
    feats_extract,
    feats_extract_conf,
    score_feats_extract,
    score_feats_extract_conf,
    pitch_extract,
    pitch_extract_conf,
    energy_extract,
    energy_extract_conf,
    normalize,
    normalize_conf,
    pitch_normalize,
    pitch_normalize_conf,
    energy_normalize,
    energy_normalize_conf,
):
    """Perform SVS model decoding."""
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. feats_extract
    feats_extract_class = feats_extractor_choices.get_class(feats_extract)
    _feats_extract = feats_extract_class(**feats_extract_conf)


    # 3. Normalization layer
    if normalize is not None:
        normalize_class = normalize_choices.get_class(normalize)
        _normalize = normalize_class(**normalize_conf)
    else:
        _normalize = None

    # 4. Extra components
    _score_feats_extract = None
    _pitch_extract = None
    _energy_extract = None
    _pitch_normalize = None
    _energy_normalize = None
    args = dict(
        score_feats_extract=score_feats_extract,
        pitch_extract=pitch_extract,
        energy_extract=energy_extract,
        pitch_normalize=pitch_normalize,
        energy_normalize=energy_normalize,
    )
    logging.info(f"score_feats_extract: {score_feats_extract}")
    if score_feats_extract is not None:
        score_feats_extract_class = score_feats_extractor_choices.get_class(
            score_feats_extract
        )
        logging.info(f"score_feats_extract_class: {score_feats_extract_class}")
        _score_feats_extract = score_feats_extract_class(
            **score_feats_extract_conf
        )
    if getattr(args, "pitch_extract", None) is not None:
        pitch_extract_class = pitch_extractor_choices.get_class(args.pitch_extract)
        if pitch_extract_conf.get("reduction_factor", None) is not None:
            assert pitch_extract_conf.get(
                "reduction_factor", None
            ) == args.svs_conf.get("reduction_factor", 1)
        else:
            pitch_extract_conf["reduction_factor"] = svs_conf.get(
                "reduction_factor", 1
            )
        pitch_extract = pitch_extract_class(**pitch_extract_conf)
    # logging.info(f'pitch_extract:{pitch_extract}')
    if getattr(args, "energy_extract", None) is not None:
        if args.energy_extract_conf.get("reduction_factor", None) is not None:
            assert args.energy_extract_conf.get(
                "reduction_factor", None
            ) == args.svs_conf.get("reduction_factor", 1)
        else:
            args.energy_extract_conf["reduction_factor"] = args.svs_conf.get(
                "reduction_factor", 1
            )
        energy_extract_class = energy_extractor_choices.get_class(
            args.energy_extract
        )
        energy_extract = energy_extract_class(**args.energy_extract_conf)
    if getattr(args, "pitch_normalize", None) is not None:
        pitch_normalize_class = pitch_normalize_choices.get_class(
            args.pitch_normalize
        )
        pitch_normalize = pitch_normalize_class(**args.pitch_normalize_conf)
    if getattr(args, "energy_normalize", None) is not None:
        energy_normalize_class = energy_normalize_choices.get_class(
            args.energy_normalize
        )
        energy_normalize = energy_normalize_class(**args.energy_normalize_conf)

    logging.info(f"_score_feats_extract:{_score_feats_extract}")
    logging.info(f"_feats_extract:{_feats_extract}")
    logging.info(f"_pitch_extract:{_pitch_extract}")
    logging.info(f"_energy_extract:{_energy_extract}")
    logging.info(f"_normalize:{_normalize}")
    logging.info(f"_pitch_normalize:{_pitch_normalize}")
    logging.info(f"_energy_normalize:{_energy_normalize}")

    # 5. Build model
    singingGenerate = SingingGenerate(
        train_config=train_config,
        # Extraction Methods
        text_extract=_score_feats_extract,
        feats_extract=_feats_extract,
        score_feats_extract=_score_feats_extract,
        durations_extract=_score_feats_extract,
        pitch_extract=_pitch_extract,
        tempo_extract=_score_feats_extract,
        energy_extract=_energy_extract,
        normalize=_normalize,
        pitch_normalize=_pitch_normalize,
        energy_normalize=_energy_normalize,
        # Emd of Extraction Methods
        model_file=model_file,
        threshold=threshold,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        use_teacher_forcing=use_teacher_forcing,
        use_att_constraint=use_att_constraint,
        backward_window=backward_window,
        forward_window=forward_window,
        speed_control_alpha=speed_control_alpha,
        vocoder_conf=vocoder_conf,
        vocoder_config=vocoder_config,
        vocoder_checkpoint=vocoder_checkpoint,
        dtype=dtype,
        device=device,
    )

    # 6. Build data-iterator
    # if not singingGenerate.use_speech:
    #     data_path_and_name_and_type = list(
    #         filter(lambda x: x[1] != "speech", data_path_and_name_and_type)
    #     )

    logging.info(f"data_path_and_name_and_type: {data_path_and_name_and_type}")

    loader = SVSTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=SVSTask.build_preprocess_fn(singingGenerate.train_args, False),
        collate_fn=SVSTask.build_collate_fn(singingGenerate.train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 7. Start for-loop
    output_dir = Path(output_dir)
    (output_dir / "norm").mkdir(parents=True, exist_ok=True)
    (output_dir / "denorm").mkdir(parents=True, exist_ok=True)
    (output_dir / "speech_shape").mkdir(parents=True, exist_ok=True)
    (output_dir / "wav").mkdir(parents=True, exist_ok=True)
    (output_dir / "att_ws").mkdir(parents=True, exist_ok=True)
    (output_dir / "probs").mkdir(parents=True, exist_ok=True)
    (output_dir / "durations").mkdir(parents=True, exist_ok=True)
    (output_dir / "focus_rates").mkdir(parents=True, exist_ok=True)

    # Lazy load to avoid the backend error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with NpyScpWriter(
        output_dir / "norm",
        output_dir / "norm/feats.scp",
    ) as norm_writer, NpyScpWriter(
        output_dir / "denorm", output_dir / "denorm/feats.scp"
    ) as denorm_writer, open(
        output_dir / "speech_shape/speech_shape", "w"
    ) as shape_writer, open(
        output_dir / "durations/durations", "w"
    ) as duration_writer, open(
        output_dir / "focus_rates/focus_rates", "w"
    ) as focus_rate_writer:
        for idx, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert _bs == 1, _bs

            # Change to single sequence and remove *_length
            # because inference() requires 1-seq, not mini-batch.
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

            logging.info(f"batch: {batch}")

            logging.info(f"batch['pitch_aug']: {batch['pitch_aug'].item()}")
            logging.info(f"batch['time_aug']: {batch['time_aug'].item()}")
            
            assert batch['pitch_aug'].item() == 0
            assert batch['time_aug'].item() == 1

            del batch['pitch_aug']
            del batch['time_aug']

            start_time = time.perf_counter()
            
            wav, outs, outs_denorm, probs, att_ws, duration, focus_rate = singingGenerate(
                **batch
            )

            key = keys[0]
            insize = next(iter(batch.values())).size(0) + 1
            logging.info(
                "inference speed = {:.1f} frames / sec.".format(
                    int(outs.size(0)) / (time.perf_counter() - start_time)
                )
            )
            logging.info(f"{key} (size:{insize}->{outs.size(0)})")
            if outs.size(0) == insize * maxlenratio:
                logging.warning(f"output length reaches maximum length ({key}).")

            norm_writer[key] = outs.cpu().numpy()
            shape_writer.write(f"{key} " + ",".join(map(str, outs.shape)) + "\n")

            denorm_writer[key] = outs_denorm.cpu().numpy()

            if duration is not None:
                # Save duration and fucus rates
                duration_writer.write(
                    f"{key} " + " ".join(map(str, duration.cpu().numpy())) + "\n"
                )
                focus_rate_writer.write(f"{key} {float(focus_rate):.5f}\n")

                # Plot attention weight
                att_ws = att_ws.cpu().numpy()

                if att_ws.ndim == 2:
                    att_ws = att_ws[None][None]
                elif att_ws.ndim != 4:
                    raise RuntimeError(f"Must be 2 or 4 dimension: {att_ws.ndim}")

                w, h = plt.figaspect(att_ws.shape[0] / att_ws.shape[1])
                fig = plt.Figure(
                    figsize=(
                        w * 1.3 * min(att_ws.shape[0], 2.5),
                        h * 1.3 * min(att_ws.shape[1], 2.5),
                    )
                )
                fig.suptitle(f"{key}")
                axes = fig.subplots(att_ws.shape[0], att_ws.shape[1])
                if len(att_ws) == 1:
                    axes = [[axes]]
                for ax, att_w in zip(axes, att_ws):
                    for ax_, att_w_ in zip(ax, att_w):
                        ax_.imshow(att_w_.astype(np.float32), aspect="auto")
                        ax_.set_xlabel("Input")
                        ax_.set_ylabel("Output")
                        ax_.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax_.yaxis.set_major_locator(MaxNLocator(integer=True))

                fig.set_tight_layout({"rect": [0, 0.03, 1, 0.95]})
                fig.savefig(output_dir / f"att_ws/{key}.png")
                fig.clf()

            if probs is not None:
                # Plot stop token prediction
                probs = probs.cpu().numpy()

                fig = plt.Figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(probs)
                ax.set_title(f"{key}")
                ax.set_xlabel("Output")
                ax.set_ylabel("Stop probability")
                ax.set_ylim(0, 1)
                ax.grid(which="both")

                fig.set_tight_layout(True)
                fig.savefig(output_dir / f"probs/{key}.png")
                fig.clf()

            # TODO(kamo): Write scp
            if wav is not None:
                sf.write(
                    f"{output_dir}/wav/{key}.wav", wav.numpy(), singingGenerate.fs, "PCM_16"
                )

    # remove duration related files if attention is not provided
    if att_ws is None:
        shutil.rmtree(output_dir / "att_ws")
        shutil.rmtree(output_dir / "durations")
        shutil.rmtree(output_dir / "focus_rates")
    if probs is None:
        shutil.rmtree(output_dir / "probs")


def get_parser():
    """Get argument parser."""

    # Add variable objects configurations
    class_choices_list = [
        # --score_extractor and --score_extractor_conf
        score_feats_extractor_choices,
        # --feats_extractor and --feats_extractor_conf
        feats_extractor_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --pitch_extract and --pitch_extract_conf
        pitch_extractor_choices,
        # --pitch_normalize and --pitch_normalize_conf
        pitch_normalize_choices,
        # --energy_extract and --energy_extract_conf
        energy_extractor_choices,
        # --energy_normalize and --energy_normalize_conf
        energy_normalize_choices,
    ]

    parser = config_argparse.ArgumentParser(
        description="SVS Decode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use "_" instead of "-" as separator.
    # "-" is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--key_file",
        type=str_or_none,
    )
    group.add_argument(
        "--allow_variable_data_keys",
        type=str2bool,
        default=False,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file.",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file.",
    )

    group = parser.add_argument_group("Decoding related")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=10.0,
        help="Maximum length ratio in decoding",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Minimum length ratio in decoding",
    )
    group.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold value in decoding",
    )
    group.add_argument(
        "--use_att_constraint",
        type=str2bool,
        default=False,
        help="Whether to use attention constraint",
    )
    group.add_argument(
        "--backward_window",
        type=int,
        default=1,
        help="Backward window value in attention constraint",
    )
    group.add_argument(
        "--forward_window",
        type=int,
        default=3,
        help="Forward window value in attention constraint",
    )
    group.add_argument(
        "--use_teacher_forcing",
        type=str2bool,
        default=False,
        help="Whether to use teacher forcing",
    )
    for class_choices in class_choices_list:
        # Append --<name> and --<name>_conf.
        # e.g. --encoder and --encoder_conf
        class_choices.add_arguments(group)

    parser.add_argument(
        "--speed_control_alpha",
        type=float,
        default=1.0,
        help="Alpha in FastSpeech to change the speed of generated speech",
    )

    group = parser.add_argument_group("Grriffin-Lim related")
    group.add_argument(
        "--vocoder_conf",
        action=NestedDictAction,
        default=get_default_kwargs(Spectrogram2Waveform),
        help="The configuration for Grriffin-Lim",
    )
    group.add_argument(
        "--vocoder_checkpoint",
        default="/data5/gs/vocoder_peter/hifigan-vocoder/exp/train_hifigan.v1_train_nodev_clean_libritts_hifigan-2.v1/checkpoint-50000steps.pkl",
        type=str,
        help="checkpoint file to be loaded.",
    )
    group.add_argument(
        "--vocoder_config",
        default="",
        type=str,
        help="yaml format configuration file. if not explicitly provided, "
        "it will be searched in the checkpoint directory. (default=None)",
    )

    return parser


def main(cmd=None):
    """Run SVS model decoding."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()