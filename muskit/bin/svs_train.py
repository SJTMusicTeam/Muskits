#!/usr/bin/env python3
from espnet2.tasks.svs import SVSTask


def get_parser():
    parser = SVSTask.get_parser()
    return parser


def main(cmd=None):
    """SVS training
    Example:
        % python svs_train.py asr --print_config --optim adadelta
        % python svs_train.py --config conf/train_asr.yaml
    """
    SVSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()