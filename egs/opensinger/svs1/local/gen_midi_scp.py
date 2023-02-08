#!/usr/bin/env python3
import argparse
import logging
import os
from tqdm import tqdm
from typeguard import check_argument_types

from muskit.utils.cli_utils import get_commandline_args

def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description='build midi scp for opensinger',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wavscp")
    parser.add_argument("--midi_dump", default="midi_dump", type=str)
    args = parser.parse_args()

    with open(args.wavscp, 'r', encoding='utf-8') as f:
        wavscp = list(f)
    # logging.info(f'writeout:{args.wavscp[:-7]}midi.scp')
    with open(args.wavscp[:-7]+"midi.scp", 'w', encoding='utf-8') as fscp:
        for line in wavscp:
            key, _ = line.strip().split(' ')
            val = os.path.join(args.midi_dump, key + ".midi")
            if os.path.exists(val):
                fscp.write("{} {}\n".format(key,val))
            # logging.info("{} {}\n".format(key,val))


if __name__ == "__main__":
    main()
