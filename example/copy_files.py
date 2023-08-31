"""Copy files to given directory.

Usage:
    - Copy the input files in the output folder, without hierarchy:
        python copy_files.py -i input_files -o output_folder [--move] [--dry-run]
    - Replicate the input folder hierarchy in the output folder:
        python copy_files.py --root root_folder -i input_files -o output_folder/:dir [--move] [--dry-run]
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from batch_processing import Batch


def parse_command_line(batch: Batch):
    parser = argparse.ArgumentParser(description='Copy files to given directory.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--move', action='store_true', help='move files instead of copy')
    parser.add_argument('-n',
                        '--dry-run',
                        action='store_true',
                        help='print the commands that would be executed, but do not execute them')

    return batch.parse_args(parser)


def parallel_process(input: Path, output: Path, args):
    if args.move:
        logging.info('Moving %s to %s' % (str(input), str(output)))
        if not args.dry_run:
            shutil.move(input, output)
    else:
        logging.info('Copying %s to %s' % (str(input), str(output)))
        if not args.dry_run:
            shutil.copy(input, output)


def main(argv):
    #  Batch instantiation
    batch = Batch(argv)
    batch.set_io_description(input_help='input files', output_help='output directory')

    # Parse arguments
    parse_command_line(batch)

    # Start processing
    batch.run(parallel_process, output_ext=Batch.USE_INPUT_EXT)


if __name__ == "__main__":
    main(sys.argv[1:])
