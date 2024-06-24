import argparse
from .preprocess import Cleanups
from . import download_from_osf


def arg_parser():
    parser = argparse.ArgumentParser(description='Babylm')
    parser.add_argument('--train', action='store_true', help='Train dataset')
    parser.add_argument('--evaluate', action='store_true', help='Evaluation dataset')
    parser.add_argument('--test', action='store_true', help='Test dataset')
    parser.add_argument('--dataset', type=str, help='Path to the dataset to use')
    parser.add_argument('--proc_dir', type=str, help='Path to the processed dataset')
    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()
    if args.train:
        mode = "*train"
    elif args.evaluate:
        mode = "*dev"
    elif args.predict:
        mode = "*test"
    else:
        raise ValueError("Invalid mode")

    download_from_osf(
        save_to="./text_data/", renew=False,
    )

    cleanups = Cleanups(
        data_dir=args.dataset,
        proc_dir=args.proc_dir,
        rx_format=mode,
    )

