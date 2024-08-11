import argparse
from utils.learning.train_part_naive import train
from pathlib import Path

import os, sys

if os.getcwd() + "/utils/common/" not in sys.path:
    sys.path.insert(1, os.getcwd() + "/utils/common/")
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(
        description="Train Flow Matching on FastMRI challenge Images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-g", "--GPU-NUM", type=int, default=0, help="GPU number to allocate"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "-e", "--num-epochs", type=int, default=3, help="Number of epochs"
    )
    parser.add_argument("-l", "--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "-r", "--report-interval", type=int, default=500, help="Report interval"
    )
    parser.add_argument(
        "-n", "--net-name", type=Path, default="test_FM", help="Name of network"
    )
    parser.add_argument(
        "-t",
        "--data-path-train",
        type=Path,
        default="/Data/train/image/",
        help="Directory of train data",
    )
    parser.add_argument(
        "-v",
        "--data-path-val",
        type=Path,
        default="/Data/val/image/",
        help="Directory of validation data",
    )
    parser.add_argument(
        "-ic",
        "--in_chan",
        type=int,
        default=1,
        help="Size of input channels for network",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=384,
        help="Size of height of inputs for network",
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=384,
        help="Size of height of width for network",
    )
    parser.add_argument(
        "-nc", "--num_channels", type=int, default=64, help="Name of Channels to UNet"
    )
    parser.add_argument(
        "-nr",
        "--num_res_blocks",
        type=int,
        default=1,
        help="Name of Residual Blocks to UNet",
    )
    parser.add_argument(
        "--input-key", type=str, default="image_input", help="Name of input key"
    )
    parser.add_argument(
        "--target-key", type=str, default="image_label", help="Name of target key"
    )
    parser.add_argument(
        "--max-key", type=str, default="max", help="Name of max key in attributes"
    )
    parser.add_argument("--seed", type=int, default=901, help="Fix random seed")
    parser.add_argument(
        "--sigma", type=float, default=0.0, help="Seed for Flow Matching"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = "../result" / args.net_name / "checkpoints"
    args.val_dir = "../result" / args.net_name / "reconstructions_val"
    args.main_dir = "../result" / args.net_name / __file__
    args.val_loss_dir = "../result" / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
