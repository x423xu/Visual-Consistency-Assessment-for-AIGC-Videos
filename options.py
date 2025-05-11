import argparse
import os


parser = argparse.ArgumentParser(
    description="Options for temporal consistency video quality assessment"
)

######################Data options############################
parser.add_argument(
    "--data_name",
    type=str,
    default="T2VQA",
    help="dataset name, e.g. T2VQA, LGVQ",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/data0/xxy/data/T2VQA-DB",
    help="dataset path",
)
parser.add_argument("--batch_size", type=int, default=4, help="batch size for training")
parser.add_argument(
    "--num_workers", type=int, default=8, help="number of workers for data loading"
)

"""options for T2VQA Dataset"""
parser.add_argument(
    "--clip_len", type=int, default=8, help="get 8 frames for each clip"
)
parser.add_argument("--size", type=int, default=224, help="image size for each frame")
parser.add_argument(
    "--frame_interval",
    type=int,
    default=2,
    help="frame interval for each clip, i.e. sample every frame_interval frames",
)
parser.add_argument(
    "--anno_file",
    type=str,
    default="/data0/xxy/data/info.txt",
    help="annotation file path",
)
parser.add_argument(
    "--split",
    type=str,
    default="8-1-1",
    help="train/val/test split ratio, e.g. 8-1-1",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for train/val/test split",
)
parser.add_argument(
    "--debias",
    action="store_true",
    help="if use debiasing dataset, set to True else False",
)

######################Training options############################
parser.add_argument(
    "--mixed_precision",
    action="store_true",
    help="if use mixed precision, set variables to float16 else float32",
)
parser.add_argument(
    "--num_epochs", type=int, default=10, help="number of epochs for training"
)
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for training")
parser.add_argument(
    "--weight_decay", type=float, default=0, help="weight decay for training"
)
parser.add_argument(
    "--wandb",
    action="store_true",
    help="if use wandb for logging, set variables to True else False",
)
parser.add_argument(
    "--eval",
    action="store_true",
    help="eval mode",
)
parser.add_argument(
    "--hard_train",
    action="store_true",
    help="eval mode",
)

#######################Model options############################
parser.add_argument(
    "--backbone",
    type=str,
    default="swinv2",
    choices=["swinv2", "vivit"],
    help="backbone model for feature extraction",
)

parser.add_argument(
    "--flow",
    action="store_true",
    help="if use flow feature, set variables to True else False",
)

parser.add_argument(
    "--tv_align",
    action="store_true",
    help="if use text vision aligner",
)

parser.add_argument(
    "--low_level",
    action="store_true",
    help="if use low_level features",
)

parser.add_argument(
    "--naturalness",
    action="store_true",
    help="if use naturalness features",
)

parser.add_argument(
    "--levels",
    type=int,
    default=10,
    help="number of quality levels",
)

parser.add_argument(
    "--ada_voter",
    action="store_true",
    help="if use adaptive voter",
)
parser.add_argument(
    "--F_branch",
    action="store_true",
    help="if use F-branch",
)
parser.add_argument(
    "--V_branch",
    action="store_true",
    help="if use V-branch",
)

