import argparse
import os


parser = argparse.ArgumentParser(
    description="Options for temporal consistency video quality assessment"
)

parser.add_argument(
    "--data_path",
    type=str,
    default="/SSD_zfs/xxy/data/t2vqa/T2VQA-DB",
    help="dataset path",
)
parser.add_argument(
    "--anno_file",
    type=str,
    default="/SSD_zfs/xxy/data/t2vqa/T2VQA_DB_info.txt",
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
