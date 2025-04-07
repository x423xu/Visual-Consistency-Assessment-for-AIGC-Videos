"""Dataset Warpper for T2VQA, T2VQA2, LGVQ"""

import torch
import pytorch_lightning as pl
from .utils import train_test_split_t2vqa_db
from .t2vqa import T2VDataset
from torch.utils.data import DataLoader


class VQADataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        (annos_train, annos_val, annos_test) = train_test_split_t2vqa_db(
            args.data_path, args.anno_file, split=args.split, seed=args.seed
        )
        print(
            f"train: {len(annos_train)}, val: {len(annos_val)}, test: {len(annos_test)}"
        )
        self.annos_train = annos_train
        self.annos_val = annos_val
        self.annos_test = annos_test
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = T2VDataset(
                self.args, anno_file=self.annos_train, phase="train"
            )
            self.val_dataset = T2VDataset(
                self.args, anno_file=self.annos_val, phase="val"
            )
        if stage == "test" or stage is None:
            self.test_dataset = T2VDataset(
                self.args, anno_file=self.annos_test, phase="test"
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        return test_loader
