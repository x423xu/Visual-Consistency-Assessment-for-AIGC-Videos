"""Dataset Warpper for T2VQA, T2VQA2, LGVQ"""

import pytorch_lightning as pl
from .utils import train_test_split_t2vqa_db


class VQADataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        (train, val, test) = train_test_split_t2vqa_db(
            args.data_path, args.anno_file, split=args.split, seed=args.seed
        )
        print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset
