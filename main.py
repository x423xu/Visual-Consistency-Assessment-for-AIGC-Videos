import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from options import parser
import pytorch_lightning as pl
from dataset import VQADataModule
from models import TCVQAModule
from pytorch_lightning.loggers import WandbLogger, CSVLogger

WANDB = True


def main():
    args = parser.parse_args()
    args.wandb = WANDB
    # wandb notes
    wandb_notes = f"Learning rate: {args.lr}, Batch size: {args.batch_size}, Epochs: {args.num_epochs}, mixed precision: {args.mixed_precision}"

    data_module = VQADataModule(args)
    model = TCVQAModule(args)
    if WANDB:
        logger = WandbLogger(
            project="TCVQA",
            name="tcvqa",
            save_dir="logs",
            notes=wandb_notes,
        )
    else:
        logger = CSVLogger(
            save_dir="logs",
            name="tcvqa",
            version="v1",
        )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.num_epochs,
        logger=logger,
        precision=16 if args.mixed_precision else 32,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
