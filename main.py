import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from options import parser
import pytorch_lightning as pl
from dataset import VQADataModule
from models import TCVQAModule
from pytorch_lightning.loggers import WandbLogger, CSVLogger


def main():
    args = parser.parse_args()
    
    # wandb notes
    wandb_notes = f"Learning rate: {args.lr}, Batch size: {args.batch_size}, Epochs: {args.num_epochs}, mixed precision: {args.mixed_precision}"

    data_module = VQADataModule(args)
    if args.hard_train:
        model = TCVQAModule.load_from_checkpoint(
            checkpoint_path='/SSD_zfs/xxy/code/Visual-Consistency-Assessment-for-AIGC-Videos/logs/TCVQA/hy89tzgf/checkpoints/epoch=99-step=190900.ckpt',
            args=args,
        )
    else:
        model = TCVQAModule(args)
    if args.wandb:
        logger = WandbLogger(
            project="TCVQA",
            name="tcvqa",
            save_dir="logs",
            notes=wandb_notes,
        )
        logger.watch(model, log_graph=True, log=None)
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
    if not args.eval:  
        trainer.fit(model, data_module)
    else:
        model = TCVQAModule.load_from_checkpoint(
            checkpoint_path='/SSD_zfs/xxy/code/Visual-Consistency-Assessment-for-AIGC-Videos/logs/TCVQA/hy89tzgf/checkpoints/epoch=99-step=190900.ckpt',
            args=args,
        )
        # trainer.test(model, data_module)
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        with open('train.lst', 'w') as f:
            for batch in train_loader:
                filename = batch['filename']
                for fn in filename:
                    f.write(fn + '\n')
        with open('val.lst', 'w') as f:
            for batch in val_loader:
                filename = batch['filename']
                for fn in filename:
                    f.write(fn + '\n')
            
        # model.get_hard_samples(data_module.train_dataloader())


if __name__ == "__main__":
    main()
