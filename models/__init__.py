"""Module for TCVQA"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import Swinv2Model
from einops import rearrange
from .utils import plcc_loss, rank_loss
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import wandb


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out + x


class QRegressor(nn.Module):
    def __init__(self, levels=10):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(),
            ResBlock(16),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            ResBlock(32),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 8, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(8),
            nn.Sigmoid(),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),
            nn.Linear(16, levels),
        )
        self.levels = levels

    def forward(self, x):
        level = self.decoder(x)
        q_level = torch.nn.functional.softmax(level, dim=-1)
        satisfaction = (
            torch.arange(1, self.levels + 1).to(q_level.device).to(q_level.dtype)
        )
        q_score = q_level @ satisfaction
        q_score = q_score.unsqueeze(-1)
        # x = x.mean(dim=-1)
        # x = torch.clamp(x, min=0, max=100)
        return q_score * (100 / self.levels)


class TCVQAModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.venc = self.vision_encoder()
        self.venc.eval()
        # freeze the vision encoder
        # for p in self.venc.parameters():
        #     p.requires_grad = False
        self.q_regressor = QRegressor()
        self.val_pred = []
        self.val_gt = []

    def vision_encoder(self):
        vision_encoder = Swinv2Model.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256"
        )
        vision_encoder.train()
        return vision_encoder

    def forward(self, x):

        # rearrange input tensor fron (b,c,f,h,w) -> (bf,c,h,w)
        b, c, f, h, w = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")
        out = self.venc(pixel_values=x)
        v_embed = out.last_hidden_state
        v_embed = rearrange(v_embed, "(b f) l c -> b f l c", b=b, f=f)
        q_score = self.q_regressor(v_embed)
        return q_score

    def training_step(self, batch, batch_idx):
        x = batch["video"]
        q_score = self(x)
        y = batch["gt_label"]
        p_loss = plcc_loss(q_score, y)
        mse_loss = torch.nn.functional.mse_loss(q_score, y)
        r_loss = rank_loss(q_score, y)
        loss = p_loss + 0.3 * r_loss + 1e-4 * mse_loss

        train_loss_dict = {
            "train_loss": loss,
            "train_mse_loss": mse_loss,
            "train_plcc_loss": p_loss,
            "train_rank_loss": r_loss,
        }
        self.log_dict(train_loss_dict, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["video"]
        q_score = self(x)
        y = batch["gt_label"]
        self.val_pred.append(q_score)
        self.val_gt.append(y)

    def on_validation_epoch_end(self):
        q_scores = torch.cat(self.val_pred)
        gt_labels = torch.cat(self.val_gt)

        mse_loss = torch.nn.functional.mse_loss(q_scores, gt_labels)
        plcc = pearsonr(q_scores.cpu().numpy(), gt_labels.cpu().numpy())[0]
        srcc = spearmanr(q_scores.cpu().numpy(), gt_labels.cpu().numpy())[0]

        if self.args.wandb:
            # Create a Matplotlib figure
            fig, ax = plt.subplots()
            ax.scatter(q_scores.cpu().numpy(), gt_labels.cpu().numpy(), alpha=0.5)
            ax.set_xlabel("Predicted Quality Score")
            ax.set_ylabel("Ground Truth Quality Score")

            # Log the plot to WandB
            self.logger.experiment.log({"Quality Score": wandb.Image(fig)})
            plt.close(fig)

        val_loss_dict = {
            "val_mse_loss": mse_loss,
            "val_pcc": plcc[0],
            "val_scc": srcc,
        }
        self.log_dict(val_loss_dict, prog_bar=True)
        self.val_pred.clear()
        self.val_gt.clear()

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.num_epochs,
            eta_min=0,
        )
        return [optimizer], [scheduler]
