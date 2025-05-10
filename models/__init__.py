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
from .backbone import Backbone
# from T2VQA.model.blip_pretrain import BLIP_Pretrain
import numpy as np


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
    def __init__(self, levels=10, group=8, reg_type="swinv2"):
        super().__init__()
        self.reg_type = reg_type
        self.levels = levels
        if reg_type == "swinv2":
            self.spatial_decoder = nn.Sequential(
                nn.Conv2d(8, 32, 3, 2, 1, groups=group),
                nn.PReLU(),
                ResBlock(32),
                nn.Conv2d(32, 64, 3, 2, 1, groups=group),
                nn.PReLU(),
                ResBlock(64),
                nn.Conv2d(64, 128, 3, 2, 1, groups=group),
                nn.PReLU(),
                ResBlock(128),
                nn.Conv2d(128, 64, 3, 2, 1, groups=group),
                nn.PReLU(),
                nn.Conv2d(64, 32, 3, 1, 1, groups=group),
                nn.PReLU(),
                nn.Conv2d(32, 8, 3, 1, 1, groups=group),
            )
            self.temporal_decoder = nn.Sequential(
                nn.Conv2d(8, 32, 3, 2, 1),
                nn.BatchNorm2d(32),
                nn.PReLU(),
                ResBlock(32),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                ResBlock(64),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.PReLU(),
                ResBlock(128),
                nn.Conv2d(128, 64, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.PReLU(),
                nn.Conv2d(32, 32, 3, 1, 1),
            )
            self.f_regressor = nn.Sequential(
                nn.Flatten(start_dim=2),
                nn.Linear(4 * 48, 64),
                nn.Tanh(),
                nn.Linear(64, levels),
            )
            self.v_regressor = nn.Sequential(
                nn.AdaptiveMaxPool2d((1, 1)),
                nn.Flatten(),
                nn.BatchNorm1d(8 + 32),
                nn.Sigmoid(),
                nn.Linear(8 + 32, 16),
                nn.BatchNorm1d(16),
                nn.Sigmoid(),
                nn.Linear(16, levels),
            )
            self.frame_attention = nn.MultiheadAttention(
                embed_dim=4 * 48, num_heads=1, batch_first=True
            )
        elif reg_type == "vivit":
            self.temporal_decoder = nn.MultiheadAttention(
                embed_dim=768, num_heads=4, batch_first=True
            )
            self.v_regressor = nn.Linear(768, levels)

    def forward(self, x):
        if self.reg_type == "swinv2":
            # get score for each frame
            fs = self.spatial_decoder(x)
            b, n, h, w = fs.shape
            qkv = fs.view(b, n, h * w).contiguous()
            frame_attn = self.frame_attention(qkv, qkv, qkv)[1]
            f_level = self.f_regressor(fs)
            f_level = torch.nn.functional.softmax(f_level, dim=-1)
            f_level_agg = torch.einsum("bkm,bnk->bnm", f_level, frame_attn)

            # get score for the whole video
            ft = self.temporal_decoder(x)
            fts = torch.cat([fs, ft], dim=1)
            v_level = self.v_regressor(fts)
            v_level = torch.nn.functional.softmax(v_level, dim=-1)
            satisfaction = (
                torch.arange(1, self.levels + 1).to(f_level.device).to(f_level.dtype)
            )

            # get final score
            f_score = (f_level_agg @ satisfaction).mean(dim=-1)
            v_score = v_level @ satisfaction
            score = (f_score + v_score).unsqueeze(-1) / 2
            # x = x.mean(dim=-1)
            # x = torch.clamp(x, min=0, max=100)
            return score * (100 / self.levels)
        elif self.reg_type == "vivit":
            ft, _ = self.temporal_decoder(x, x, x)  # 4,785,768
            v_level = self.v_regressor(ft)
            v_level = torch.nn.functional.softmax(v_level, dim=-1)
            v_level = v_level.mean(dim=-2)
            satisfaction = torch.arange(1, self.levels + 1).to(ft.device).to(ft.dtype)
            v_score = v_level @ satisfaction
            v_score = v_score.unsqueeze(-1)
            return v_score * (100 / self.levels)
        elif self.reg_type == "vlm":
            pass


class DifferentialRegressor(nn.Module):
    def __init__(self, args, levels=10):
        super().__init__()
        self.args = args
        self.v1_emb = nn.Linear(768, 32)
        self.v2_emb = nn.Linear(768, 32)
        self.self_attention1 = nn.MultiheadAttention(
            embed_dim=49 * 32, num_heads=4, batch_first=True
        )
        self.self_attention2 = nn.MultiheadAttention(
            embed_dim=49 * 32, num_heads=4, batch_first=True
        )
        self.cross_attention1 = nn.MultiheadAttention(
            embed_dim=49 * 32, num_heads=4, batch_first=True
        )
        self.cross_attention2 = nn.MultiheadAttention(
            embed_dim=49 * 32, num_heads=4, batch_first=True
        )
        self.diff_regressor = nn.Sequential(
            nn.Conv1d(4 * 7, 16, 3, 1, 1),
            nn.Linear(49 * 32, 128),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, levels),
        )
        self.levels = levels

    def forward(self, v1, v2, return_attn=False):
        # v1: (b, f, c)
        # v2: (b, f, c)
        # self attention
        v1 = self.v1_emb(v1)
        v2 = self.v2_emb(v2)
        v1 = rearrange(v1, "b f p c -> b f (p c)")
        v2 = rearrange(v2, "b f p c -> b f (p c)")
        v1a, attn1 = self.self_attention1(v1, v1, v1)
        v2a, attn2 = self.self_attention2(v2, v2, v2)
        # cross attention
        cross1, attn3 = self.cross_attention1(v1, v2, v2)
        cross2, attn4 = self.cross_attention2(v2, v1, v1)
        # get the difference
        fstack = torch.cat([v1a, v2a, cross1, cross2], dim=1)
        # get the score
        diff_level = self.diff_regressor(fstack)
        diff_level = torch.nn.functional.softmax(diff_level, dim=-1)
        satisfaction = (
            torch.arange(1, self.levels + 1).to(diff_level.device).to(diff_level.dtype)
        )
        score = diff_level @ satisfaction
        if return_attn:
            return score, attn1, attn2, attn3, attn4, v1a, v2a, cross1, cross2
        return score * (100 / self.levels)


class TCVQAModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.venc = self.vision_encoder(args)
        self.q_regressor = QRegressor(reg_type=args.backbone, levels=args.levels)
        ############text video aligner#############
        # fixing the configuration of the blip
        if args.tv_align:
            # self.tv_aligner = BLIP_Pretrain(
            #     image_size=args.size,
            #     vit="large",
            #     embed_dim=256,
            #     med_config="/SSD_zfs/xxy/code/Visual-Consistency-Assessment-for-AIGC-Videos/T2VQA/med_config.json",
            # )
            self.tv_aligner = None
            state_dict = torch.load(
                "/SSD_zfs/xxy/code/Visual-Consistency-Assessment-for-AIGC-Videos/weights/model_large.pth",
                map_location="cpu",
            )
            self.tv_aligner.load_state_dict(state_dict["model"], strict=False)

            for name, param in self.tv_aligner.named_parameters():
                if "text_encoder" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.finetune_text_proj = nn.Linear(
                self.tv_aligner.text_encoder.config.hidden_size, 256
            )
        ######################################
        if args.flow:
            self.flow_regressor = DifferentialRegressor(args, levels=args.levels)

        
        if not args.ada_voter:
            num_voters = 1
            if args.flow:
                num_voters += 1
            if args.tv_align:
                num_voters += 1
            if args.naturalness:
                num_voters += 1
            if args.low_level:
                num_voters += 1
            self.voters = nn.Parameter(
                torch.ones(num_voters, dtype=torch.float32).to(self.device) / num_voters,
                requires_grad=True,
            )

        self.val_pred = []
        self.val_gt = []
        self.train_pred = []
        self.train_gt = []

        self.test_pred = []
        self.test_gt = []
        self.save_hyperparameters(args)

    def vision_encoder(self, args):
        vision_encoder = Backbone(backbone=self.args.backbone, args=args)
        vision_encoder.train()
        return vision_encoder

    def forward(self, x, caption=None, prompt=None, flow=None, return_feature=False, return_attn=False):

        # rearrange input tensor fron (b,c,f,h,w) -> (bf,c,h,w)
        b, c, f, h, w = x.shape
        # rearange x for different backbones
        if self.args.backbone == "swinv2":
            x = rearrange(x, "b c f h w -> (b f) c h w")
            out = self.venc(pixel_values=x)
            v_embed = out.last_hidden_state
            v_embed = rearrange(v_embed, "(b f) l c -> b f l c", b=b, f=f)
        elif self.args.backbone == "vivit":
            x = rearrange(x, "b f c h w -> b c f h w")
            out = self.venc(pixel_values=x)
            v_embed = out.last_hidden_state
        q_score = self.q_regressor(v_embed)
        ## TODO: add flow features
        if self.args.flow:
            v_embed_batch1 = v_embed[:, :-1, ...]
            v_embed_batch2 = v_embed[:, 1:, ...]
            h = np.sqrt(v_embed_batch1.shape[-2]).astype(int)
            # downsample the flow into shape of h,w
            b, f, _, ho, wo = flow.shape
            flow = rearrange(flow, "b f c h w -> (b f) c h w", b=b, f=f)
            flow = torch.nn.functional.interpolate(
                flow, size=(h, h), mode="bilinear", align_corners=False
            )
            # flow = rearrange(flow, "(b f) c h w -> b f c h w", b=b, f=f)
            # warp the v_embed_batch1 with the flow
            ##########################################
            grid_x, grid_y = torch.meshgrid(
                torch.arange(h), torch.arange(h), indexing="ij"
            )
            base_grid = (
                torch.stack((grid_x, grid_y), dim=0)
                .float()
                .unsqueeze(0)
                .to(flow.device)
            )
            coords = base_grid + flow / (ho / h)
            coords[:, 0, ...] = 2 * (coords[:, 0, ...] / (h - 1)) - 1  # Normalize x
            coords[:, 1, ...] = 2 * (coords[:, 1, ...] / (h - 1)) - 1  # Normalize y
            coords = coords.permute(0, 2, 3, 1)
            v_embed_batch1_reshape = rearrange(
                v_embed_batch1, "b c (h w) l-> (b c) h w l", h=h, w=h
            )
            v_embed_batch1_reshape = v_embed_batch1_reshape.permute(0, 3, 1, 2)
            v_embed_batch1_warp = torch.nn.functional.grid_sample(
                v_embed_batch1_reshape,
                coords,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            v_embed_batch1_warp = v_embed_batch1_warp.permute(0, 2, 3, 1)
            v_embed_batch1_warp = rearrange(
                v_embed_batch1_warp, "(b c) h w l -> b c (h w) l", b=b, h=h, w=h
            )
            ##########################################
            flow_score = self.flow_regressor(v_embed_batch1_warp, v_embed_batch2, return_attn=return_attn)
            if return_attn:
                flow_score, attn1, attn2, attn3, attn4, v1a, v2a, cross1, cross2 = flow_score
            q_score_cat = torch.cat(
                [q_score, flow_score.squeeze().unsqueeze(-1)], dim=-1
            )

        ## TODO: add text video aligner
        if self.args.tv_align:
            pass
        # text = self.tv_aligner.tokenizer(
        #     caption,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=35,
        #     return_tensors="pt",
        # ).to(x.device)
        # f_align = []
        # for j in range(x.size(2)):
        #     image = x[:, :, j, :, :]

        #     image_embeds = self.tv_aligner.visual_encoder(image)

        #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        #         x.device
        #     )
        #     output = self.tv_aligner.text_encoder(
        #         text.input_ids,
        #         attention_mask=text.attention_mask,
        #         encoder_hidden_states=image_embeds,
        #         encoder_attention_mask=image_atts,
        #         return_dict=True,
        #     )

        #     output = self.finetune_text_proj(output.last_hidden_state[:, 0, :])

        #     f_align.append(output)
        # f_align = torch.stack(f_align, dim=1)
        ##################################
        q_score_out = (nn.functional.softmax(self.voters, dim=-1) * q_score_cat).sum(
            dim=-1, keepdim=True
        )
        if return_feature:
            return q_score_out, v_embed
        if return_attn:
            if not self.args.flow:
                raise 'attention is only available for flow regressor'
            return (flow_score, attn1, attn2, attn3, attn4, v1a, v2a, cross1, cross2)
        return q_score_out

    def training_step(self, batch, batch_idx):
        x = batch["video"]
        flow = batch.get("flow", None)
        q_score = self(x, flow=flow)
        y = batch["gt_label"]
        p_loss = plcc_loss(q_score, y)
        mse_loss = torch.nn.functional.mse_loss(q_score, y)
        r_loss = rank_loss(q_score, y)
        loss = p_loss + 0.3 * r_loss + 0 * mse_loss
        self.train_pred.append(q_score)
        self.train_gt.append(y)

        train_loss_dict = {
            "train_loss": loss,
            "train_mse_loss": mse_loss,
            "train_plcc_loss": p_loss,
            "train_rank_loss": r_loss,
        }
        self.log_dict(train_loss_dict, prog_bar=True)
        self.log(
            "learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self):
        q_scores = torch.cat(self.train_pred)
        gt_labels = torch.cat(self.train_gt)
        plcc = pearsonr(
            q_scores.detach().cpu().numpy(), gt_labels.detach().cpu().numpy()
        )[0]
        srcc = spearmanr(
            q_scores.detach().cpu().numpy(), gt_labels.detach().cpu().numpy()
        )[0]

        q_levels = [0, 20, 40, 60, 80, 100]
        plcc_groups = []
        srcc_groups = []
        for ql in range(len(q_levels) - 1):
            q_ind = torch.where(
                (gt_labels >= q_levels[ql]) & (gt_labels < q_levels[ql + 1])
            )[0]
            if len(q_ind) > 1:
                plcc_l = pearsonr(
                    q_scores[q_ind].detach().cpu().numpy(),
                    gt_labels[q_ind].detach().cpu().numpy(),
                )[0]
                srcc_l = spearmanr(
                    q_scores[q_ind].detach().cpu().numpy(),
                    gt_labels[q_ind].detach().cpu().numpy(),
                )[0]
                plcc_groups.append(plcc_l[0])
                srcc_groups.append(srcc_l)
            else:
                plcc_groups.append(0)
                srcc_groups.append(0)

        train_loss_dict = {
            "train_plcc_l1": plcc_groups[0],
            "train_plcc_l2": plcc_groups[1],
            "train_plcc_l3": plcc_groups[2],
            "train_plcc_l4": plcc_groups[3],
            "train_plcc_l5": plcc_groups[4],
            "train_srcc_l1": srcc_groups[0],
            "train_srcc_l2": srcc_groups[1],
            "train_srcc_l3": srcc_groups[2],
            "train_srcc_l4": srcc_groups[3],
            "train_srcc_l5": srcc_groups[4],
            "train_plcc_all": plcc[0],
            "train_srcc_all": srcc,
        }
        self.log_dict(train_loss_dict, prog_bar=False)

        self.train_pred.clear()
        self.train_gt.clear()

    def validation_step(self, batch, batch_idx):
        x = batch["video"]
        flow = batch.get("flow", None)
        q_score = self(x, flow=flow)
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

        q_levels = [0, 20, 40, 60, 80, 100]
        plcc_groups = []
        srcc_groups = []
        for ql in range(len(q_levels) - 1):
            q_ind = torch.where(
                (gt_labels >= q_levels[ql]) & (gt_labels < q_levels[ql + 1])
            )[0]
            if len(q_ind) > 1:
                plcc_l = pearsonr(
                    q_scores[q_ind].cpu().numpy(),
                    gt_labels[q_ind].cpu().numpy(),
                )[0]
                srcc_l = spearmanr(
                    q_scores[q_ind].cpu().numpy(),
                    gt_labels[q_ind].cpu().numpy(),
                )[0]
                plcc_groups.append(plcc_l[0])
                srcc_groups.append(srcc_l)
            else:
                plcc_groups.append(0)
                srcc_groups.append(0)

        val_loss_dict = {
            "val_mse_loss": mse_loss,
            "val_plcc_l1": plcc_groups[0],
            "val_plcc_l2": plcc_groups[1],
            "val_plcc_l3": plcc_groups[2],
            "val_plcc_l4": plcc_groups[3],
            "val_plcc_l5": plcc_groups[4],
            "val_srcc_l1": srcc_groups[0],
            "val_srcc_l2": srcc_groups[1],
            "val_srcc_l3": srcc_groups[2],
            "val_srcc_l4": srcc_groups[3],
            "val_srcc_l5": srcc_groups[4],
            "val_plcc_all": plcc[0],
            "val_srcc_all": srcc,
        }

        self.log_dict(val_loss_dict, prog_bar=False)
        self.val_pred.clear()
        self.val_gt.clear()

    def test_step(self, batch, batch_idx):
        x = batch["video"]
        filename = batch["filename"]
        flow = batch.get("flow", None)
        q_score = self(x, flow=flow)
        y = batch["gt_label"]
        self.test_pred.append((filename, q_score))
        self.test_gt.append((filename, y))

    def on_test_epoch_end(self):
        torch.save(
            self.test_pred,
            "test_pred.pt",
        )
        torch.save(
            self.test_gt,
            "test_gt.pt",
        )
        self.test_pred.clear()
        self.test_gt.clear()

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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or 'step' if you want per-batch scheduling
                "frequency": 1,
            },
        }

    def get_hard_samples(self, loader):
        self.venc.eval()
        self.tv_aligner.eval()
        self.q_regressor.eval()
        q_score_all = []
        gt_label_all = []
        filename_all = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # if batch_idx >100:
                #     break
                print(batch_idx)
                x = batch["video"].to(self.device)
                q_score = self(x)
                y = batch["gt_label"].to(self.device)
                q_score_all.append(q_score)
                gt_label_all.append(y)
                filename_all.extend(batch["filename"])
        q_score_all = torch.cat(q_score_all)
        q_score_all = (
            100
            * (q_score_all - q_score_all.min())
            / (q_score_all.max() - q_score_all.min())
        )
        gt_label_all = torch.cat(gt_label_all)
        # filename_all = torch.cat(filename_all)
        diff = torch.abs(q_score_all - gt_label_all)
        hard_ind = torch.where(diff > 5)[0]
        print(hard_ind.shape)
        hard_samples = []
        for hi in hard_ind:
            hard_samples.append(filename_all[hi])
        import numpy as np

        np.save(
            "hard_samples.npy",
            np.array(hard_samples),
            allow_pickle=True,
        )
