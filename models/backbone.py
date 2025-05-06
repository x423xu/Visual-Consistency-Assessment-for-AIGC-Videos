import torch
from transformers import Swinv2Model, AutoModel, AutoConfig


class Backbone(torch.nn.Module):
    def __init__(self, backbone="vivit", args=None):
        super(Backbone, self).__init__()
        if backbone == "vivit":
            config = AutoConfig.from_pretrained("google/vivit-b-16x2-kinetics400")

            # Set number of frames
            config.num_frames = args.clip_len

            # ViViT patch and temporal settings
            patch_size = 16
            img_size = config.image_size  # should be 224
            temporal_downsample = 2  # because vivit-b-16x2

            # Calculate expected number of tokens
            num_patches_per_frame = (img_size // patch_size) ** 2  # 14 x 14 = 196
            num_tokens = (
                num_patches_per_frame * config.num_frames
            ) // temporal_downsample + 1  # +1 for CLS

            # Set positional embedding length
            config.num_positions = num_tokens

            # Initialize the model from scratch (or load weights and interpolate later)
            model = AutoModel.from_config(config)
        elif backbone == "swinv2":
            model = Swinv2Model.from_pretrained(
                "microsoft/swinv2-tiny-patch4-window8-256"
            )

        self.backbone = model

    def __call__(self, pixel_values=None):
        out = self.backbone(pixel_values=pixel_values)
        return out
