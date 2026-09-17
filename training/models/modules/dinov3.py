import einops
import torch
from torch import nn
from transformers import AutoModel


class Dinov3Backbone(nn.Module):
    def __init__(self, model_name, hf_config, return_layers):
        super().__init__()
        model = AutoModel.from_pretrained(model_name, config=hf_config)
        self.return_layers = return_layers
        self.is_vit = "vit" in model_name
        self.model = model
        self.feature_maps = []

        self.channels = self.dryrun_channels()

    def dryrun_channels(self):
        x = torch.rand(1, 3, 256, 256)
        self.model.eval()
        with torch.no_grad():
            self(x)
        return [f.shape[1] for f in self.feature_maps]

    def vit_forward(self, x):
        feats = self.model(x, output_hidden_states=True).hidden_states
        out = []
        for f in feats:
            f = f[:, 5:]  # remove cls and registers in vit
            b, l, c = f.shape
            h = int(l**0.5)
            feat = einops.rearrange(f, "b (h w) c -> b c h w", h=h, w=h)
            out.append(feat)
        return out

    def convnext_forward(self, x):
        feats = self.model(x, output_hidden_states=True).hidden_states
        return list(feats)

    def __call__(self, x, output_hidden_states=True):
        if self.is_vit:
            feat = self.vit_forward(x)
        else:
            feat = self.convnext_forward(x)
        self.feature_maps = [feat[i] for i in self.return_layers]
        return self
