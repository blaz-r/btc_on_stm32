import torch
from torch import nn

from models.modules.base import BaseCDEncoder


class DinoV2Encoder(BaseCDEncoder):
    name = "dinov2"

    def __init__(
        self,
        return_layers: list[int],
        model="dinov2_vits14",
        finetune=False,
        *args,
        **kwargs,
    ):
        super().__init__(return_layers=return_layers, *args, **kwargs)

        self.finetune = finetune

        model = torch.hub.load("facebookresearch/dinov2", model)
        if not finetune:
            for param in model.parameters():
                param.requires_grad = False
        self.model = model

        self.return_indices = [ln - 1 for ln in self.return_layers]

    @torch.no_grad()
    def no_grad_forward(self, x):
        self.model.eval()
        return self.model.get_intermediate_layers(
            x, n=self.return_indices, reshape=True
        )

    def grad_forward(self, x):
        return self.model.get_intermediate_layers(
            x, n=self.return_indices, reshape=True
        )

    def forward(self, x):
        if self.finetune:
            feats = self.grad_forward(x)
        else:
            feats = self.no_grad_forward(x)
        return {layer: feat for layer, feat in zip(self.return_layers, feats)}

    def get_out_dims(self):
        features = self.forward(torch.rand(1, 3, 224, 224))
        return [f.shape[1] for f in features.values()]

    def get_strides(self):
        features = self.forward(torch.rand(1, 3, 224, 224))
        return [256 / f.shape[2] for f in features.values()]

    def get_parameters(self, pretraining: bool) -> list[dict]:
        if self.finetune:
            return [{"params": self.model.parameters(), **self.get_lr_dict()}]
        else:
            return []


def build_dinov2(config):
    return DinoV2Encoder(
        **config.encoder.init_args.as_dict(),
    )
