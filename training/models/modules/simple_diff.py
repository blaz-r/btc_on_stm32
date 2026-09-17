import torch.nn as nn
import torch

from models.modules.base import BaseCDDiff


def split_pair_batch(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    is_onnx_export = getattr(torch.onnx, "is_in_onnx_export", lambda: False)
    if (torch.jit.is_tracing() or is_onnx_export()) and feat.shape[0] == 2:
        return feat[:1], feat[1:2]
    return feat.chunk(2, dim=0)


class SubAbsDiff(BaseCDDiff):
    """
    Simple difference module by subtraction and difference.

    """

    name = "subabs"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feat_dict):
        if self.pretraining and not self.use_in_pretrain:
            return feat_dict

        for i, feat in feat_dict.items():
            f1, f2 = split_pair_batch(feat)
            feat_dict[i] = torch.abs(f1 - f2)

        return feat_dict

    def forward_pair(self, feat_a: dict, feat_b: dict) -> dict:
        return {i: torch.abs(feat - feat_b[i]) for i, feat in feat_a.items()}

    def get_parameters(self, pretraining=False):
        return []


class LayerNormPermute(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class SubDiff(BaseCDDiff):
    """
    Simple difference module by subtraction.

    """

    name = "sub"

    def __init__(self, norm: bool = False, dims: list[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if norm:
            self.name += "LN"

        if self.pretraining and not self.use_in_pretrain:
            print("SubDiff won't be used during pretrain")
            return

        self.norms = nn.ModuleList()
        for dim in dims:
            if norm:
                self.norms.append(LayerNormPermute(dim))
            else:
                self.norms.append(nn.Identity())

    def forward(self, feat_dict):
        # skip in pretrain
        if self.pretraining and not self.use_in_pretrain:
            return feat_dict

        out_dict = {}
        for norm, (i, feat) in zip(self.norms, feat_dict.items()):
            f1, f2 = split_pair_batch(feat)
            out_dict[i] = norm(f1 - f2)

        return out_dict

    def forward_pair(self, feat_a: dict, feat_b: dict) -> dict:
        out_dict = {}
        for norm, (i, feat) in zip(self.norms, feat_a.items()):
            out_dict[i] = norm(feat - feat_b[i])
        return out_dict

    def get_parameters(self, pretraining=False):
        return [{"params": self.parameters()}]


class AvgDiff(BaseCDDiff):
    """
    Simple difference module by averaging.

    """

    name = "avg"

    def __init__(self, norm: bool = False, dims: list[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if norm:
            self.name += "LN"

        if self.pretraining and not self.use_in_pretrain:
            print("SubDiff won't be used during pretrain")
            return

        self.norms = nn.ModuleList()
        for dim in dims:
            if norm:
                self.norms.append(LayerNormPermute(dim))
            else:
                self.norms.append(nn.Identity())

    def forward(self, feat_dict):
        # skip in pretrain
        if self.pretraining and not self.use_in_pretrain:
            return feat_dict

        out_dict = {}
        for norm, (i, feat) in zip(self.norms, feat_dict.items()):
            f1, f2 = split_pair_batch(feat)
            out_dict[i] = norm((f1 + f2) / 2)

        return out_dict

    def forward_pair(self, feat_a: dict, feat_b: dict) -> dict:
        out_dict = {}
        for norm, (i, feat) in zip(self.norms, feat_a.items()):
            out_dict[i] = norm((feat + feat_b[i]) / 2)
        return out_dict

    def get_parameters(self, pretraining=False):
        return [{"params": self.parameters()}]


class NoDiff(BaseCDDiff):
    """
    No differencing.

    """

    name = "noDiff"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, feat_dict):
        return feat_dict

    def forward_pair(self, feat_a: dict, feat_b: dict) -> dict:
        return feat_a

    def get_parameters(self, pretraining: bool) -> list[dict]:
        return []


class CatDiff(BaseCDDiff):
    """
    Simple difference module by concatenation.

    """

    name = "cat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.pretraining and not self.use_in_pretrain:
            print("CatDiff won't be used during pretrain")
            return

    def forward(self, feat_dict):
        out_dict = {}
        for i, feat in feat_dict.items():
            f1, f2 = split_pair_batch(feat)
            # channel cat
            out_dict[i] = torch.cat((f1, f2), dim=1)

        return out_dict

    def forward_pair(self, feat_a: dict, feat_b: dict) -> dict:
        return {i: torch.cat((feat, feat_b[i]), dim=1) for i, feat in feat_a.items()}

    def adjust_dims(self, feat_dims: list[int]) -> list[int]:
        """
        Adjust output dims for concat style - by doubling.

        Args:
            feat_dims: list of input feature dims

        Returns:
            list of updated feature dims based on diff style
        """
        return [d * 2 for d in feat_dims]

    def get_parameters(self, pretraining=False) -> list[dict]:
        return []
