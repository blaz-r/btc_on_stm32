import torch
from torch import nn


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == "exp":
        return vmin + x.exp().clip(max=vmax - vmin)
    if mode == "sigmoid":
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f"bad {mode=}")


class ConfLoss(nn.Module):
    def __init__(self, base_loss, alpha=0.5, mode=("exp", 1, float("inf"))):
        super().__init__()
        self.base_l = base_loss
        self.mode = mode
        self.alpha = alpha

    def get_conf_log(self, x):
        return x, torch.log(x)

    def forward(self, pred, conf, target):
        loss = self.base_l(pred, target)
        conf = reg_dense_conf(conf, self.mode)
        conf, log_conf = self.get_conf_log(conf)
        conf_loss = loss * conf - self.alpha * log_conf
        # nan business
        conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else torch.tensor(0)
        return conf_loss
