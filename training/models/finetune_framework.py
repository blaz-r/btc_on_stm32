import copy
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection

from torchvision.ops.focal_loss import sigmoid_focal_loss

from models.framework import Framework
from models.loss.conf_loss import ConfLoss, reg_dense_conf
from models.loss.dice import dice_loss_smooth, dice_loss_eps
from models.loss.l1 import BinaryTruncatedL1
from models.modules.base import BaseCDModule


class FinetuneFramework(Framework):
    pretraining = False

    def __init__(self, metrics: MetricCollection, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            if hasattr(self.enc.backbone.embeddings, "mask_token"):
                self.enc.backbone.embeddings.mask_token = None
        except:
            pass

        self.use_conf = self.config.train.use_conf
        if self.use_conf:
            assert self.dec.out_channels == 2

        self.val_metrics = metrics
        self.test_metrics = copy.deepcopy(metrics)

        self.val_metrics.prefix = "val/"
        self.test_metrics.prefix = "test/"

    def generate_and_verify_res_path(self, save_path):
        if self.res_path is None:
            config = self.config
            # omit dataset name and config tag from name, rather have them in path
            res_run_name = f"{config.tag}_[{self.enc.name}_{self.diff.name}_{self.dec.name}_{self.loss_name}]"

            if self.pre_diff is not None:
                res_run_name += f"({self.pre_diff.name})"
            if self.out_proc is not None:
                res_run_name += f"({self.out_proc.name})"

            res_path = (
                Path(save_path) / config.config_tag / res_run_name / config.data.dataset
            )
            # raise exception if exists in non dev mode
            res_path.mkdir(parents=True, exist_ok=True)

            self.res_path = res_path

        return self.res_path

    def dump_config(self):
        with (self.res_path / "ft_config.yaml").open("w", encoding="utf-8") as f:
            yaml.dump(self.config, f)

    def build_loss(self):
        loss_list = self.config.train.loss

        loss_fn_dict = {}
        name = ""

        for i, l in enumerate(loss_list):
            name += l
            if i < len(loss_list) - 1:
                name += "-"

            if l == "focal":
                loss_fn_dict[l] = partial(
                    sigmoid_focal_loss,
                    reduction="none" if self.config.train.use_conf else "mean",
                    alpha=-1,
                )
            elif l == "tl1":
                loss_fn_dict[l] = BinaryTruncatedL1(force_negative_t=True)
            elif l == "tl1nFN":
                loss_fn_dict[l] = BinaryTruncatedL1(force_negative_t=False)
            elif l == "ce":
                loss_fn_dict[l] = torch.nn.BCEWithLogitsLoss(
                    reduction="none" if self.config.train.use_conf else "mean"
                )
            elif l == "2chce":
                f = torch.nn.CrossEntropyLoss()

                def two_ch_ce(pred, t):
                    return f(pred, t.squeeze().long())

                loss_fn_dict[l] = two_ch_ce
            elif l == "dice":
                loss_fn_dict[l] = dice_loss_smooth
            elif l == "diceE":
                loss_fn_dict[l] = dice_loss_eps
            else:
                raise ValueError(f"Unknown loss function {l}")

        if self.config.train.use_conf:
            loss_fn_dict = {f"conf_{n}": ConfLoss(v) for n, v in loss_fn_dict.items()}

        def loss_callable(preds, target):
            total_loss = 0
            loss_terms = {}
            for name, curr_loss_f in loss_fn_dict.items():
                if self.use_conf:
                    change = preds[:, 0:1]
                    conf = preds[:, 1:2]
                    curr_loss_val = curr_loss_f(pred=change, target=target, conf=conf)
                else:
                    curr_loss_val = curr_loss_f(preds, target)

                loss_terms[name] = curr_loss_val.item()
                total_loss += curr_loss_val
            return total_loss, loss_terms

        return loss_callable, name

    def load_from_pretrained(self, pretrain_framework: Framework):
        for name in ["in_proc", "enc", "pre_diff", "diff", "dec", "out_proc"]:
            pt_module = getattr(pretrain_framework, name, None)
            if pt_module is None:
                continue
            if not pt_module.transfer:
                # not to be transferred to finetune phase
                print(f"!!! Skipping {name} - transfer flag set to false.")
                continue

            curr_module: BaseCDModule = getattr(self, name, None)
            if pt_module.__class__ != curr_module.__class__:
                print(f"!!! Skipping {name} - {pt_module.__class__} does not match.")
                continue

            if curr_module is not None:
                print(f"Loading pretrained weights for {name}")
                pretrained_state_dict = pt_module.state_dict()
                curr_module.transfer_from_pretrained(pretrained_state_dict)
            else:
                print(f"Skipping {name} - not present in finetune framework.")

    @staticmethod
    def batch_to_tensor(batch):
        return torch.cat([batch["imageA"], batch["imageB"]], dim=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs

        lbl = batch["label"]

        x = self.forward(self.batch_to_tensor(batch))

        try:
            loss, loss_terms = self.criterion(x, lbl)
        except:
            return None

        self.log("loss", loss, prog_bar=True, logger=True)
        self.log_dict(loss_terms, prog_bar=False, logger=True)

        return {"loss": loss}

    def predict_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        del args, kwargs

        x = self.forward(self.batch_to_tensor(batch))

        if self.use_conf:
            change = torch.sigmoid(x[:, 0:1])

            mode = ("exp", 1, float("inf"))
            conf = x[:, 1:2]
            conf = reg_dense_conf(conf, mode)

            conf = (conf - conf.min()) / (conf.max() - conf.min())

            return {"pred": change, "conf": conf}

        if len(x.shape) == 4 and x.shape[1] == 2:
            pred = torch.softmax(x, dim=1)[:, 1].unsqueeze(1)
            return pred
        else:
            return torch.sigmoid(x)  # .argmax(dim=1, keepdim=True)

    def test_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        preds = self.predict_step(batch, *args, **kwargs)
        if isinstance(preds, dict):
            batch = {**batch, **preds}
        else:
            batch["pred"] = preds

        if preds.shape == batch["label"].shape:
            self.test_metrics.update(batch["pred"], torch.where(batch["label"] > 0.5, 1, 0))
        self.log_dict(self.test_metrics, on_epoch=True, prog_bar=True)

        return batch

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        try:
            preds = self.predict_step(batch, *args, **kwargs)
            if isinstance(preds, dict):
                batch = {**batch, **preds}
            else:
                batch["pred"] = preds

            results = batch

            self.val_metrics.update(
                batch["pred"], torch.where(batch["label"] > 0.5, 1, 0)
            )
            self.log_dict(self.val_metrics, on_epoch=True, prog_bar=True)
        except Exception as e:
            # catch only in val, test should always pass
            print(
                f"Error in validation {e}, validation metrics will now be unreliable, rely on test metrics."
            )
            results = None

        return results
