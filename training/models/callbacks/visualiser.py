from pathlib import Path
from typing import Any

import cv2
import torch
from lightning import Callback, Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
from torchmetrics import Metric


class Visualizer(Callback):
    def __init__(
        self,
        res_path: Path,
        add_vis_keys: list = None,
        criterion: Metric | None = None,
        criterion_limit: float = 0.5,
    ):
        self.save_path = res_path / "visual"
        # local override is fine
        self.save_path.mkdir(exist_ok="local" in str(res_path), parents=True)

        vis_keys = [
            ((1, 2), "pred", "Pred. map"),
            ((1, 1), "pred_mask", "Pred. mask"),
            ((0, 1), "label", "GT"),
            ((0, 0), "imageA_unnorm", "Pre"),
            ((1, 0), "imageB_unnorm", "Post"),
        ]
        vis_keys.extend(add_vis_keys)

        self.vis_keys = vis_keys

        self.criterion = criterion
        self.criterion_limit = criterion_limit

    def visualize(self, batch):
        num = len(batch["img_idx"])
        for s_idx in range(num):
            img_idx = batch["img_idx"][s_idx]

            pred_map = batch["pred"][s_idx].detach().cpu()
            gt_mask = batch["label"][s_idx].detach().cpu()

            plot_current = True
            val = None
            if self.criterion is not None:
                val = self.criterion(pred_map, torch.where(gt_mask > 0.5, 1, 0))

                plot_current = val < self.criterion_limit

            if not plot_current:
                continue
            # plot only if criterion indicates a poor sample

            pred_map = pred_map.squeeze().numpy()
            di = max(p for (p, _), _, _ in self.vis_keys) + 1
            dj = max(p for (_, p), _, _ in self.vis_keys) + 1

            fig, plots = plt.subplots(di, dj, figsize=(9, 6))
            for s_plt in plots.flatten():
                s_plt.axis("off")
            fig.tight_layout()

            if val is not None:
                plots[0][2].title.set_text(f"F1: {val:.3f}")

            for pidx, key, name in self.vis_keys:
                item = batch.get(key, None)

                if item is not None:
                    item = item[s_idx].detach().cpu().squeeze().numpy()

                if key == "pred_mask":
                    plots[pidx].imshow((pred_map > 0.5), vmax=1, vmin=0, cmap="gray")
                    plots[pidx].title.set_text("Pred. mask")
                elif key == "pred":
                    plots[pidx].imshow(pred_map, vmax=1, vmin=0)
                    plots[pidx].title.set_text("Pred. map")

                    pred_maps_dir = self.save_path / "pred_map"
                    pred_maps_dir.mkdir(exist_ok=True, parents=True)

                    # save as png for lossless mask
                    cv2.imwrite(str(pred_maps_dir / f"{img_idx}.png"), pred_map * 255)
                else:
                    if len(item.shape) == 3 and item.shape[-1] > 3:
                        # assumes satlas RGB indexing
                        item = item[..., [0, 1, 2]]
                    plots[pidx].imshow(item)
                    plots[pidx].title.set_text(name)

            fig.tight_layout()
            plt.savefig(self.save_path / f"{img_idx}.jpg", bbox_inches="tight")

            plt.close("all")

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.visualize(batch=outputs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)
