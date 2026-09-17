from pathlib import Path
from typing import Any, Tuple

import cv2
import torch
from torchvision.transforms.v2 import Normalize
from torchvision.transforms.functional import resize
from lightning import Callback, Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt


class MIMVisualizer(Callback):
    def __init__(self, res_path: Path, mask_recon: bool = True, do_unnorm: bool = True):
        if do_unnorm:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            self.denorm = Normalize(
                mean=[-m / s for m, s in zip(mean, std)], std=[1.0 / s for s in std]
            )
        else:
            self.denorm = None
        self.mask_recon = mask_recon
        self.save_path = res_path / "visual" / "recon"
        # TODO - make existok - false
        self.save_path.mkdir(exist_ok=True, parents=True)

    def visualize(self, batch):
        for idx, imageA, imageB, Are, Bre, Amask, Bmask, d_name in zip(
            batch["img_idx"],
            batch["imageA_unnorm"].cpu(),
            batch["imageB_unnorm"].cpu(),
            batch["i1_recon"].detach().cpu(),
            batch["i2_recon"].detach().cpu(),
            batch["i1_mask"].detach().cpu(),
            batch["i2_mask"].detach().cpu(),
            batch["d_name"],
        ):
            # plot only if criterion indicates a poor sample
            fig, plots = plt.subplots(2, 3, figsize=(9, 6))
            for s_plt in plots.flatten():
                s_plt.axis("off")

            fig.tight_layout()

            plots[0][0].imshow(imageA)
            plots[0][0].title.set_text("Pre-Image")

            plots[1][0].imshow(imageB)
            plots[1][0].title.set_text("Post-Image")

            Amask = (
                Amask.repeat_interleave(4, 0)
                .repeat_interleave(4, 1)
                .unsqueeze(0)
                .contiguous()
                .to(Are.device)
            )
            Bmask = (
                Bmask.repeat_interleave(4, 0)
                .repeat_interleave(4, 1)
                .unsqueeze(0)
                .contiguous()
                .to(Bre.device)
            )

            if self.denorm:
                Are = self.denorm(Are)
                Bre = self.denorm(Bre)
            if self.mask_recon:
                imageA = imageA.permute(2, 0, 1)
                imageB = imageB.permute(2, 0, 1)
                if imageA.shape[:2] != Are.shape[1:]:
                    imageA = resize(imageA, (Are.shape[1], Are.shape[2]))
                    imageB = resize(imageB, (Bre.shape[1], Bre.shape[2]))
                Are = imageA / 255 * (1 - Amask) + Amask * Are
                Bre = imageB / 255 * (1 - Bmask) + Bmask * Bre

                Aout = Are * Amask
                Bout = Bre * Bmask
            else:
                Aout = Are
                Bout = Bre

            Aout = torch.clip(Aout, 0, 1).permute(1, 2, 0).numpy()
            Bout = torch.clip(Bout, 0, 1).permute(1, 2, 0).numpy()

            Are = torch.clip(Are, 0, 1).permute(1, 2, 0).numpy()
            Bre = torch.clip(Bre, 0, 1).permute(1, 2, 0).numpy()

            plots[0][1].imshow(Are)
            plots[0][1].title.set_text("Pre - Recon")

            plots[1][1].imshow(Bre)
            plots[1][1].title.set_text("Post- Recon")

            plots[0][2].imshow(Amask.squeeze().numpy(), vmax=1, vmin=0)
            plots[0][2].title.set_text("mask")
            plots[1][2].imshow(Bmask.squeeze().numpy(), vmax=1, vmin=0)
            plots[1][2].title.set_text("mask")

            fig.tight_layout()

            plt.savefig(self.save_path / f"{idx}.jpg", bbox_inches="tight")

            # visual / recon / dataset name / img
            recon_dir = self.save_path / d_name
            recon_dir.mkdir(exist_ok=True, parents=True)

            # save as png for lossless mask
            cv2.imwrite(str(recon_dir / f"{idx}.png"), Aout * 255)
            cv2.imwrite(str(recon_dir / f"{idx}.png"), Bout * 255)

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
        self.visualize(batch=batch)

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
