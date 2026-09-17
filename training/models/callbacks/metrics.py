from typing import Any

from lightning import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from lightning.pytorch import Trainer, LightningModule


class MetricsCallback(Callback):
    def __init__(self, metrics: MetricCollection):
        self.metrics = metrics

    def setup(
        self, trainer: Trainer, module: LightningModule, stage: str | None = None
    ) -> None:
        module.metrics = self.metrics

    def reset(self):
        self.metrics.reset()

    def update(self, batch):
        self.metrics.update(batch["pred"], batch["label"])

    def calculate_and_log(self, module: LightningModule):
        module.log_dict(self.metrics, prog_bar=True)

    def on_test_epoch_start(self, trainer: Trainer, module: LightningModule) -> None:
        del trainer, module

        self.reset()

    def on_validation_epoch_start(
        self, trainer: Trainer, module: LightningModule
    ) -> None:
        del trainer, module

        self.reset()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, batch, batch_idx, dataloader_idx

        self.update(outputs)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, batch, batch_idx, dataloader_idx

        if outputs is not None:
            self.update(outputs)

    def on_test_epoch_end(self, trainer: Trainer, module: LightningModule) -> None:
        self.calculate_and_log(module)

    def on_validation_epoch_end(
        self, trainer: Trainer, module: LightningModule
    ) -> None:
        self.calculate_and_log(module)
