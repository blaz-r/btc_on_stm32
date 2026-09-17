import os
import pathlib
import shutil
from pathlib import Path

import pandas as pd
import torch

import wandb
from lightning import Trainer, seed_everything
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryRecall,
    BinaryPrecision,
    BinaryJaccardIndex,
    MulticlassJaccardIndex,
)

from configs.config_parser import get_parser
from data.datamodule import CDDataModule
from models.callbacks.visualiser import Visualizer
from models.common.mim_visualiser import MIMVisualizer

from models.finetune_framework import FinetuneFramework


WANDB = os.environ.get("WANDB_MODE", "disabled") != "disabled"


def finetune(framework, config, data_path, wandb_proj):
    run_name = framework.generate_exp_name()
    print(f"Starting finetune for {run_name}")

    datamodule = CDDataModule(
        config,
        data_path=data_path,
        pretrain=False,
        use_hf=config.data.use_hf,
        load_in_mem=config.data.load_in_mem,
        val_on_test=True,
    )

    if WANDB:
        wandb_logger = WandbLogger(
            project=wandb_proj, log_model=False, name=run_name
        )
    else:
        wandb_logger = None

    has_cuda = torch.cuda.is_available()

    weights_path = (
            Path("checkpoints") / f"{config.config_tag}_{config.tag}" / "weights.pt"
    )

    trainer = Trainer(
        max_epochs=config.train.epochs,
        check_val_every_n_epoch=config.train.val_freq,
        logger=wandb_logger,  # if config.dev else wandb_logger,
        accelerator="auto",
        devices=config.devices if has_cuda else "auto",
        enable_checkpointing=False,
        precision="16-mixed",
        fast_dev_run=config.dev,
        # deterministic=True,
        # callbacks=[ckpt_callback],
        gradient_clip_val=config.train.grad_clip_val,
        gradient_clip_algorithm="norm",
        # limit_train_batches=2,
        # strategy='ddp_find_unused_parameters_true'
    )

    trainer.fit(
        model=framework,
        datamodule=datamodule,
    )

    # if trainer.global_rank == 0:
    callbacks = None
    if config.vis_path is not None:
        if config.seed == 42:
            visualizer = Visualizer(
                res_path=framework.res_path,
                add_vis_keys=[((1, 3), "conf", "Conf")]
                if config.train.use_conf
                else [],
                criterion=None,
                criterion_limit=1,
            )
            callbacks = [visualizer]
        else:
            print(f"Skipping visualisation since seed {config.seed} != 42")

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(framework.model.state_dict(), weights_path)

    test_trainer = Trainer(
        logger=None if config.dev else wandb_logger,
        accelerator="auto",
        devices=[0] if has_cuda else "auto",
        fast_dev_run=config.dev,
        callbacks=callbacks,
    )
    results = test_trainer.test(
        model=framework,
        datamodule=datamodule,
        # ckpt_path=ckpt_path if ckpt_path != "" else None,
    )
    pd.DataFrame(results).to_csv(framework.res_path / "res.csv", index=False)

def main(seed=None):
    parser = get_parser()
    config = parser.parse_args()
    if seed is not None:
        config.seed = seed

    seed_everything(config.seed)

    Path("./logs").mkdir(exist_ok=True)

    data_path = os.environ.get("CD_DATA_ROOT", "../datasets/images")
    wandb_proj = "stm-change-detection"
    config.data.use_hf = False
    config.data.load_in_mem = (
        "hdf5" if os.environ.get("CD_DATA_FORMAT", "images") == "hdf5" else False
    )

    if config.wandb_proj is not None:
        print(
            f"!!! Overriding wandb project '{wandb_proj}' with the one from config: {config.wandb_proj}"
        )
        wandb_proj = config.wandb_proj

    finetune_framework = FinetuneFramework(
        config_namespace=config,
        config=config.as_dict(),
        metrics=MetricCollection(
            {
                "F1": BinaryF1Score(),
                "Recall": BinaryRecall(),
                "Precision": BinaryPrecision(),
                "cIoU": BinaryJaccardIndex(),
                # "mIoU": MulticlassJaccardIndex(num_classes=2, average="macro"),
            }
        ),
    )
    finetune_framework.generate_and_verify_res_path(Path(config.res_path) / str(config.seed))
    finetune(finetune_framework, config, data_path, wandb_proj)

    if WANDB:
        wandb.finish()


if __name__ == "__main__":
    import faulthandler

    faulthandler.enable()
    main()
