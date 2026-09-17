from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from data.dataset import CDDataset
from data.pt_dataset import CDPTDataset
from data.transforms import build_transforms

from data.hf_datasets import (
    get_oscd,
    get_levircd,
    get_minenetcd,
    get_sysu,
    get_egybcd,
    get_clcd,
    get_gvlm,
)


class CDDataModule(L.LightningDataModule):
    def __init__(
        self,
        config,
        pretrain: bool,
        data_path: str = None,
        use_hf=True,
        load_in_mem=False,
        val_on_test: bool = True,
        all_pairs: bool = False,
    ):
        super().__init__()

        self.config = config
        self.data_path = Path(data_path)
        self.pretrain = pretrain
        self.use_hf = use_hf
        self.load_in_mem = load_in_mem
        self.val_on_test = val_on_test
        self.all_pairs = all_pairs
        self.bands = config.data.bands

        if self.pretrain and config.pretrain.batch_size is not None:
            self.batch_size = config.pretrain.batch_size
        else:
            self.batch_size = config.data.batch_size

        if self.pretrain and config.pretrain.external_data:
            print("Using external data for pretrain")
            self.prepare_external()
        else:
            self.prepare_dataset()

        print(f"Num workers: {config.data.num_workers}")

    def prepare_dataset(self):
        if self.pretrain:
            self.dataset_name = self.config.data.pretrain_dataset
        else:
            self.dataset_name = self.config.data.dataset

        if isinstance(self.dataset_name, list):
            assert self.pretrain, "only pretrain supports multiple datasets"
            assert self.load_in_mem == "hdf5", "Multidata needs to use hdf5 format"
            data = {
                "train": [self.data_path / d / "train" for d in self.dataset_name],
                "test": [self.data_path / d / "test" for d in self.dataset_name],
                "val": [self.data_path / d / "val" for d in self.dataset_name],
            }
        elif self.use_hf:
            data = get_hf_dataset(self.dataset_name, self.data_path)
            data.set_format("numpy")
        else:
            data = {
                "train": self.data_path / self.dataset_name / "train",
                "test": self.data_path / self.dataset_name / "test",
                "val": self.data_path / self.dataset_name / "val",
            }

        train_transforms = build_transforms(
            self.config, pretrain=self.pretrain, test=False
        )
        test_transforms = build_transforms(
            self.config, pretrain=self.pretrain, test=True
        )

        self.train_data = CDDataset(
            data["train"],
            train_transforms,
            use_hf=self.use_hf,
            load_in_mem=self.load_in_mem,
            bands=self.bands,
        )
        self.test_data = CDDataset(
            data["test"],
            test_transforms,
            use_hf=self.use_hf,
            load_in_mem=self.load_in_mem,
            bands=self.bands,
        )

        if self.val_on_test:
            self.val_data = self.test_data
        elif Path(f"{data['val']}.h5").exists():
            self.val_data = CDDataset(
                data["val"],
                test_transforms,
                use_hf=self.use_hf,
                load_in_mem=self.load_in_mem,
                bands=self.bands,
            )
        else:
            print("Validation data not present, using train set as validation.")
            self.val_data = self.train_data

    def prepare_external(self):
        self.dataset_name = self.config.data.pretrain_dataset

        # pretrain on external and eval on main dataset test set
        main_dataset = self.config.data.dataset
        data = {
            "test": self.data_path / main_dataset / "test",
            "pretrain": self.data_path / self.dataset_name / "pretrain",
        }

        train_transforms = build_transforms(
            self.config, pretrain=True, test=False, has_mask=False
        )
        test_transforms = build_transforms(
            self.config, pretrain=True, test=True, has_mask=True
        )

        self.train_data = CDPTDataset(
            data["pretrain"],
            transform=train_transforms,
            all_pairs=self.all_pairs,
            bands=self.bands,
        )
        self.test_data = CDDataset(
            data["test"],
            transform=test_transforms,
            use_hf=self.use_hf,
            load_in_mem=self.load_in_mem,
            bands=self.bands,
        )
        self.val_data = self.test_data

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.config.data.num_workers,
            # pin_memory=self.config.data.pin_memory,
            drop_last=True,
            shuffle=True,
            persistent_workers=self.config.data.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.config.data.num_workers,
            # pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.config.data.num_workers,
            # pin_memory=self.config.data.pin_memory,
            persistent_workers=self.config.data.num_workers > 0,
        )


def get_hf_dataset(name, data_path):
    if name == "oscd":
        ds = get_oscd(data_path)
    elif name == "levir":
        ds = get_levircd(data_path)
    elif name == "minenet":
        ds = get_minenetcd(data_path)
    elif name == "sysu":
        ds = get_sysu(data_path)
    elif name == "egybcd":
        ds = get_egybcd(data_path)
    elif name == "clcd":
        ds = get_clcd(data_path)
    elif name == "gvlm":
        ds = get_gvlm(data_path)
    else:
        raise ValueError(f"Unknown dataset {name}")

    for key in ds:
        ds[key] = ds[key].add_column("img_idx", range(len(ds[key])))

    return ds
