from itertools import combinations, product
from pathlib import Path

import h5py
import timeit
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CDPTDataset(Dataset):
    def __init__(self, data: str | list[str], bands, transform, all_pairs=False):
        self.transform = transform
        self.bands = bands

        if isinstance(data, list):
            print("loading multiple datasets")
            arr = []
            for d in data:
                arr.extend(self.load_from_hdf5(d))
            self.data = arr
        else:
            self.data = self.load_from_hdf5(data)

        # get number of images per tile
        n_per_tile = len(
            list(k for k in self.data[0] if k.startswith("img") and k != "img_idx")
        )
        if all_pairs:
            # create all possible pairs
            self.pair_per_tile = n_per_tile * (n_per_tile - 1) // 2
            self.pairs = list(combinations(range(n_per_tile), 2))
        else:
            # create pairs of first with all others
            self.pair_per_tile = n_per_tile - 1
            self.pairs = list(product([0], range(1, n_per_tile)))

    def load_from_hdf5(self, path):
        d_name = Path(path).parts[-2]
        split = Path(path).parts[-1]
        print(f"Loading from HDF5 {d_name} {split}")
        t1 = timeit.default_timer()

        file = h5py.File(str(path) + ".h5", "r")
        # load all items in h5 (img1 .. imgN)
        res_dict = {k: np.array(file[k]) for k in file.keys()}
        file.close()

        reindex = (self.bands == "rgb") and (res_dict["img_0"].shape[-1] != 3)
        if reindex:
            new_dict = {}
            for k, v in res_dict.items():
                if len(v.shape) == 4 and v.shape[-1] > 3:
                    # only take image values, first 3 are rgb
                    new_dict[k] = v[..., :3]
                else:
                    new_dict[k] = v
            res_dict = new_dict

        t2 = timeit.default_timer()
        print(f"Load done in {t2 - t1:.2f} seconds")

        ret_list = []
        # change to each image (all samples per tile) into separate dict item in list
        for i in range(len(res_dict["img_0"])):
            item = {
                "d_name": d_name,
            }
            for k, v in res_dict.items():
                item[k] = v[i]
            ret_list.append(item)

        return ret_list

    def get_indices(self, index):
        tile_idx = index // self.pair_per_tile
        i1_idx, i2_idx = self.pairs[
            index % self.pair_per_tile
        ]  # get pair for current index

        return tile_idx, (i1_idx, i2_idx)

    def __getitem__(self, index):
        tile_idx, (i1_idx, i2_idx) = self.get_indices(index)
        data_sample = self.data[tile_idx]

        data = {
            "imageA": data_sample[f"img_{i1_idx}"],
            "imageB": data_sample[f"img_{i2_idx}"],
            "img_idx": data_sample["img_idx"],
        }

        transformed = self.transform(data)
        # save unnormed for visualiser
        transformed["imageA_unnorm"] = data["imageA"]
        transformed["imageB_unnorm"] = data["imageB"]
        transformed["img_idx"] = data["img_idx"]
        if "d_name" in data:
            transformed["d_name"] = data["d_name"]

        return transformed

    def __len__(self):
        return len(self.data) * self.pair_per_tile


if __name__ == "__main__":
    ds = CDPTDataset(
        data=Path("../../datasets/hf/hdf5/seco100k/pretrain"), transform=None
    )
    dl = DataLoader(ds, batch_size=10, shuffle=True)
    print(len(dl))
    next(iter(dl))
