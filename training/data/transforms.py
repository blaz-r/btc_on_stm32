import importlib
from enum import Enum

from albumentations.pytorch import ToTensorV2

import albumentations as A


def build_transforms(config, pretrain: bool, test: bool, has_mask: bool = True):
    return DualTransform(config, pretrain, test, has_mask)


class Target(str, Enum):
    I1 = "i1"
    I2 = "i2"
    BOTH_SAME = "both_same"
    BOTH_DIFF = "both_diff"


class DualTransform:
    def __init__(self, config, pretrain: bool, test: bool, has_mask: bool):
        self.img_size = (config.data.img_size, config.data.img_size)
        self.pretrain = pretrain
        self.test = test
        self.has_mask = has_mask
        # always resize
        transforms = [(Target.BOTH_SAME, A.Resize(*self.img_size))]

        # build other transforms instances from config
        transforms.extend(self.build_from_config(config))

        transforms.append((Target.BOTH_SAME, ToTensorV2()))

        assert (
            not test or len(transforms) == 2
        ), "Testing transforms should only contain resize and ToTensor"

        proc_str = "testing" if test else "training"
        print(f"Transforms used for {proc_str}:")
        for tr in transforms:
            print(tr)

        if self.has_mask:
            add = {"imageB": "image", "label": "mask"}
        else:
            add = {"imageB": "image"}

        if all([target == Target.BOTH_SAME for target, _ in transforms]):
            # all same - just make one compose
            self.transforms = [
                (
                    Target.BOTH_SAME,
                    A.Compose(
                        [t_obj for _, t_obj in transforms], additional_targets=add
                    ),
                )
            ]
        else:
            # mixed usage
            add_dict = {
                Target.BOTH_SAME: add,
                # no mask transforms when mixed augs -> unchecked usage: take care to not mess it up
                Target.BOTH_DIFF: {},
                Target.I1: {},
                Target.I2: {},
            }

            composed = []
            for target, t_obj in transforms:
                # make target and transform object pair
                pair = (target, A.Compose(t_obj, additional_targets=add_dict[target]))
                composed.append(pair)

            self.transforms = composed

    def build_from_config(self, config) -> list:
        transforms = []
        if self.pretrain:
            transform_config = config.pretrain.get("transforms", [])
            # emtpy list or not present
            if not transform_config:
                print("No pretrain transforms, using training transforms")
                transform_config = config.train.transforms
        else:
            transform_config = config.train.transforms

        if self.test:
            tr_config_list = []
        else:
            tr_config_list = transform_config[:-1]

        for tr_conf in tr_config_list:
            cls_name, init_args = next(iter(tr_conf.items()))
            if "size" in init_args:
                init_args["size"] = self.img_size

            # backward compatible
            if "target" in init_args:
                target = Target(init_args["target"])
            else:
                target = Target.BOTH_SAME

            tr_cls = getattr(importlib.import_module("albumentations"), cls_name)
            t_obj = tr_cls(**init_args)
            transforms.append((target, t_obj))

        return transforms

    def transform(self, image, imageB, label=None) -> dict:
        if label is not None:
            out_dict = {"image": image, "imageB": imageB, "label": label}
        else:
            out_dict = {"image": image, "imageB": imageB}

        for target, t_obj in self.transforms:
            if target == Target.BOTH_SAME:
                # transformed output is already same as out
                out_dict = t_obj(**out_dict)
            if target == Target.BOTH_DIFF:
                # separately transform both - labels not needed here
                out_dict["image"] = t_obj(image=out_dict["image"])["image"]
                out_dict["imageB"] = t_obj(image=out_dict["imageB"])["image"]
            if target == Target.I1:
                out_dict["image"] = t_obj(image=out_dict["image"])["image"]
            if target == Target.I2:
                out_dict["imageB"] = t_obj(image=out_dict["imageB"])["image"]

        return out_dict

    def __call__(self, data):
        if self.has_mask:
            transformed = self.transform(
                image=data["imageA"], imageB=data["imageB"], label=data["label"]
            )
        else:
            transformed = self.transform(image=data["imageA"], imageB=data["imageB"])

        # transformed["imageA_unnorm"] = data["imageA"]
        # transformed["imageB_unnorm"] = data["imageB"]
        # transformed["img_idx"] = data["img_idx"]
        # # get to 0-1 range and add channel dim
        # transformed["label"] = (transformed["label"] / 255).unsqueeze(0)

        transformed_batch = {
            "imageA": transformed["image"],
            "imageB": transformed["imageB"],
            # "imageA_unnorm": data["imageA"],
            # "imageB_unnorm": data["imageB"],
            # "img_idx": data["img_idx"],
        }
        if self.has_mask:
            transformed_batch["label"] = (transformed["label"] / 255).unsqueeze(0)
        # for iA, iB, lbl in zip(batch["imageA"], batch["imageB"], batch["label"]):
        #
        #
        #     transformed_batch["imageA"].append(transformed["image"])
        #     transformed_batch["imageB"].append(transformed["imageB"])
        #     # get to 0-1 range and add channel dim
        #     transformed_batch["label"].append((transformed["label"] / 255).unsqueeze(0))

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2, 2)
        # axs[1, 0].imshow(transformed_batch["imageA"].permute(1, 2, 0))
        # axs[1, 1].imshow(transformed_batch["imageB"].permute(1, 2, 0))
        #
        # axs[0, 0].imshow(data["imageA"])
        # axs[0, 1].imshow(data["imageB"])
        # plt.show()

        return transformed_batch
