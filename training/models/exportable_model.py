from pathlib import Path

import torch
from torch import nn

from models.modules import (
    build_decoder,
    build_diff,
    build_encoder,
    build_out_proc,
    build_pre_diff,
)


class ChangeDetectionModel(nn.Module):
    """
    Tensor-in/tensor-out change detection model.

    The exported inference contract is intentionally simple:
        input:  [N, 6, H, W] where channels are imageA RGB followed by imageB RGB
        output: decoder logits tensor

    Training still runs the encoder on a doubled batch to reuse the existing
    modules. Export runs the shared encoder on each image separately to avoid
    batch-axis concat/slice patterns in the ONNX graph.
    """

    def __init__(
        self,
        enc: nn.Module,
        diff: nn.Module,
        dec: nn.Module,
        pre_diff: nn.Module | None = None,
        out_proc: nn.Module | None = None,
        in_proc: nn.Module | None = None,
        image_channels: int = 3,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.in_proc = in_proc
        self.enc = enc
        self.pre_diff = pre_diff
        self.diff = diff
        self.dec = dec
        self.out_proc = out_proc
        self.image_channels = image_channels
        mean = mean or [0.0] * image_channels
        std = std or [1.0] * image_channels
        self.register_buffer(
            "mean", torch.tensor(mean).view(1, -1, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(std).view(1, -1, 1, 1), persistent=False
        )

    @property
    def input_channels(self) -> int:
        return 2 * self.image_channels

    @staticmethod
    def _is_exporting() -> bool:
        is_onnx_export = getattr(torch.onnx, "is_in_onnx_export", lambda: False)
        return torch.jit.is_tracing() or is_onnx_export()

    def _split_pair(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._is_exporting():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected [N, C, H, W] tensor, got shape {tuple(x.shape)}"
                )
            if x.shape[1] != self.input_channels:
                raise ValueError(
                    f"Expected {self.input_channels} input channels "
                    f"({self.image_channels}+{self.image_channels}), got {x.shape[1]}"
                )

        return x[:, : self.image_channels], x[:, self.image_channels :]

    def _forward_export(self, image_a: torch.Tensor, image_b: torch.Tensor):
        image_a = (image_a / 255.0 - self.mean) / self.std
        image_b = (image_b / 255.0 - self.mean) / self.std
        if self.in_proc is not None:
            image_a = self.in_proc(image_a)
            image_b = self.in_proc(image_b)

        feat_a = self.enc(image_a)
        feat_b = self.enc(image_b)

        if self.pre_diff is not None:
            raise NotImplementedError(
                "pre_diff export without batch packing is not implemented"
            )
        x = (
            self.diff.forward_pair(feat_a, feat_b)
            if self.diff is not None
            else feat_a
        )
        x = self.dec(x)
        if self.out_proc is not None:
            x = self.out_proc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_a, image_b = self._split_pair(x)
        if self._is_exporting():
            return self._forward_export(image_a, image_b)

        x = torch.cat([image_a, image_b], dim=0)
        x = (x / 255.0 - self.mean) / self.std

        if self.in_proc is not None:
            x = self.in_proc(x)
        x = self.enc(x)
        if self.pre_diff is not None:
            x = self.pre_diff(x)
        if self.diff is not None:
            x = self.diff(x)
        x = self.dec(x)
        if self.out_proc is not None:
            x = self.out_proc(x)

        return x


def build_change_detection_model(config, pretraining: bool) -> ChangeDetectionModel:
    transform_config = config.train.transforms
    normalize_args = transform_config[-1]["Normalize"]
    enc = build_encoder(config, pretraining=pretraining)
    dims = enc.get_out_dims()
    strides = enc.get_strides()
    resolutions = [int(config.data.img_size / s) for s in strides]

    diff = build_diff(
        config,
        dims=dims,
        resolutions=resolutions,
        pretraining=pretraining,
        enc=enc,
    )
    dims = diff.adjust_dims(dims)

    dec = build_decoder(
        config,
        input_sizes=dims,
        encoder_strides=strides,
        pretraining=pretraining,
        enc=enc,
    )
    pre_diff = build_pre_diff(
        config,
        dims=dims,
        resolutions=resolutions,
        pretraining=pretraining,
        enc=enc,
    )
    out_proc = build_out_proc(config, pretraining=pretraining)

    return ChangeDetectionModel(
        enc=enc,
        pre_diff=pre_diff,
        diff=diff,
        dec=dec,
        out_proc=out_proc,
        image_channels=len(normalize_args["mean"]),
        mean=normalize_args["mean"],
        std=normalize_args["std"],
    )


def load_exportable_state_dict(
    model: ChangeDetectionModel,
    checkpoint_path: str | Path,
    strict: bool = True,
):
    state_dict = torch.load(Path(checkpoint_path), map_location="cpu")
    return model.load_state_dict(state_dict, strict=strict)


def count_parameters(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _first_modules(
    model: nn.Module,
    module_types: tuple[type[nn.Module], ...],
    limit: int = 5,
) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, module_types):
            names.append(name or "<root>")
            if len(names) >= limit:
                break
    return names


def collect_deployment_notes(model: nn.Module, dynamic_batch: bool = False) -> list[str]:
    notes = [
        "Export with fixed batch and image size for STM32N6 unless you have verified dynamic shape support in the full toolchain.",
        "The exported graph returns logits; keep sigmoid/thresholding outside the model unless the target runtime requires it fused.",
    ]

    if dynamic_batch:
        notes.append(
            "Dynamic batch is higher risk here because the pair is encoded as a doubled batch and split back in the diff stage."
        )

    sync_bn = _first_modules(model, (nn.SyncBatchNorm,))
    if sync_bn:
        notes.append(
            f"SyncBatchNorm found at {', '.join(sync_bn)}; prefer BatchNorm2d before ONNX export and quantization."
        )

    layer_norm = _first_modules(model, (nn.LayerNorm,))
    if layer_norm:
        notes.append(
            f"LayerNorm found at {', '.join(layer_norm)}; this often maps poorly to small NPUs compared with Conv/BatchNorm/ReLU."
        )

    instance_norm = _first_modules(model, (nn.InstanceNorm2d,))
    if instance_norm:
        notes.append(
            f"InstanceNorm found at {', '.join(instance_norm)}; verify quantized kernel support or expect CPU fallback."
        )

    pixel_shuffle = _first_modules(model, (nn.PixelShuffle,))
    if pixel_shuffle:
        notes.append(
            f"PixelShuffle found at {', '.join(pixel_shuffle)}; check that the STM compiler lowers it without fallback."
        )

    adaptive_pool = _first_modules(model, (nn.AdaptiveAvgPool2d,))
    if adaptive_pool:
        notes.append(
            f"AdaptiveAvgPool2d found at {', '.join(adaptive_pool)}; fixed input sizes are safest for deployment."
        )

    dropout = _first_modules(model, (nn.Dropout, nn.Dropout2d))
    if dropout:
        notes.append(
            f"Dropout found at {', '.join(dropout)}; it is disabled in eval export, but keep the model in eval mode."
        )

    notes.append(
        "The thin UNet decoder uses Resize ops for skip alignment and output scaling; profile bilinear Resize on the target NPU and switch to nearest only if deployment requires it."
    )
    return notes
