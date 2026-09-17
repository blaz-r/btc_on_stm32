import argparse
import importlib
import inspect
from pathlib import Path

import torch

from configs.config_parser import get_parser
from models.exportable_model import (
    build_change_detection_model,
    collect_deployment_notes,
    count_parameters,
    load_exportable_state_dict,
)


INPUT_NAME = "INPUT"
OUTPUT_NAME = "OUTPUT"


def parse_args():
    parser = argparse.ArgumentParser("Export change detection model to ONNX")
    parser.add_argument("-c", "--config", required=True, help="Training YAML config")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output ONNX path",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional Lightning or plain PyTorch checkpoint to load before export",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset to export. Opset 13 keeps Resize simpler for STM tools.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export batch as a dynamic axis. Fixed batch is safer for STM32N6.",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Dynamo export",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run ONNX checker and compare with onnxruntime when installed",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run ONNX checker and compare with onnxruntime without another conversion",
    )
    return parser.parse_args()


def load_config(config_path: str):
    parser = get_parser()
    return parser.parse_args(["--config", config_path])


def missing_export_packages() -> list[str]:
    missing = []

    try:
        onnx = importlib.import_module("onnx")
    except ImportError:
        onnx = None
    if onnx is None or not hasattr(onnx, "load"):
        missing.append("onnx")

    return missing


def require_export_packages() -> None:
    missing = missing_export_packages()
    if missing:
        raise SystemExit(
            "Missing required ONNX export package(s): "
            + ", ".join(missing)
            + ". Install them, then rerun this command."
        )


def export_onnx(
    model,
    dummy_input,
    output_path: Path,
    opset: int,
    dynamic_batch: bool,
    dynamo: bool
):
    require_export_packages()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}}

    export_kwargs = {
        "input_names": [INPUT_NAME],
        "output_names": [OUTPUT_NAME],
        "opset_version": opset,
        "do_constant_folding": True,
        "dynamic_axes": dynamic_axes,
        "external_data": False,
    }
    # if "dynamo" in inspect.signature(torch.onnx.export).parameters:
    #     export_kwargs["dynamo"] = False

    print(f"ONNX export: legacy exporter, opset={opset}")
    torch.onnx.export(model, (dummy_input,), str(output_path), dynamo=dynamo, **export_kwargs)


def verify_export(model, dummy_input, output_path: Path):
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX checker: OK")

    with torch.no_grad():
        torch_output = model(dummy_input).detach().cpu().numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_output = session.run([OUTPUT_NAME], {INPUT_NAME: dummy_input.cpu().numpy()})[0]
    max_abs_diff = abs(torch_output - ort_output).max()
    print(f"ONNX Runtime max abs diff: {max_abs_diff:.6g}")


def main():
    args = parse_args()
    require_export_packages()

    config = load_config(args.config)
    img_size = args.img_size or config.data.img_size
    device = torch.device(args.device)

    model = build_change_detection_model(config, pretraining=False).to(device)
    model.eval()

    if args.checkpoint is None:
        print("Warning: no checkpoint supplied; exporting initialized weights.")
    else:
        result = load_exportable_state_dict(
            model,
            args.checkpoint,
            strict=not args.non_strict,
        )
        if result.missing_keys or result.unexpected_keys:
            print(f"Missing checkpoint keys: {result.missing_keys}")
            print(f"Unexpected checkpoint keys: {result.unexpected_keys}")

    total, trainable = count_parameters(model)
    enc_total, _ = count_parameters(model.enc)
    print(f"Parameters: total={total / 1e6:.3f}M, trainable={trainable / 1e6:.3f}M")
    print(f"Encoder parameters: {enc_total / 1e6:.3f}M")

    for note in collect_deployment_notes(model, dynamic_batch=args.dynamic_batch):
        print(f"Deployment note: {note}")

    dummy_input = torch.rand(args.batch_size, 6, img_size, img_size, device=device) * 255
    output_path = Path(args.output)

    print("dynamo:", args.dynamo)

    if not args.verify_only:
        with torch.no_grad():
            export_onnx(
                model=model,
                dummy_input=dummy_input,
                output_path=output_path,
                opset=args.opset,
                dynamic_batch=args.dynamic_batch,
                dynamo=args.dynamo
            )

    print(f"Exported ONNX model to {output_path}")

    if args.verify:
        verify_export(model, dummy_input, output_path)


if __name__ == "__main__":
    main()
