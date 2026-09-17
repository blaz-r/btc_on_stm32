import argparse, csv, json, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall

from configs.config_parser import get_parser
from data.dataset import CDDataset
from data.transforms import build_transforms

DEFAULT_MODEL = Path("quant_model/test-unet-tr_PerChannel_quant_oscd_44_npz_1.onnx")
DEFAULT_DATA = Path("../datasets/hdf5")
IN_SCALE, IN_ZERO_POINT = 1.0, -128
OUT_SCALE, OUT_ZERO_POINT = 0.06261828541755676, -69

def parse_args():
    p = argparse.ArgumentParser("Evaluate quantized ONNX CD model with ONNX Runtime")
    p.add_argument("-c", "--config", default="configs/dev.yaml")
    p.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--limit-batches", type=int)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()

def make_dataloader(config_path, data_path, split, batch_size, num_workers):
    config = get_parser().parse_args(["--config", config_path])
    config.data.batch_size = batch_size
    config.data.num_workers = num_workers
    transform = build_transforms(config, pretrain=False, test=True)
    path = data_path / config.data.dataset / split
    ds = CDDataset(path, transform, bands=config.data.bands, use_hf=False, load_in_mem="hdf5")
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False), config

def make_metrics(prefix):
    metrics = {"F1": BinaryF1Score(), "Recall": BinaryRecall(), "Precision": BinaryPrecision(), "cIoU": BinaryJaccardIndex()}
    return MetricCollection(metrics, prefix=f"{prefix}/")

def to_ort_input(batch, input_type):
    x = torch.cat([batch["imageA"], batch["imageB"]], dim=1).numpy().astype(np.float32)
    if "int8" in input_type:
        x = np.clip(np.rint(x / IN_SCALE + IN_ZERO_POINT), -128, 127).astype(np.int8)
    return np.ascontiguousarray(x)

def to_probs(output, output_type):
    if "int8" in output_type:
        output = (output.astype(np.float32) - OUT_ZERO_POINT) * OUT_SCALE
    return torch.sigmoid(torch.from_numpy(output.astype(np.float32)))

def evaluate_dataloader(dataloader, model_path=DEFAULT_MODEL, limit_batches=None, log_every=50, prefix="test"):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    metrics = make_metrics(prefix)
    start = time.perf_counter()
    samples = 0
    for batch_idx, batch in enumerate(dataloader, start=1):
        if limit_batches and batch_idx > limit_batches:
            break
        logits = session.run([out.name], {inp.name: to_ort_input(batch, inp.type)})[0]
        probs, labels = to_probs(logits, out.type), (batch["label"] > 0.5).int()
        metrics.update(probs, labels)
        samples += labels.shape[0]
        if log_every and batch_idx % log_every == 0:
            print(f"Evaluated {batch_idx} batches / {samples} samples")
    result = {k: float(v.cpu()) for k, v in metrics.compute().items()}
    result["seconds"] = time.perf_counter() - start
    result["samples"] = samples
    result["input_type"] = inp.type
    result["output_type"] = out.type
    return result

def write_results(output_dir, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {k: v for k, v in result.items() if "/" in k}
    with (output_dir / "res.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metrics.keys())
        w.writeheader()
        w.writerow(metrics)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

def main():
    args = parse_args()
    dl, config = make_dataloader(args.config, args.data_path, args.split, args.batch_size, args.num_workers)
    name = f"ort_{args.model.stem}"
    out_dir = args.output_dir or Path(config.res_path) / str(config.seed) / config.config_tag / name / config.data.dataset
    result = evaluate_dataloader(dl, args.model, args.limit_batches, args.log_every, args.split)
    write_results(out_dir, result)
    for key, value in result.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

if __name__ == "__main__":
    main()
