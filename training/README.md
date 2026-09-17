# Training and export

Run commands from this directory so `configs`, imports, and output paths resolve
correctly. Code is largely based on our [BTC](https://github.com/blaz-r/BTC-change-detection).

```bash
python -m pip install -r requirements.txt
$env:CD_DATA_ROOT = "../datasets/images"
$env:CD_DATA_FORMAT = "images"
python train.py --config configs/dev.yaml
```

Image datasets follow `<CD_DATA_ROOT>/<dataset>/<split>/{A,B,label}/*.png`, with
matching filenames for each pair and mask. The example config uses `oscd96` and
96-by-96 RGB crops. Supply your own prepared dataset. For the existing HDF5
loader use `CD_DATA_FORMAT=hdf5` and point the root at your HDF5 dataset tree;
see `data/dataset.py` for its required keys and layout. The current training
entry point uses the test split for validation (`val_on_test=True`); account for
this when reporting results.

W&B is disabled by default. Set `WANDB_MODE=online` (and
authenticate) or `offline` to enable it; `wandb_proj` in the YAML selects the
project. Checkpoints are saved at `checkpoints/<config_tag>_<tag>/weights.pt`.

Refer to [BTC](https://github.com/blaz-r/BTC-change-detection) for more details and if you actually just wanna train BTC.


To get onnx: 
```bash
python export_onnx.py --config configs/dev.yaml --checkpoint checkpoints/<run>/weights.pt --output onnx_out/model.onnx --verify
```

Replace `<run>` with the actual checkpoint directory. Omitting `--checkpoint`
exports initialized weights. Keep batch size fixed at 1 for the deployment
workflow. Export takes six channels in `[A_R,A_G,A_B,B_R,B_G,B_B]` order, with
float pixel values in 0–255; inspect `models/exportable_model.py` for preprocessing.

Quantize the exported model in STM32Cube AI Studio/ST Edge AI. The notebook
`notebooks/quant_data.ipynb` shows how to prepare a single-sample NPZ (run it
from `notebooks/`) but that's not really needed. 
The original checkpoints and exact Studio quantization settings aren't included.
The base and quantized ONNX files are included via Git LFS;
see [MODELS.md](../docs/MODELS.md).


This is used to verify that perofrmance stays mostly same:
```bash
python eval_quant_onnx.py --config configs/dev.yaml --model quant_model/model.onnx --data-path ../datasets/hdf5
```

The evaluator uses the HDF5 loader. Its input/output scale and zero point are
constants from the original model; check and update them for your quantized ONNX.
