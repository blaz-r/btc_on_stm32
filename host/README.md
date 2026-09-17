# Host tools

Run from this directory. Install `requirements.txt` (not needed if you use env from root), then install or expose the
`stm_ai_runner` package supplied with your ST Edge AI installation. Follow the below steps (also in[load_tricks.md](../docs/load_tricks.md)) to
generate and load ST's NPU validation firmware first:

generate c code stuff:
```shell
stedgeai generate -m .\training\quant_model\test-unet-tr_PerChannel_quant_oscd_44_npz_1.onnx --target stm32n6 --st-neural-art
```

load on device with
```shell
python C:\ST\STEdgeAI\4.0\scripts\N6_scripts\n6_loader.py --config-file "C:\ST\STEdgeAI\4.0\scripts\N6_scripts\config_n6l.json"
```

Inside `N6_scripts` folder you need to modify config.json to contain paths to CUBE IDE, and config_n6l.json to point to network.c output from steadgeai and also path to  `STEdgeAI/4.0/Projects/STM32N6570-DK/Applications/NPU_Validation`

Then you can run the inference on device:
```shell
python run.py
```

Use your board's serial port. `run.py` displays image A, image B, the reference
mask, and the target's prediction. It requires `A/<id>.png`, `B/<id>.png`, and
`label/<id>.png`. Inputs must be 96-by-96 RGB. The default serial descriptor
is `serial:921600` (automatic discovery).

Input quantization is `pixel - 128`; output logits are `(q + 69) * 0.062618285`,
followed by sigmoid. These values describe the original model only. Confirm
them with the runner summary for any regenerated model.

For the standalone firmware's built-in image pairs:

```shell
python make_ims.py
```

This reads sample IDs 46, 226, and 364 from `CD_TEST_ROOT`, then writes `images.c`
and `images.h` into the included firmware tree. Edit `SAMPLE_IDS` for different
data (and keep the header's pair count consistent if changing the count).
Demo arrays are already included; this replaces them with your selected images.

`inspect.ipynb` was used for some ViT exploration. `make_ims.ipynb` is the older
single-pair experiment; use `make_ims.py` for the current multi-pair LCD demo.
