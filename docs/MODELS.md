# Models

Included through Git LFS:

- Base model: `training/onnx_out/test-unet-tr.onnx`
- Quantized model: `training/quant_model/test-unet-tr_PerChannel_quant_oscd_44_npz_1.onnx`
- Cube network: `firmware/BTC_STM/Appli/AI/App/network*` and `stai_network*`
- 

Install Git LFS before cloning. If you already cloned, run:

```powershell
git lfs install --local
git lfs pull
```

Keep the Cube network files together. `network.c` contains the network code;
`network_atonbuf.xSPI2.c` contains its weights and constants. The validation
loader instead flashes a matching `.raw`/`.hex` weights file at `0x71000000`.

The Cube network expects LL_ATON 1.1.3, revision 275. ST runtime libraries aren't
included. Hashes are in [MODEL_SHA256SUMS.txt](MODEL_SHA256SUMS.txt).
