# STM32N6 firmware

LCD/touchscreen demo for the STM32N6570-DK. `BTC_STM` includes the application,
FSBL, Cube configuration, IDE files, generated network, and demo image arrays.
Restore the ST SDK libraries before building.

Tools used: CubeMX 6.17.0, STM32Cube FW_N6 V1.3.0, CubeIDE 2.2.0, and ST Edge AI 4.0.
See [MODELS.md](../docs/MODELS.md) to download the network files through Git LFS.

Some files were selected by me, some by codex. A clean build of this copy with the SDK restored hasn't been tested yet.

![Example of on board inference](../docs/btc_on_stm.png)

## Build setup

1. Generate an STM32N6570-DK application in STM32Cube AI Studio using the included
   quantized ONNX, with context `Appli` / `FullSecure` (`BYOM` / `ApplicationTemplate`).
   The `.ioc` has the peripheral settings but doesn't recreate the AI integration.
2. Restore `Drivers`, `Middlewares`, `Utilities`, and the AI runtime from that
   project, then apply the sources and IDE files here. Back up custom code before
   regenerating: some of it is outside USER CODE blocks.
3. Import FSBL and Appli into CubeIDE. Check linked files, include paths, and the
   linker script. Some BSP/HAL sources are already in `Application/User/Core`;
   don't add them twice.
4. Update the signing tool path in Appli's post-build command for your installation.
   Follow [load_tricks.md](../docs/load_tricks.md) for signing and flashing.

The included `images.c` has the demo pairs. To replace them, run `host/make_ims.py`
with your own data. If building elsewhere, copy both `images.c` and `images.h`.

If you change the model, regenerate the whole network bundle and check tensor
shapes, quantization, buffer sizes, and weight addresses in the application.

The linker script `STM32CubeIDE/Appli/STM32N657X0HXQ_ROMxspi2.ld` reserves the
800-by-480 RGB565 framebuffer at `0x34000000`; application RAM starts at
`0x340C0000`. Keep the cache handling around NPU and LCD buffers.

## Load the provided binaries

You can also directly load the provided binaries directly into the STM32N6570-DK to just use the app:

```shell
$cp = "C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304\tools\bin\STM32_Programmer_CLI.exe"

$fsbl = "..binaries\STM32N6570-DK_FSBL-Trusted.bin"
$app  = "..binaries\STM32N6570-DK_Appli-Trusted.bin"

```
Connect to the board in SWD mode:
```shell
& $cp -c port=SWD mode=HOTPLUG
```
Program the trusted FSBL at external-flash base:
```shell
& $cp -c port=SWD mode=HOTPLUG -d $fsbl 0x70000000
```
Program the trusted application at its application offset:
```shell
& $cp -c port=SWD mode=HOTPLUG -d $app 0x70100000
```

Then disconnect, power-cycle the board, and return BOOT1 to the normal boot setting.
